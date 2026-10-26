"""Checkpoint 2. Run: pixi run --locked checkpoint2.
Author: Wenqing Zhao. """
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import warnings
import sys
from pathlib import Path

# Avoid bytecode/cache writes outside output and excessive BLAS parallelism.
os.environ.setdefault("MPLCONFIGDIR", str(Path("output/cp2/matplotlib")))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from cp2_utils import (BOUNDS, CATEGORICAL, FORBIDDEN, SPLINE_CONTINUOUS,
                       SPLINE_FEATURES, RestrictedCubicSpline, construct_features)
from cp2_report import write_report
from cp2_coefficients import plot_ridge_coefficients

OUT = Path("output")
DETAIL = OUT / "cp2"
SEED = 7072026
BOOTSTRAPS = 300
MODEL_NAMES = {"ridge": "Ridge logistic", "spline": "Restricted cubic spline logistic", "forest": "Random forest"}
SCORE_NAMES = {**MODEL_NAMES, "SAPS-I": "SAPS-I", "SOFA": "SOFA"}
GRIDS = {
    "ridge": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "spline": {"preprocess__continuous__spline__n_knots": [3, 4], "model__C": [0.1, 1, 10]},
    "forest": {"model__n_estimators": [100], "model__max_depth": [None, 10],
               "model__min_samples_leaf": [5, 20], "model__max_features": ["sqrt", 0.5]},
}


def dump_json(path, content):
    path.write_text(json.dumps(content, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_pipeline(kind, features):
    continuous = SPLINE_CONTINUOUS if kind == "spline" else [c for c in features if c not in CATEGORICAL]
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if kind == "spline":
        numeric_steps.append(("spline", RestrictedCubicSpline()))
    transformers = [("continuous", Pipeline(numeric_steps), continuous),
                    ("categorical", Pipeline([
                        ("impute", SimpleImputer(strategy="constant", fill_value=-999, keep_empty_features=True)),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]), CATEGORICAL)]
    if kind == "spline":
        extra = [c for c in features if c not in SPLINE_CONTINUOUS + CATEGORICAL]
        transformers.append(("indicators", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True), extra))
    steps = [("preprocess", ColumnTransformer(transformers, remainder="drop", sparse_threshold=0))]
    if kind != "forest":
        steps.append(("scale", StandardScaler()))
        estimator = LogisticRegression(solver="lbfgs", max_iter=3000, tol=1e-6, random_state=SEED)
    else:
        estimator = RandomForestClassifier(random_state=SEED, n_jobs=1)
    steps.append(("model", estimator))
    return Pipeline(steps)


def search_model(kind, X, y, splits, label, tuning_records):
    fit_started = time.perf_counter()
    print(f"  {label}: {MODEL_NAMES[kind]} tuning", flush=True)
    search = GridSearchCV(make_pipeline(kind, list(X.columns)), GRIDS[kind],
                          scoring="neg_log_loss", cv=splits, n_jobs=2,
                          refit=True, error_score="raise", return_train_score=False)
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_)
    results.insert(0, "model", kind)
    results.insert(0, "stage", label)
    tuning_records.append(results)
    print(f"    best inner log loss {-search.best_score_:.5f}; {search.best_params_}; {time.perf_counter()-fit_started:.1f} s", flush=True)
    return search.best_estimator_, search.best_params_


def compare_models(oof, common):
    records = []
    for key, name in SCORE_NAMES.items():
        scores = oof.loc[common, key]
        row = {"model": name, "key": key, "n": int(common.sum()),
               "deaths": int(oof.loc[common, "In-hospital_death"].sum())}
        for fold in range(1, 6):
            selected = common & oof.outer_fold.eq(fold)
            assert oof.loc[selected, "In-hospital_death"].nunique() == 2
            row[f"fold_{fold}"] = roc_auc_score(oof.loc[selected, "In-hospital_death"], oof.loc[selected, key])
        fold_scores = [row[f"fold_{fold}"] for fold in range(1, 6)]
        row.update(mean=np.mean(fold_scores), sd=np.std(fold_scores, ddof=1),
                   pooled_oof_auc=roc_auc_score(oof.loc[common, "In-hospital_death"], scores))
        records.append(row)
    return pd.DataFrame(records)


def coefficient_figure(model):
    names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = pd.DataFrame({"predictor": names, "standardized_coefficient": model.named_steps["model"].coef_[0]})
    coefficients.to_csv(DETAIL / "ridge_coefficients.csv", index=False)
    # Presentation only: plot signed Top 20 and four complete supplements.
    plot_ridge_coefficients(coefficients, DETAIL)
    coefficients = coefficients.sort_values("standardized_coefficient")
    return coefficients


def spline_effects(model, X, y):
    """Pairs bootstrap: each replicate refits imputers, knots, scaling and coefficients.

    Pointwise percentile bands on log OR, then exponentiate. Model configuration
    is fixed; bands are conditional on its selection, not simultaneous bands.
    """
    plot_features = [("Age", 65.0), ("HR_mean", 80.0), ("Temp_mean", 37.0)]
    baseline = X.median(numeric_only=True).to_frame().T.reindex(columns=X.columns)
    for category in CATEGORICAL:
        baseline[category] = X[category].mode().iloc[0]
    grids, designs = {}, {}
    for feature, reference in plot_features:
        grid = np.linspace(*X[feature].quantile([0.025, 0.975]), 100)
        design = pd.concat([baseline] * (len(grid) + 1), ignore_index=True)
        design[feature] = np.r_[grid, reference]
        # Hold missingness and all other predictors fixed. Additive model means
        # this log-odds contrast does not depend on the chosen other covariates.
        design[feature + "_missing"] = 0
        grids[feature], designs[feature] = grid, design
    curves = {}
    for feature, _ in plot_features:
        logits = model.decision_function(designs[feature])
        curves[feature] = logits[:-1] - logits[-1]
    rng = np.random.default_rng(SEED + 1000)
    samples = {feature: [] for feature, _ in plot_features}
    for iteration in range(BOOTSTRAPS):
        indices = rng.integers(0, len(X), len(X))
        fitted = clone(model).fit(X.iloc[indices], y.iloc[indices])
        for feature, _ in plot_features:
            logits = fitted.decision_function(designs[feature])
            samples[feature].append(logits[:-1] - logits[-1])
        if (iteration + 1) % 50 == 0:
            print(f"  Spline bootstrap {iteration+1}/{BOOTSTRAPS}", flush=True)
    records, summaries = [], []
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (feature, reference) in zip(axes, plot_features):
        low, high = np.quantile(samples[feature], [0.025, 0.975], axis=0)
        grid, curve = grids[feature], curves[feature]
        ax.plot(grid, np.exp(curve), color="#287c8e")
        ax.fill_between(grid, np.exp(low), np.exp(high), alpha=0.2, color="#287c8e")
        ax.axhline(1, color="grey", linestyle="--")
        ax.set_xlabel(feature)
        ax.set_ylabel(f"Odds ratio relative to {reference:g}")
        ax.set_yscale("log")
        for i, value in enumerate(grid):
            records.append({"predictor": feature, "value": value, "reference": reference,
                            "log_or": curve[i], "or": np.exp(curve[i]),
                            "lower_95": np.exp(low[i]), "upper_95": np.exp(high[i])})
        # A descriptive curvature diagnostic, not a test of nonlinearity.
        chord = np.interp(grid, [grid[0], grid[-1]], [curve[0], curve[-1]])
        summaries.append({"predictor": feature, "reference": reference,
                          "low_value": grid[0], "high_value": grid[-1],
                          "or_low": np.exp(curve[0]), "or_high": np.exp(curve[-1]),
                          "max_log_or_departure_from_chord": float(np.max(np.abs(curve-chord)))})
    fig.suptitle("Figure 2. Restricted cubic spline effects; pointwise 95% bootstrap bands")
    fig.tight_layout()
    fig.savefig(DETAIL / "spline_effects.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(records).to_csv(DETAIL / "spline_effects.csv", index=False)
    return summaries


def main():
    started = time.perf_counter()
    DETAIL.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("error", category=ConvergenceWarning)
    print("Checkpoint 2: building Set A modeling table", flush=True)
    table, dictionary, audit, inputs = construct_features(Path("data"))
    print(f"  Modeling table built in {time.perf_counter()-started:.1f} s", flush=True)
    table.to_csv(DETAIL / "modeling_table.csv", index=False)
    dictionary.to_csv(DETAIL / "feature_dictionary.csv", index=False)
    audit.to_csv(DETAIL / "cleaning_audit.csv", index=False)
    dump_json(DETAIL / "cleaning_rules.json", {"bounds_inclusive": BOUNDS,
        "notes": "Fixed broad analysis assumptions, not validated clinical reference intervals. Negative measurements, -1 codes and invalid categories are missing. Zero HR/BP/PaO2 invalid; zero urine/RespRate valid."})
    all_features = dictionary.predictor.tolist()
    spline_features = SPLINE_FEATURES + [name + "_missing" for name in SPLINE_CONTINUOUS]
    feature_sets = {"ridge": all_features, "spline": spline_features, "forest": all_features}
    assert all(not FORBIDDEN.intersection(columns) for columns in feature_sets.values())
    y = table["In-hospital_death"].astype(int)
    outer = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(table, y))
    oof = table[["RecordID", "In-hospital_death", "SAPS-I", "SOFA"]].copy()
    oof["outer_fold"] = 0
    for key in MODEL_NAMES:
        oof[key] = np.nan
    tuning, configurations, inner_assignments, preprocessing_audit = [], [], [], []
    for fold, (train, test) in enumerate(outer, 1):
        assert not set(table.RecordID.iloc[train]) & set(table.RecordID.iloc[test])
        oof.loc[test, "outer_fold"] = fold
        inner = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED+fold).split(table.iloc[train], y.iloc[train]))
        for inner_fold, (inner_train, inner_test) in enumerate(inner, 1):
            assert not set(train[inner_train]) & set(train[inner_test])
            assert not set(train[inner_train]) & set(test)
            for index in train[inner_test]:
                inner_assignments.append({"outer_fold": fold, "inner_fold": inner_fold, "RecordID": int(table.RecordID.iloc[index])})
        for kind, features in feature_sets.items():
            X = table[features]
            model, params = search_model(kind, X.iloc[train], y.iloc[train], inner, f"outer_{fold}", tuning)
            oof.loc[test, kind] = model.predict_proba(X.iloc[test])[:, 1]
            auc = roc_auc_score(y.iloc[test], oof.loc[test, kind])
            print(f"    held-out fold {fold}: AUROC {auc:.6f}", flush=True)
            configurations.append({"stage": f"outer_{fold}", "model": kind, "params": params, "auc": auc})
            numeric = SPLINE_CONTINUOUS if kind == "spline" else [f for f in features if f not in CATEGORICAL]
            # Direct verification that the refit imputer uses only the outer training data.
            imputer = model.named_steps["preprocess"].named_transformers_["continuous"].named_steps["impute"]
            medians = X.iloc[train][numeric].median().fillna(0).to_numpy()
            np.testing.assert_allclose(imputer.statistics_, medians)
            preprocessing_audit.append({"outer_fold": fold, "model": kind,
                "all_missing_training_features": [c for c in numeric if X.iloc[train][c].isna().all()],
                "constant_training_features": [c for c in numeric if X.iloc[train][c].nunique(dropna=True) <= 1],
                "median_training_only_check": True})
    assert oof.outer_fold.between(1, 5).all() and oof[list(MODEL_NAMES)].notna().all().all()
    assert np.isfinite(oof[list(MODEL_NAMES)]).all().all()
    for key in MODEL_NAMES:
        assert oof[key].between(0, 1).all()
    common = oof[["SAPS-I", "SOFA"]].notna().all(axis=1)
    oof["benchmark_complete"] = common
    oof.to_csv(OUT / "cp2_oof_predictions.csv", index=False)
    oof[["RecordID", "In-hospital_death", "outer_fold"]].to_csv(DETAIL / "outer_folds.csv", index=False)
    pd.DataFrame(inner_assignments).to_csv(DETAIL / "inner_folds.csv", index=False)
    comparison = compare_models(oof, common)
    comparison.to_csv(DETAIL / "comparison_common_sample.csv", index=False)
    full_rows = []
    for key, name in MODEL_NAMES.items():
        values = [roc_auc_score(y[oof.outer_fold.eq(f)], oof.loc[oof.outer_fold.eq(f), key]) for f in range(1, 6)]
        full_rows.append({"model": name, "key": key, "n": len(table), "deaths": int(y.sum()),
                          **{f"fold_{f}": value for f, value in enumerate(values, 1)},
                          "mean": np.mean(values), "sd": np.std(values, ddof=1),
                          "pooled_oof_auc": roc_auc_score(y, oof[key]), "oof_log_loss": log_loss(y, oof[key])})
    full = pd.DataFrame(full_rows)
    full.to_csv(DETAIL / "comparison_all_stays.csv", index=False)
    # Prespecified practical tie rule: within 0.01 AUROC of best, prefer simpler
    # family (ridge, then spline, then forest). Compare families on all Set A.
    threshold = full["mean"].max() - 0.01
    eligible = set(full.loc[full["mean"].ge(threshold), "key"])
    selected = next(key for key in MODEL_NAMES if key in eligible)
    final_models, final_parameters = {}, {}
    full_inner = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED+99).split(table, y))
    # Refit ridge/spline for the required descriptive figures even if not selected.
    for kind in MODEL_NAMES:
        if kind in {"ridge", "spline", selected}:
            final_models[kind], final_parameters[kind] = search_model(kind, table[feature_sets[kind]], y, full_inner, "all_set_a", tuning)
    final_model = final_models[selected]
    joblib.dump(final_model, OUT / "cp2_final_model.joblib", compress=3)
    joblib.dump(final_models["spline"], DETAIL / "spline_model.joblib", compress=3)
    loaded = joblib.load(OUT / "cp2_final_model.joblib")
    np.testing.assert_allclose(final_model.predict_proba(table[feature_sets[selected]]),
                               loaded.predict_proba(table[feature_sets[selected]]), rtol=0, atol=0)
    coefficients = coefficient_figure(final_models["ridge"])
    print("Computing spline confidence bands", flush=True)
    effects = spline_effects(final_models["spline"], table[spline_features], y)
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, name in SCORE_NAMES.items():
        fpr, tpr, _ = roc_curve(y[common], oof.loc[common, key])
        auc = roc_auc_score(y[common], oof.loc[common, key])
        ax.plot(fpr, tpr, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Figure 3. Out-of-fold ROC curves: common benchmark sample")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DETAIL / "roc_comparison.png", dpi=180)
    plt.close(fig)
    pd.concat(tuning, ignore_index=True).to_csv(DETAIL / "tuning_results.csv", index=False)
    dump_json(DETAIL / "outer_selected_parameters.json", configurations)
    dump_json(DETAIL / "preprocessing_audit.json", preprocessing_audit)
    print(f"Saving metadata and assembling PDF; elapsed {time.perf_counter()-started:.1f} s", flush=True)
    metadata = {"seed": SEED, "outer_folds": 5, "inner_folds": 3, "bootstrap_replicates": BOOTSTRAPS,
        "selected_model": selected, "selected_model_name": MODEL_NAMES[selected],
        "selection_rule": "Within 0.01 mean outer AUROC of best on all stays: prefer ridge, then spline, then forest",
        "final_parameters": final_parameters, "feature_sets": feature_sets,
        "encoded_features": {k: len(m.named_steps["model"].feature_importances_) if k == "forest" else len(m.named_steps["model"].coef_[0]) for k, m in final_models.items()},
        "grids": GRIDS, "n_stays": len(table), "deaths": int(y.sum()),
        "benchmark_sample_n": int(common.sum()), "benchmark_sample_deaths": int(y[common].sum()),
        "missing_SAPS_I": int(oof["SAPS-I"].isna().sum()), "missing_SOFA": int(oof.SOFA.isna().sum()),
        "effects": effects, "python": platform.python_version(), "platform": sys.platform,
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib", "reportlab"]},
        "input_sha256": {p.as_posix(): sha256(p) for p in inputs},
        "code_sha256": {p: sha256(p) for p in ["checkpoint2.py", "cp2_utils.py", "cp2_report.py", "pixi.toml", "pixi.lock"]},
        "validation": {"complete_oof_predictions": True, "saved_pipeline_reload_exact": True,
                       "training_only_imputer_checks": True}}
    metadata["runtime_before_report_seconds"] = time.perf_counter() - started
    dump_json(DETAIL / "run_metadata.json", metadata)
    write_report(metadata, table, dictionary, audit, comparison, full, coefficients)
    metadata["total_runtime_seconds"] = time.perf_counter() - started
    dump_json(DETAIL / "run_metadata.json", metadata)
    from pypdf import PdfReader
    reader = PdfReader(OUT / "checkpoint2.pdf")
    assert len(reader.pages) >= 5
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert all(section in text for section in ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "AI-use"])
    print("\nFive-model comparison (same complete-benchmark stays):", flush=True)
    print(comparison[["model", "n", "mean", "sd"]].to_string(index=False), flush=True)
    print(f"Final model: {MODEL_NAMES[selected]}", flush=True)
    print(f"Total runtime: {metadata['total_runtime_seconds']:.1f} s ({metadata['total_runtime_seconds']/60:.2f} min)", flush=True)
    print("Report: output/checkpoint2.pdf", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1), parallel_backend("threading", n_jobs=2):
        main()
