"""Functions for creating Set A features and restricted cubic splines.
Importing this file does not run any models or create files.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

STATIC = ["Age", "Gender", "Height", "ICUType", "Weight"]
MEASUREMENTS = [
    "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "MAP", "Mg", "NIDiasABP", "NIMAP", "NISysABP",
    "Na", "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2", "SysABP",
    "Temp", "TroponinI", "TroponinT", "WBC", "pH",
]
CATEGORICAL = ["Gender", "ICUType"]
SPLINE_CONTINUOUS = [
    "Age", "HR_mean", "Temp_mean", "CombinedMAP_min", "GCS_min",
    "RespRate_mean", "BUN_max", "Creatinine_max", "Lactate_max",
    "Bilirubin_max", "Platelets_min", "Urine_total",
]
SPLINE_FEATURES = SPLINE_CONTINUOUS + CATEGORICAL + ["MechVent_recorded"]
FORBIDDEN = {"RecordID", "In-hospital_death", "Length_of_stay", "Survival", "SAPS-I", "SOFA"}

# Fixed, deliberately broad plausibility rules, not fitted cutoffs. These are
# analysis assumptions, not claims of validated clinical reference intervals.
BOUNDS = {
    "Age": (0, 120), "Height": (50, 250), "Weight": (1, 400),
    "Temp": (25, 45), "pH": (6.5, 8.0), "FiO2": (0.21, 1),
    "SaO2": (0, 100), "GCS": (3, 15),
}


def clean_values(frame):
    """Return cleaned copy and counts. Impossible values become NaN, never repaired."""
    result = frame.copy()
    value = pd.to_numeric(result["Value"], errors="coerce").astype(float)
    parameter = result["Parameter"]
    reason = pd.Series("", index=result.index)
    reason.loc[~np.isfinite(value)] = "nonfinite_or_nonnumeric"
    reason.loc[value.eq(-1)] = "missing_code_minus_one"
    for name, (low, high) in BOUNDS.items():
        bad = parameter.eq(name) & ((value < low) | (value > high))
        reason.loc[bad & reason.eq("")] = "outside_fixed_bounds"
    for name, codes in {"Gender": [0, 1], "ICUType": [1, 2, 3, 4], "MechVent": [0, 1]}.items():
        bad = parameter.eq(name) & ~value.isin(codes)
        reason.loc[bad & reason.eq("")] = "invalid_category"
    numeric = parameter.isin(MEASUREMENTS + ["Urine"])
    reason.loc[numeric & value.lt(0) & reason.eq("")] = "negative_measurement"
    # Zero HR/BP/arterial oxygen is treated as an invalid observation here;
    # zero urine and respiratory rate are retained (anuria/apnea are possible).
    positive = parameter.isin(["HR", "DiasABP", "MAP", "SysABP", "NIDiasABP", "NIMAP", "NISysABP", "PaO2"])
    reason.loc[positive & value.eq(0)] = "zero_positive_measurement"
    invalid = reason.ne("")
    result["Value"] = value.mask(invalid)
    audit = Counter(zip(parameter[invalid], reason[invalid]))
    return result, audit


def construct_features(data_dir):
    """Read exactly Set A; return one row/stay, predictor dictionary, audit, inputs."""
    data_dir = Path(data_dir)
    files = sorted((data_dir / "set-a").glob("*.txt"))
    assert len(files) == 4000, f"Expected 4000 Set A records, got {len(files)}"
    frames = []
    for file in files:
        frame = pd.read_csv(file)
        assert list(frame.columns) == ["Time", "Parameter", "Value"]
        record = frame.loc[frame.Parameter.eq("RecordID"), "Value"]
        assert len(record) == 1 and int(record.iloc[0]) == int(file.stem)
        frame["RecordID"] = int(file.stem)
        frames.append(frame.loc[~frame.Parameter.eq("RecordID")])
    raw = pd.concat(frames, ignore_index=True)
    parts = raw.Time.str.extract(r"^(\d+):(\d{2})$").astype(float)
    assert parts.notna().all().all() and parts[1].between(0, 59).all()
    raw["Minute"] = parts[0] * 60 + parts[1]
    outside = ~raw.Minute.between(0, 2880)
    audit = Counter({("ALL", "outside_0_to_48_hours"): int(outside.sum())})
    raw, cleaning = clean_values(raw.loc[~outside])
    audit.update(cleaning)
    raw = raw.sort_values(["RecordID", "Minute"], kind="stable")
    ids = pd.Index([int(f.stem) for f in files], name="RecordID")
    features = {}
    descriptions = []

    def add(name, values, group, reason):
        features[name] = values.reindex(ids)
        descriptions.append({"predictor": name, "group": group, "clinical_reason": reason})

    # Admission measurements only; a missing admission descriptor remains missing.
    admission = raw.loc[raw.Minute.eq(0) & raw.Parameter.isin(STATIC)]
    admission = admission.drop_duplicates(["RecordID", "Parameter"], keep="first")
    admission = admission.pivot(index="RecordID", columns="Parameter", values="Value")
    for name in STATIC:
        add(name, admission[name], "Admission", "Baseline demographics, body size, or ICU context")
        if name not in CATEGORICAL:
            add(name + "_missing", features[name].isna().astype(int), "Missingness", "Admission descriptor availability")
    for name in MEASUREMENTS:
        series = raw.loc[raw.Parameter.eq(name)].groupby("RecordID")["Value"]
        summary = series.agg(["first", "last", "min", "max", "mean", "count"])
        for statistic in ["first", "last", "min", "max", "mean"]:
            add(f"{name}_{statistic}", summary[statistic], statistic, {
                "first": "Initial measured state", "last": "Most recent measured state",
                "min": "Low extreme during observation", "max": "High extreme during observation",
                "mean": "Average state during observation",
            }[statistic])
        count = summary["count"].reindex(ids, fill_value=0)
        add(name + "_count", count, "Measurement count", "Monitoring or test ordering intensity")
        add(name + "_missing", count.eq(0).astype(int), "Missingness", "Never validly measured during observation")
    urine = raw.loc[raw.Parameter.eq("Urine")].groupby("RecordID")["Value"]
    add("Urine_total", urine.sum(min_count=1), "Special", "Total recorded output; not normalized to a full stay")
    add("Urine_count", urine.count().reindex(ids, fill_value=0), "Measurement count", "Completeness of urine recording")
    add("Urine_missing", features["Urine_count"].eq(0).astype(int), "Missingness", "Distinguish no measurement from zero urine")
    vent = raw.loc[raw.Parameter.eq("MechVent")].groupby("RecordID")["Value"].max()
    add("MechVent_recorded", vent.reindex(ids).eq(1).astype(int), "Special", "Ventilation recorded; absence does not prove no ventilation")
    map_min = raw.loc[raw.Parameter.isin(["MAP", "NIMAP"])].groupby("RecordID")["Value"].min()
    add("CombinedMAP_min", map_min, "Special", "Lowest recorded invasive or noninvasive mean arterial pressure")
    for name in SPLINE_CONTINUOUS:
        indicator = name + "_missing"
        if indicator not in features:
            add(indicator, features[name].isna().astype(int), "Missingness", "Availability of the selected clinical measurement")
    outcomes_path = data_dir / "Outcomes-a.txt"
    outcomes = pd.read_csv(outcomes_path)
    assert len(outcomes) == 4000 and outcomes.RecordID.is_unique
    assert set(ids) == set(outcomes.RecordID)
    assert outcomes["In-hospital_death"].isin([0, 1]).all()
    # Never bring post-window outcome fields into the modeling table.
    outcomes = outcomes[["RecordID", "In-hospital_death", "SAPS-I", "SOFA"]].copy()
    for name in ["SAPS-I", "SOFA"]:
        value = pd.to_numeric(outcomes[name], errors="coerce")
        upper = 56 if name == "SAPS-I" else 24
        invalid = ~np.isfinite(value) | ~value.between(0, upper)
        audit[(name, "missing_or_invalid_benchmark")] = int(invalid.sum())
        outcomes[name] = value.mask(invalid)
    table = pd.DataFrame(features, index=ids).reset_index().merge(outcomes, on="RecordID", validate="one_to_one")
    assert len(table) == 4000 and table.RecordID.is_unique
    dictionary = pd.DataFrame(descriptions)
    assert not FORBIDDEN.intersection(dictionary.predictor)
    audit_table = pd.DataFrame([
        {"parameter": p, "rule": r, "n_values": n} for (p, r), n in sorted(audit.items())
    ])
    return table, dictionary, audit_table, files + [outcomes_path]


class RestrictedCubicSpline(BaseEstimator, TransformerMixin):
    """Natural/restricted cubic basis with linear tails and training-only knots.

    For knots t1 < ... < tK, keep x plus K-2 contrasts:
    [(x-tj)+^3 - (x-tK-1)+^3*(tK-tj)/(tK-tK-1)
     + (x-tK)+^3*(tK-1-tj)/(tK-tK-1)] / (tK-t1)^2.
    This is cubic within the boundary knots and linear outside them.
    """
    def __init__(self, n_knots=3):
        self.n_knots = n_knots

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        probabilities = [0.1, 0.5, 0.9] if self.n_knots == 3 else [0.05, 0.35, 0.65, 0.95]
        self.knots_ = []
        for column in X.T:
            knots = np.unique(np.quantile(column, probabilities))
            # With extensive median imputation, quantile knots can coincide.
            # Fall back to equally spaced training-support knots, never test data.
            if len(knots) < self.n_knots and column.max() > column.min():
                knots = np.linspace(column.min(), column.max(), self.n_knots)
            self.knots_.append(knots)
        return self

    def transform(self, X):
        check_is_fitted(self, "knots_")
        X = check_array(X)
        assert X.shape[1] == self.n_features_in_
        columns = []
        for x, t in zip(X.T, self.knots_):
            columns.append(x)
            if len(t) < 3:
                continue  # Constant input: linear term only, then scaler handles it.
            penultimate, last = t[-2:]
            for knot in t[:-2]:
                basis = (np.maximum(x-knot, 0)**3
                         - np.maximum(x-penultimate, 0)**3 * (last-knot)/(last-penultimate)
                         + np.maximum(x-last, 0)**3 * (penultimate-knot)/(last-penultimate))
                columns.append(basis / (last-t[0])**2)
        return np.column_stack(columns)

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else [f"x{i}" for i in range(self.n_features_in_)]
        return np.array([label for name, knots in zip(names, self.knots_)
                         for label in [str(name)] + [f"{name}_rcs{j+1}" for j in range(max(0, len(knots)-2))]], dtype=object)
