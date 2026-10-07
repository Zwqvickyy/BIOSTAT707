"""BIOSTAT 707 Checkpoint 1: cohort characterization and EDA, set-a.

Run with:  pixi run --locked checkpoint1
No arguments. Writes everything to output/.
"""

# Name: Wenqing (Vicky) Zhao

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #figures


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
EXPECTED_N = 4000

def check_set_a() -> None:
    """Confirm data/set-a/ and data/Outcomes-a.txt are already in place."""
    assert (DATA / "Outcomes-a.txt").exists(), "data/Outcomes-a.txt is missing"
    n = len(list((DATA / "set-a").glob("*.txt")))
    assert n == EXPECTED_N, f"data/set-a has {n} records, expected {EXPECTED_N}"


def load_set_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long data and general descriptor data."""

    frames = []

    for f in sorted((DATA / "set-a").glob("*.txt")):
        df = pd.read_csv(f)

        # Get RecordID from the file name
        df["RecordID"] = int(f.stem)

        frames.append(df)

    # Combine all 4000 files into one long table
    raw = pd.concat(frames, ignore_index=True)

    # RecordID is already a column, so remove the original RecordID row
    long_data = raw[raw["Parameter"] != "RecordID"].copy()

    # General descriptors measured at admission
    static_vars = ["Age", "Gender", "Height", "ICUType", "Weight"]

    static = (
        raw[
            (raw["Time"] == "00:00")
            & raw["Parameter"].isin(static_vars)
        ]
        .pivot_table(
            index="RecordID",
            columns="Parameter",
            values="Value",
            aggfunc="first"
        )
        .reset_index()
    )

    static = static.replace(-1, pd.NA)
    ts = raw[~raw["Parameter"].isin(static_vars)] 
    return ts, static


def main() -> None:
    OUT.mkdir(exist_ok=True)

    check_set_a()

    ts, static = load_set_a()

    outcomes = pd.read_csv(DATA / "Outcomes-a.txt")

    assert len(outcomes) == EXPECTED_N
 # Data wrangling: save the combined long table
    ts.to_csv(OUT / "set-a_long.csv", index=False)

    print("Long data shape:", ts.shape)
    print("Number of records:", ts["RecordID"].nunique())
    print(ts.head())
    

    # 1. Table 1 -> output/table1.csv
        # Merge patient characteristics with outcome
    table1_data = static.merge(
        outcomes[["RecordID", "In-hospital_death"]],
        on="RecordID",
        how="left"
    )
    
    # Make continuous variables numeric
    numeric_vars = ["Age", "Height", "Weight"]

    table1_data[numeric_vars] = (
        table1_data[numeric_vars]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
    )

    # Summary by in-hospital death
    table1 = table1_data.groupby("In-hospital_death").agg(
        N=("RecordID", "count"),
        Age_mean=("Age", "mean"),
        Age_sd=("Age", "std"),
        Height_mean=("Height", "mean"),
        Height_sd=("Height", "std"),
        Weight_mean=("Weight", "mean"),
        Weight_sd=("Weight", "std")
    ).round(2)

    table1.to_csv(OUT / "table1.csv")
    
    
    # 2. Outcome summary -> output/outcomes.csv
    outcome_summary = outcomes["In-hospital_death"].value_counts().sort_index()

    outcome_table = pd.DataFrame({
        "In-hospital_death": outcome_summary.index,
        "Count": outcome_summary.values
    })

    outcome_table["Percent"] = (
        outcome_table["Count"] / len(outcomes) * 100
    ).round(2)

    outcome_table.to_csv(
        OUT / "outcomes.csv",
        index=False
    )

    print(outcome_table)
    
    # check measurement ranges
    measurement_summary = (
        ts[ts["Parameter"] != "RecordID"]
        .groupby("Parameter")["Value"]
        .agg(["count", "min", "median", "max"])
        .reset_index()
    )

    measurement_summary.to_csv(
        OUT / "measurement_ranges.csv",
        index=False
    )

    print(measurement_summary)
    ##So as I generated the measurement summary, I found out that
    ##some values are not reasonable, for example, the min Temperature is -17.8
    ##So how to fix it? I would inspect these values before I completely delete them.
    ##Because some of the extreme values might be still possible in very sick patient.
    
    # wide table: one row per patient
    ts_wide = ts[ts["Parameter"] != "RecordID"].copy()

    wide_summary = (
        ts_wide
        .groupby(["RecordID", "Parameter"])["Value"]
        .agg(["count", "first", "last", "min", "max", "mean"])
        .unstack("Parameter")
    )

    wide_summary.columns = [
        f"{parameter}_{summary}"
        for summary, parameter in wide_summary.columns
    ]

    wide_summary = wide_summary.reset_index()

    wide_data = (
        static
        .merge(wide_summary, on="RecordID", how="left")
        .merge(outcomes, on="RecordID", how="left")
    )

    wide_data.to_csv(
        OUT / "set-a_wide.csv",
        index=False
    )

    print("Wide data shape:", wide_data.shape)
    ###For this checkpoint, I summarized each variable over the full 
    ###48-hour period instead of creating separate summaries for the first and second 24 hours.
    
    # 3. Missingness map -> output/missingness_map.png
    #graph from wide table
    count_cols = [
        col for col in wide_data.columns
        if col.endswith("_count")
    ]

    missing_data = wide_data[count_cols].copy()

    # Make variable names shorter for the plot
    missing_data.columns = [
        col.replace("_count", "")
        for col in missing_data.columns
    ]

    fig, ax = plt.subplots(figsize=(7, 6))

    prop_missing = missing_data.isna().mean().sort_values()

    prop_missing.plot.barh(color="#bd21a0", ax=ax)

    ax.set_xlabel("proportion missing")
    ax.set_title("Set A: proportion missing per variable")

    plt.tight_layout()
    plt.savefig(OUT / "missingness_map.png")
    plt.close()
    
    # 4. Missingness-as-signal -> output/missingness_vs_death.csv
    # 4. Missingness as signal
    count_cols = [
        col for col in wide_data.columns
        if col.endswith("_count")
    ]

    missingness_vs_death = []

    for col in count_cols:
        variable = col.replace("_count", "")

        temp = wide_data[[col, "In-hospital_death"]].copy()
        temp["missing"] = temp[col].isna()

        summary = (
            temp
            .groupby("missing")["In-hospital_death"]
            .agg(["count", "mean"])
            .reset_index()
        )

        summary["Variable"] = variable

        missingness_vs_death.append(summary)

    missingness_vs_death = pd.concat(
        missingness_vs_death,
        ignore_index=True
    )

    missingness_vs_death.to_csv(
        OUT / "missingness_vs_death.csv",
        index=False
    )

    print(missingness_vs_death.head())
    
    print("Checkpoint 1 complete. See output/.")


if __name__ == "__main__":
    main()
    
    
    