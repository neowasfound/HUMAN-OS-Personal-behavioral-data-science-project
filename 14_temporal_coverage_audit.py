import pandas as pd
import numpy as np

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("14 - TEMPORAL COVERAGE AUDIT")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print("\n1. DATASET OVERVIEW")
print("-" * 70)

print("First observation:", df["date"].min().date())
print("Last observation:", df["date"].max().date())
print("Total observations:", len(df))

total_days = (
    df["date"].max() - df["date"].min()
).days + 1

coverage = len(df) / total_days * 100

print("Calendar days:", total_days)
print(f"Observation coverage: {coverage:.2f}%")

# ---------------------------------------------------------
# 2. OBSERVATION GAPS
# ---------------------------------------------------------

print("\n2. OBSERVATION GAPS")
print("-" * 70)

df["gap_days"] = df["date"].diff().dt.days

gaps = df["gap_days"].dropna()

print("Mean gap:", round(gaps.mean(), 2), "days")
print("Median gap:", round(gaps.median(), 2), "days")
print("Maximum gap:", gaps.max(), "days")

print("\nGap distribution:")

print(
    gaps.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90]
    )
)

# ---------------------------------------------------------
# 3. LARGEST OBSERVATION GAPS
# ---------------------------------------------------------

print("\n3. LARGEST OBSERVATION GAPS")
print("-" * 70)

gap_table = df.loc[
    df["gap_days"].notna(),
    ["date", "gap_days"]
].copy()

gap_table["previous_date"] = (
    df["date"].shift(1)
)

gap_table = gap_table.sort_values(
    "gap_days",
    ascending=False
)

print(
    gap_table[
        ["previous_date", "date", "gap_days"]
    ].head(15).to_string(index=False)
)

# ---------------------------------------------------------
# 4. MONTHLY OBSERVATION COVERAGE
# ---------------------------------------------------------

print("\n4. MONTHLY OBSERVATION COVERAGE")
print("-" * 70)

df["month"] = df["date"].dt.to_period("M")

monthly = df.groupby("month").agg(
    observations=("date", "count"),
    mean_stress=("stress_level", "mean"),
    mean_productivity=("productivity", "mean"),
    mean_deep_work=("deep_work", "mean")
)

monthly["calendar_days"] = (
    monthly.index.days_in_month
)

monthly["coverage_percent"] = (
    monthly["observations"] /
    monthly["calendar_days"] * 100
)

print(monthly.round(2).to_string())

# ---------------------------------------------------------
# 5. OBSERVATION GAP CATEGORIES
# ---------------------------------------------------------

print("\n5. GAP CATEGORIES")
print("-" * 70)

categories = pd.cut(
    gaps,
    bins=[0, 1, 3, 7, 14, np.inf],
    labels=[
        "1 day",
        "2-3 days",
        "4-7 days",
        "8-14 days",
        "15+ days"
    ]
)

print(categories.value_counts().sort_index())

# ---------------------------------------------------------
# 6. SAVE RESULTS
# ---------------------------------------------------------

monthly.to_csv(
    "tables/temporal_coverage_monthly.csv"
)

gap_table.to_csv(
    "tables/temporal_observation_gaps.csv",
    index=False
)

print("\n6. FILES SAVED")
print("-" * 70)

print("tables/temporal_coverage_monthly.csv")
print("tables/temporal_observation_gaps.csv")

print("\nTEMPORAL COVERAGE AUDIT COMPLETE")
