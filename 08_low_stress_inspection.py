
import pandas as pd

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("08 - LOW STRESS DAY INSPECTION")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

# Convert numerical variables
numeric_columns = [
    "stress_level",
    "productivity",
    "deep_work",
    "mood",
    "energy",
    "sleep_hours",
    "exercise_yn"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ---------------------------------------------------------
# 2. SELECT LOW STRESS DAYS
# ---------------------------------------------------------

low_stress = df[
    df["stress_level"].between(0, 1)
].copy()

low_stress = low_stress.sort_values("date")

print("\nLOW STRESS DAYS:", len(low_stress))
print("-" * 70)


# ---------------------------------------------------------
# 3. DISPLAY INDIVIDUAL DAYS
# ---------------------------------------------------------

columns = [
    "date",
    "stress_level",
    "productivity",
    "deep_work",
    "mood",
    "energy",
    "sleep_hours",
    "exercise",
    "social_contact",
    "meals_quality",
    "health_issues"
]

print(
    low_stress[columns].to_string(index=False)
)


# ---------------------------------------------------------
# 4. NUMERICAL SUMMARY
# ---------------------------------------------------------

print("\n\nNUMERICAL SUMMARY")
print("-" * 70)

summary_columns = [
    "productivity",
    "deep_work",
    "mood",
    "energy",
    "sleep_hours"
]

print(
    low_stress[summary_columns].describe().round(2)
)


# ---------------------------------------------------------
# 5. ZERO PRODUCTIVITY AND DEEP WORK
# ---------------------------------------------------------

print("\n\nSTUDY SCORE DISTRIBUTION")
print("-" * 70)

for col in ["productivity", "deep_work"]:

    values = low_stress[col]

    print(f"\n{col.upper()}")

    print("Zero scores:", (values == 0).sum())
    print("Positive scores:", (values > 0).sum())
    print("Missing scores:", values.isna().sum())


# ---------------------------------------------------------
# 6. CATEGORICAL VARIABLES
# ---------------------------------------------------------

print("\n\nCATEGORICAL DISTRIBUTIONS")
print("-" * 70)

for col in [
    "exercise",
    "social_contact",
    "meals_quality",
    "health_issues"
]:

    print(f"\n{col.upper()}")

    print(
        low_stress[col]
        .value_counts(dropna=False)
        .to_string()
    )


# ---------------------------------------------------------
# 7. SAVE INSPECTION
# ---------------------------------------------------------

low_stress[columns].to_csv(
    "tables/low_stress_day_inspection.csv",
    index=False
)

print("\n\nSaved:")
print("tables/low_stress_day_inspection.csv")

print("\nINSPECTION COMPLETE")
print("=" * 70)
