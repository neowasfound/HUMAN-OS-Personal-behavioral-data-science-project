
import pandas as pd

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("10 - MODERATE STRESS DAY INSPECTION")
print("=" * 70)

# Load data
df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

# Convert numerical columns
numeric_columns = [
    "stress_level", "productivity", "deep_work",
    "mood", "energy", "sleep_hours"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ---------------------------------------------------------
# 1. SELECT MODERATE STRESS DAYS
# ---------------------------------------------------------

moderate_stress = df[
    df["stress_level"].between(1.5,3.5)
].copy()

moderate_stress = moderate_stress.sort_values("date")

print("\nMODERATE STRESS DAYS:", len(moderate_stress))
print("-" * 70)


# ---------------------------------------------------------
# 2. DISPLAY INDIVIDUAL DAYS
# ---------------------------------------------------------

columns = [
    "date", "stress_level", "productivity",
    "deep_work", "mood", "energy", "sleep_hours",
    "exercise", "social_contact", "meals_quality",
    "health_issues"
]

print(moderate_stress[columns].to_string(index=False))


# ---------------------------------------------------------
# 3. NUMERICAL SUMMARY
# ---------------------------------------------------------

print("\n\nNUMERICAL SUMMARY")
print("-" * 70)

summary_columns = [
    "productivity", "deep_work",
    "mood", "energy", "sleep_hours"
]

print(
    moderate_stress[summary_columns].describe().round(2)
)


# ---------------------------------------------------------
# 4. STUDY SCORE DISTRIBUTION
# ---------------------------------------------------------

print("\n\nSTUDY SCORE DISTRIBUTION")
print("-" * 70)

for col in ["productivity", "deep_work"]:

    values = moderate_stress[col]

    print(f"\n{col.upper()}")
    print("Zero scores:", (values == 0).sum())
    print("Positive scores:", (values > 0).sum())
    print("Missing scores:", values.isna().sum())


# ---------------------------------------------------------
# 5. CATEGORICAL DISTRIBUTIONS
# ---------------------------------------------------------

print("\n\nCATEGORICAL DISTRIBUTIONS")
print("-" * 70)

for col in [
    "exercise", "social_contact",
    "meals_quality", "health_issues"
]:

    print(f"\n{col.upper()}")

    print(
        moderate_stress[col]
        .value_counts(dropna=False)
        .to_string()
    )


# ---------------------------------------------------------
# 6. SAVE RESULTS
# ---------------------------------------------------------

moderate_stress[columns].to_csv(
    "tables/moderate_stress_day_inspection.csv",
    index=False
)

print("\n\nSaved:")
print("tables/moderate_stress_day_inspection.csv")

print("\nINSPECTION COMPLETE")
print("=" * 70)
