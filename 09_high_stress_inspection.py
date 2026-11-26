
import pandas as pd

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("09 - HIGH STRESS DAY INSPECTION")
print("=" * 70)

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

numeric_columns = [
    "stress_level", "productivity", "deep_work",
    "mood", "energy", "sleep_hours"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# High stress = 4 or 5
high_stress = df[
    df["stress_level"].between(4, 5)
].copy()

high_stress = high_stress.sort_values("date")

print("\nHIGH STRESS DAYS:", len(high_stress))
print("-" * 70)

columns = [
    "date", "stress_level", "productivity",
    "deep_work", "mood", "energy", "sleep_hours",
    "exercise", "social_contact", "meals_quality",
    "health_issues"
]

print(high_stress[columns].to_string(index=False))

# Numerical summary
print("\n\nNUMERICAL SUMMARY")
print("-" * 70)

summary_columns = [
    "productivity", "deep_work",
    "mood", "energy", "sleep_hours"
]

print(high_stress[summary_columns].describe().round(2))

# Score distributions
print("\n\nSTUDY SCORE DISTRIBUTION")
print("-" * 70)

for col in ["productivity", "deep_work"]:

    values = high_stress[col]

    print(f"\n{col.upper()}")
    print("Zero scores:", (values == 0).sum())
    print("Positive scores:", (values > 0).sum())
    print("Missing scores:", values.isna().sum())

# Categorical distributions
print("\n\nCATEGORICAL DISTRIBUTIONS")
print("-" * 70)

for col in [
    "exercise", "social_contact",
    "meals_quality", "health_issues"
]:

    print(f"\n{col.upper()}")
    print(high_stress[col].value_counts(dropna=False).to_string())

# Save results
high_stress[columns].to_csv(
    "tables/high_stress_day_inspection.csv",
    index=False
)

print("\n\nSaved:")
print("tables/high_stress_day_inspection.csv")

print("\nINSPECTION COMPLETE")
print("=" * 70)
