
import pandas as pd

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

df["stress_level"] = pd.to_numeric(
    df["stress_level"],
    errors="coerce"
)

print("=" * 65)
print("HUMANOS INTELLIGENCE")
print("11 - STRESS VALUE AUDIT")
print("=" * 65)

print("\nSTRESS SCORE FREQUENCIES")
print("-" * 65)

print(
    df["stress_level"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\n\nSTRESS SCORES NOT INCLUDED IN THE THREE INSPECTIONS")
print("-" * 65)

unclassified = df[
    ~(
        df["stress_level"].between(0, 1)
        | df["stress_level"].between(2, 3)
        | df["stress_level"].between(4, 5)
    )
]

print("Number of observations:", len(unclassified))

print(
    unclassified[
        ["date", "stress_level", "productivity", "deep_work"]
    ].to_string(index=False)
)

print("\nAUDIT COMPLETE")
print("=" * 65)
