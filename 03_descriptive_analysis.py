
import pandas as pd
import numpy as np
import os

print("=" * 60)
print("HUMANOS INTELLIGENCE")
print("03 - DESCRIPTIVE STATISTICAL ANALYSIS")
print("=" * 60)

# load cleaned dataset
file_path = "data/processed/humanos_cleaned.csv"
df = pd.read_csv(file_path)

# convert date
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# create output folder
os.makedirs("tables", exist_ok=True)

# numerical variables
numerical_columns = [
    "mood",
    "energy",
    "sleep_hours",
    "stress_level",
    "productivity",
    "deep_work",
    "overall_rating",
    "meals_quality_num",
    "social_contact_num",
    "exercise_yn"
]

# categorical variables
categorical_columns = [
    "day_of_week",
    "exercise",
    "social_contact",
    "meals_quality",
    "health_issues",
    "exercise_type"
]

print("\n1. DATASET OVERVIEW")
print("-" * 60)

print("Observations:", len(df))
print("Variables:", len(df.columns))
print("Date range:", df["date"].min(), "to", df["date"].max())

# numerical descriptive statistics
print("\n2. NUMERICAL DESCRIPTIVE STATISTICS")
print("-" * 60)

numeric_data = df[numerical_columns]

descriptive = pd.DataFrame({
    "Count": numeric_data.count(),
    "Mean": numeric_data.mean(),
    "Median": numeric_data.median(),
    "Standard Deviation": numeric_data.std(),
    "Variance": numeric_data.var(),
    "Minimum": numeric_data.min(),
    "Q1": numeric_data.quantile(0.25),
    "Q3": numeric_data.quantile(0.75),
    "Maximum": numeric_data.max(),
    "Skewness": numeric_data.skew()
})

descriptive = descriptive.round(3)

print(descriptive.to_string())

descriptive.to_csv("tables/descriptive_statistics.csv")

# categorical frequency distributions
print("\n3. CATEGORICAL FREQUENCY DISTRIBUTIONS")
print("-" * 60)

frequency_tables = []

for column in categorical_columns:

    print("\n", column.upper())
    print("-" * 40)

    frequencies = df[column].value_counts(dropna=False)

    percentages = (
        df[column]
        .value_counts(dropna=False, normalize=True) * 100
    )

    table = pd.DataFrame({
        "Variable": column,
        "Category": frequencies.index.astype(str),
        "Frequency": frequencies.values,
        "Percentage": percentages.values.round(2)
    })

    frequency_tables.append(table)

    print(table[["Category", "Frequency", "Percentage"]].to_string(index=False))

if frequency_tables:
    categorical_summary = pd.concat(
        frequency_tables, ignore_index=True
    )

    categorical_summary.to_csv(
        "tables/categorical_frequencies.csv",
        index=False
    )

# missing values
print("\n4. MISSING VALUE SUMMARY")
print("-" * 60)

missing = pd.DataFrame({
    "Variable": df.columns,
    "Missing": df.isna().sum().values,
    "Percentage": (df.isna().mean().values * 100).round(2)
})

print(missing.to_string(index=False))

missing.to_csv("tables/descriptive_missing_values.csv", index=False)

# outcome comparison
print("\n5. PRODUCTIVITY OUTCOME COMPARISON")
print("-" * 60)

outcomes = ["productivity", "deep_work", "overall_rating"]

for column in outcomes:

    print("\n", column)
    print("Mean:", round(df[column].mean(), 3))
    print("Median:", round(df[column].median(), 3))
    print("Standard deviation:", round(df[column].std(), 3))
    print("Available observations:", df[column].count())

# correlation overview
print("\n6. INITIAL CORRELATION MATRIX")
print("-" * 60)

correlation = numeric_data.corr(method="spearman")

print(correlation.round(2).to_string())

correlation.to_csv("tables/initial_spearman_correlations.csv")

print("\n7. OUTPUT FILES")
print("-" * 60)

print("descriptive_statistics.csv")
print("categorical_frequencies.csv")
print("descriptive_missing_values.csv")
print("initial_spearman_correlations.csv")

print("\nDESCRIPTIVE ANALYSIS COMPLETE")
print("=" * 60)
