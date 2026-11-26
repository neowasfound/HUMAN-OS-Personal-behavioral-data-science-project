
import pandas as pd
import numpy as np
import os

print("=" * 55)
print("HUMANOS INTELLIGENCE")
print("01 - INITIAL DATA AUDIT")
print("=" * 55)

# load dataset
file_path = "data/raw/HUMAN_OS_ANALYSIS_READY_SAS_censored.xlsx"

df = pd.read_excel(file_path)

# create output folder
os.makedirs("tables", exist_ok=True)
os.makedirs("report", exist_ok=True)

print("\n1. DATASET OVERVIEW")
print("-" * 55)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

# data types
print("\n2. DATA TYPES")
print("-" * 55)

print(df.dtypes)

# missing values
print("\n3. MISSING VALUES")
print("-" * 55)

missing = pd.DataFrame({
    "Variable": df.columns,
    "Missing": df.isna().sum().values,
    "Percentage": (df.isna().mean().values * 100).round(2)
})

print(missing.to_string(index=False))

missing.to_csv("tables/missing_values.csv", index=False)

# date analysis
print("\n4. DATE ANALYSIS")
print("-" * 55)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

valid_dates = df.dropna(subset=["date"])

print("Valid date records:", len(valid_dates))
print("Invalid date records:", df["date"].isna().sum())

print("Unique dates:", valid_dates["date"].nunique())

print("Duplicate date records:",
      valid_dates["date"].duplicated().sum())

if len(valid_dates) > 0:

    print("First date:", valid_dates["date"].min())
    print("Last date:", valid_dates["date"].max())

# numerical summary
print("\n5. NUMERICAL SUMMARY")
print("-" * 55)

numeric = df.select_dtypes(include=np.number)

summary = numeric.describe().T

print(summary)

summary.to_csv("tables/numerical_summary.csv")

# categorical variables
print("\n6. CATEGORICAL VARIABLES")
print("-" * 55)

categorical = df.select_dtypes(include="object")

for column in categorical.columns:

    print("\n", column)

    print(df[column].value_counts(dropna=False).head(15))

# constant variables
print("\n7. CONSTANT VARIABLES")
print("-" * 55)

for column in df.columns:

    if df[column].nunique(dropna=True) <= 1:

        print(column, "has only one unique value")

# censored values
print("\n8. CENSORED VALUES")
print("-" * 55)

for column in df.columns:

    count = (
        df[column]
        .astype(str)
        .str.contains("CENSORED", case=False, na=False)
        .sum()
    )

    if count > 0:

        print(column, ":", count)

print("\n9. AUDIT COMPLETE")
print("-" * 55)

print("Reports saved in the tables folder.")

print("=" * 55)
