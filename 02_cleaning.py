
import pandas as pd
import numpy as np
import os

print("=" * 55)
print("HUMANOS INTELLIGENCE")
print("02 - DATA CLEANING")
print("=" * 55)

# load original dataset
file_path = "data/raw/HUMAN_OS_ANALYSIS_READY_SAS_censored.xlsx"
df = pd.read_excel(file_path)

# create output folders
os.makedirs("data/processed", exist_ok=True)
os.makedirs("tables", exist_ok=True)

print("\n1. ORIGINAL DATA")
print("-" * 55)

print("Original rows:", len(df))
print("Original columns:", len(df.columns))

# exclude variables not relevant to this investigation
excluded_columns = [
    "water_intake",
    "money_spent",
    "bank_account",
    "employed",
    "employed_num",
    "major_events",
    "best_part_of_day",
    "worst_part_of_day"
]

df = df.drop(columns=excluded_columns, errors="ignore")

print("\n2. EXCLUDED VARIABLES")
print("-" * 55)

for column in excluded_columns:
    print(column)

# standardise column names
df.columns = df.columns.str.strip().str.lower()

# clean date variable
df["date"] = pd.to_datetime(df["date"], errors="coerce")

invalid_dates = df["date"].isna().sum()

print("\n3. DATE CLEANING")
print("-" * 55)

print("Invalid dates:", invalid_dates)

df = df.dropna(subset=["date"])

# remove duplicate dates
duplicate_dates = df["date"].duplicated().sum()

print("Duplicate dates:", duplicate_dates)

df = df.drop_duplicates(subset=["date"], keep="first")

df = df.sort_values("date").reset_index(drop=True)

# clean categorical text
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].astype("string").str.strip()

# standardise day names
df["day_of_week"] = df["day_of_week"].str.lower().str.capitalize()

# standardise meal quality
df["meals_quality"] = df["meals_quality"].str.lower().str.strip()

df["meals_quality"] = df["meals_quality"].replace({
    "good, low cal,": "good",
    " ok": "ok"
})

# standardise social contact
df["social_contact"] = df["social_contact"].str.lower().str.strip()

df["social_contact"] = df["social_contact"].replace({
    "hell yeah": "y"
})

# standardise exercise responses
df["exercise"] = df["exercise"].str.lower().str.strip()

df["exercise_yn"] = df["exercise"].apply(
    lambda x: 1 if isinstance(x, str) and x.startswith("y")
    else (0 if isinstance(x, str) and x.startswith("n") else np.nan)
)

# standardise health issues text
df["health_issues"] = df["health_issues"].str.lower().str.strip()

# missing values overview
print("\n4. MISSING VALUES AFTER CLEANING")
print("-" * 55)

missing = pd.DataFrame({
    "Variable": df.columns,
    "Missing": df.isna().sum().values,
    "Percentage": (df.isna().mean().values * 100).round(2)
})

print(missing.to_string(index=False))

missing.to_csv("tables/cleaned_missing_values.csv", index=False)

# dataset overview
print("\n5. CLEANED DATASET")
print("-" * 55)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nRemaining variables:")
print(df.columns.tolist())

# save cleaned dataset
output_path = "data/processed/humanos_cleaned.csv"

df.to_csv(output_path, index=False)

print("\n6. OUTPUT")
print("-" * 55)

print("Cleaned dataset saved to:", output_path)
print("Missing values report saved to tables folder.")

print("\nDATA CLEANING COMPLETE")
print("=" * 55)
