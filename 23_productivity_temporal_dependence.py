import pandas as pd
import numpy as np
import statsmodels.api as sm

from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import os

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("23 - PRODUCTIVITY TEMPORAL DEPENDENCE")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

variables = [
    "date",
    "productivity",
    "stress_level",
    "social_contact_num"
]

df = df[variables].copy()

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.sort_values("date").reset_index(drop=True)

print("\n1. DATA OVERVIEW")
print("-" * 70)

print("Complete observations:", len(df))
print("First date:", df["date"].min().date())
print("Last date:", df["date"].max().date())

# ---------------------------------------------------------
# 2. FIT PRIMARY PRODUCTIVITY MODEL
# ---------------------------------------------------------

X = sm.add_constant(
    df[["stress_level", "social_contact_num"]]
)

y = df["productivity"]

model = sm.OLS(y, X).fit()

df["fitted"] = model.fittedvalues
df["residual"] = model.resid

print("\n2. PRIMARY MODEL")
print("-" * 70)

print("R-squared:", round(model.rsquared, 4))
print("Observations:", int(model.nobs))

# ---------------------------------------------------------
# 3. MATCH RESIDUALS BY ACTUAL CALENDAR-DAY LAGS
# ---------------------------------------------------------

print("\n3. CALENDAR-DAY RESIDUAL CORRELATIONS")
print("-" * 70)

lags = [1, 2, 3, 7]

results = []

for lag in lags:

    earlier = df[["date", "residual"]].copy()

    later = df[["date", "residual"]].copy()

    earlier["date"] = earlier["date"] + pd.Timedelta(days=lag)

    pairs = earlier.merge(
        later,
        on="date",
        suffixes=("_earlier", "_later")
    )

    n = len(pairs)

    if n >= 3:

        pearson_r, pearson_p = pearsonr(
            pairs["residual_earlier"],
            pairs["residual_later"]
        )

        spearman_r, spearman_p = spearmanr(
            pairs["residual_earlier"],
            pairs["residual_later"]
        )

    else:

        pearson_r = np.nan
        pearson_p = np.nan
        spearman_r = np.nan
        spearman_p = np.nan

    results.append({
        "lag_days": lag,
        "valid_pairs": n,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p
    })

results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# 4. MULTIPLE TESTING CORRECTION
# ---------------------------------------------------------

valid = results_df["pearson_p"].notna()

results_df["pearson_p_holm"] = np.nan

if valid.sum() > 0:

    adjusted = multipletests(
        results_df.loc[valid, "pearson_p"],
        method="holm"
    )[1]

    results_df.loc[
        valid,
        "pearson_p_holm"
    ] = adjusted

print(results_df.round(4).to_string(index=False))

# ---------------------------------------------------------
# 5. OBSERVATION GAP DISTRIBUTION
# ---------------------------------------------------------

print("\n4. OBSERVATION GAPS")
print("-" * 70)

df["gap_days"] = df["date"].diff().dt.days

print(df["gap_days"].value_counts().sort_index().head(15))

print("\nMedian gap:", df["gap_days"].median())
print("Maximum gap:", df["gap_days"].max())

# ---------------------------------------------------------
# 6. RESIDUALS OVER TIME
# ---------------------------------------------------------

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

plt.figure(figsize=(12, 6))

plt.scatter(
    df["date"],
    df["residual"],
    alpha=0.7
)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Date")
plt.ylabel("Productivity regression residual")
plt.title("Productivity Model 2 - Residuals Over Calendar Time")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "figures/productivity_temporal_residuals.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# 7. SAVE RESULTS
# ---------------------------------------------------------

results_df.to_csv(
    "tables/productivity_temporal_dependence.csv",
    index=False
)

df.to_csv(
    "tables/productivity_temporal_residuals.csv",
    index=False
)

print("\n5. FILES SAVED")
print("-" * 70)

print("tables/productivity_temporal_dependence.csv")
print("tables/productivity_temporal_residuals.csv")
print("figures/productivity_temporal_residuals.png")

print("\nPRODUCTIVITY TEMPORAL DEPENDENCE COMPLETE")
