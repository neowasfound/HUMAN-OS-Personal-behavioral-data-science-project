
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.stats import spearmanr

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("12 - TEMPORAL STRESS AND PRODUCTIVITY ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

numeric_columns = [
    "stress_level",
    "productivity",
    "deep_work",
    "mood",
    "energy",
    "sleep_hours"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date"])
df = df.sort_values("date").copy()

print("\n1. DATA OVERVIEW")
print("-" * 70)

print("Observations:", len(df))
print("Start date:", df["date"].min().date())
print("End date:", df["date"].max().date())


# ---------------------------------------------------------
# 2. MONTHLY SUMMARY
# ---------------------------------------------------------

print("\n\n2. MONTHLY SUMMARY")
print("=" * 70)

df["month"] = df["date"].dt.to_period("M")

monthly = df.groupby("month").agg(
    observations=("date", "count"),
    mean_stress=("stress_level", "mean"),
    median_stress=("stress_level", "median"),
    mean_productivity=("productivity", "mean"),
    median_productivity=("productivity", "median"),
    mean_deep_work=("deep_work", "mean"),
    median_deep_work=("deep_work", "median"),
    mean_mood=("mood", "mean"),
    mean_energy=("energy", "mean")
)

print(monthly.round(3))

monthly.to_csv("tables/monthly_temporal_summary.csv")


# ---------------------------------------------------------
# 3. MONTHLY PRODUCTIVITY AND STRESS PLOTS
# ---------------------------------------------------------

print("\n\n3. GENERATING MONTHLY FIGURES")
print("-" * 70)

monthly_plot = monthly.copy()
monthly_plot.index = monthly_plot.index.astype(str)

plt.figure(figsize=(12, 5))

plt.bar(
    monthly_plot.index,
    monthly_plot["mean_productivity"]
)

plt.title("Monthly Average Productivity")
plt.xlabel("Month")
plt.ylabel("Mean Productivity Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(
    "figures/monthly_productivity.png",
    dpi=300
)
plt.close()


plt.figure(figsize=(12, 5))

plt.bar(
    monthly_plot.index,
    monthly_plot["mean_stress"]
)

plt.title("Monthly Average Stress")
plt.xlabel("Month")
plt.ylabel("Mean Stress Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(
    "figures/monthly_stress.png",
    dpi=300
)
plt.close()


# ---------------------------------------------------------
# 4. DAILY OBSERVATIONS OVER TIME
# ---------------------------------------------------------

print("\n\n4. DAILY TEMPORAL PATTERNS")
print("-" * 70)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

variables = [
    ("stress_level", "Stress Level"),
    ("productivity", "Productivity"),
    ("deep_work", "Deep Work")
]

for ax, (variable, title) in zip(axes, variables):

    temp = df.dropna(subset=[variable])

    ax.scatter(
        temp["date"],
        temp[variable],
        alpha=0.65
    )

    ax.set_title(f"{title} Over Time")
    ax.set_ylabel(title)
    ax.set_ylim(-0.3, 5.3)
    ax.grid(alpha=0.2)

axes[-1].set_xlabel("Date")

plt.tight_layout()

plt.savefig(
    "figures/daily_temporal_patterns.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 5. ROLLING 30-DAY AVERAGES
# ---------------------------------------------------------

print("\n\n5. ROLLING 30-DAY ANALYSIS")
print("-" * 70)

rolling_df = df.set_index("date").sort_index()

rolling_variables = [
    "stress_level",
    "productivity",
    "deep_work"
]

rolling_results = pd.DataFrame(index=rolling_df.index)

for variable in rolling_variables:

    rolling_results[variable] = (
        rolling_df[variable]
        .rolling("30D", min_periods=3)
        .mean()
    )

    print(
        f"{variable}: "
        f"{rolling_results[variable].notna().sum()} "
        "rolling observations"
    )


fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for ax, variable in zip(axes, rolling_variables):

    ax.plot(
        rolling_results.index,
        rolling_results[variable]
    )

    ax.set_title(
        f"30-Day Rolling Average: {variable.replace('_', ' ').title()}"
    )

    ax.set_ylabel("Rolling Mean")
    ax.grid(alpha=0.2)

axes[-1].set_xlabel("Date")

plt.tight_layout()

plt.savefig(
    "figures/rolling_30_day_averages.png",
    dpi=300
)

plt.close()

rolling_results.to_csv(
    "tables/rolling_30_day_averages.csv"
)


# ---------------------------------------------------------
# 6. OVERALL TEMPORAL ASSOCIATION
# ---------------------------------------------------------

print("\n\n6. STRESS AND PRODUCTIVITY ASSOCIATION")
print("=" * 70)

paired = df[
    ["stress_level", "productivity", "deep_work"]
].dropna()

print("Complete paired observations:", len(paired))

for outcome in ["productivity", "deep_work"]:

    correlation, p_value = spearmanr(
        paired["stress_level"],
        paired[outcome]
    )

    print(f"\nStress vs {outcome}")
    print(f"Spearman correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.6g}")


# ---------------------------------------------------------
# 7. SAVE FINAL MONTHLY RESULTS
# ---------------------------------------------------------

print("\n\n7. FILES SAVED")
print("-" * 70)

print("tables/monthly_temporal_summary.csv")
print("tables/rolling_30_day_averages.csv")

print("figures/monthly_productivity.png")
print("figures/monthly_stress.png")
print("figures/daily_temporal_patterns.png")
print("figures/rolling_30_day_averages.png")

print("\nTEMPORAL ANALYSIS COMPLETE")
print("=" * 70)
