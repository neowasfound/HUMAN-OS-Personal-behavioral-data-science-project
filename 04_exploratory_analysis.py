
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("HUMANOS INTELLIGENCE")
print("04 - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# load cleaned dataset
df = pd.read_csv("data/processed/humanos_cleaned.csv")

df["date"] = pd.to_datetime(df["date"])

# output folders
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

sns.set_theme(style="whitegrid")

# selected variables
numerical = [
    "mood",
    "energy",
    "sleep_hours",
    "stress_level",
    "productivity",
    "deep_work",
    "overall_rating"
]

# 1. HISTOGRAMS
print("\n1. GENERATING HISTOGRAMS")

for column in numerical:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        bins=10,
        kde=True
    )

    plt.title(f"Distribution of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(f"figures/histogram_{column}.png", dpi=300)
    plt.close()

print("Histograms saved.")

# 2. BOXPLOTS
print("\n2. GENERATING BOXPLOTS")

for column in numerical:

    plt.figure(figsize=(8, 4))

    sns.boxplot(data=df, x=column)

    plt.title(f"Boxplot of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace("_", " ").title())

    plt.tight_layout()

    plt.savefig(f"figures/boxplot_{column}.png", dpi=300)
    plt.close()

print("Boxplots saved.")

# 3. SCATTERPLOTS
print("\n3. GENERATING SCATTERPLOTS")

relationships = [
    ("sleep_hours", "productivity"),
    ("mood", "productivity"),
    ("energy", "productivity"),
    ("stress_level", "productivity"),
    ("social_contact_num", "productivity"),
    ("exercise_yn", "productivity"),
    ("stress_level", "deep_work"),
    ("mood", "overall_rating")
]

for x, y in relationships:

    plot_data = df[[x, y]].dropna()

    plt.figure(figsize=(8, 5))

    sns.regplot(
        data=plot_data,
        x=x,
        y=y,
        lowess=True,
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "red"}
    )

    plt.title(f"{x.replace('_', ' ').title()} vs {y.replace('_', ' ').title()}")
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())

    plt.tight_layout()

    plt.savefig(f"figures/scatter_{x}_{y}.png", dpi=300)
    plt.close()

print("Scatterplots saved.")

# 4. TIME SERIES
print("\n4. GENERATING TIME SERIES PLOTS")

time_variables = [
    "productivity",
    "deep_work",
    "mood",
    "energy",
    "stress_level",
    "sleep_hours"
]

for column in time_variables:

    plot_data = df[["date", column]].dropna().sort_values("date")

    plt.figure(figsize=(12, 5))

    plt.scatter(
        plot_data["date"],
        plot_data[column],
        alpha=0.7
    )

    plt.title(f"{column.replace('_', ' ').title()} Over Time")
    plt.xlabel("Date")
    plt.ylabel(column.replace("_", " ").title())

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(f"figures/timeseries_{column}.png", dpi=300)
    plt.close()

print("Time-series plots saved.")

# 5. CORRELATION HEATMAP
print("\n5. GENERATING CORRELATION HEATMAP")

correlation_variables = [
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

correlation = df[correlation_variables].corr(method="spearman")

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    square=True
)

plt.title("Spearman Correlation Heatmap")

plt.tight_layout()

plt.savefig("figures/correlation_heatmap.png", dpi=300)
plt.close()

print("Correlation heatmap saved.")

# 6. EXERCISE COMPARISON
print("\n6. EXERCISE AND PRODUCTIVITY")

exercise_summary = df.groupby("exercise_yn")[
    ["productivity", "deep_work", "mood", "energy", "stress_level"]
].agg(["count", "mean", "median", "std"])

print(exercise_summary.round(2))

exercise_summary.to_csv("tables/exercise_comparison.csv")

# 7. DAY OF WEEK COMPARISON
print("\n7. DAY OF WEEK AND PRODUCTIVITY")

weekday_summary = df.groupby("day_of_week")[
    ["productivity", "deep_work", "mood", "energy", "stress_level"]
].agg(["count", "mean", "median", "std"])

print(weekday_summary.round(2))

weekday_summary.to_csv("tables/weekday_comparison.csv")

# 8. COMPLETION
print("\n8. ANALYSIS COMPLETE")
print("-" * 60)

print("Figures saved in: figures/")
print("Exercise comparison saved in: tables/")
print("Weekday comparison saved in: tables/")

print("=" * 60)
