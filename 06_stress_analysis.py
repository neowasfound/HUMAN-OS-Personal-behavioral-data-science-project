
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

print("=" * 65)
print("HUMANOS INTELLIGENCE")
print("06 - STRESS AND ACADEMIC PRESSURE ANALYSIS")
print("=" * 65)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

numeric_columns = [
    "stress_level",
    "productivity",
    "deep_work",
    "sleep_hours",
    "mood",
    "energy"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ---------------------------------------------------------
# 2. CREATE CORRECT STRESS GROUPS
# ---------------------------------------------------------

def classify_stress(value):

    if pd.isna(value):
        return np.nan

    elif 0 <= value <= 1:
        return "Low"

    elif 1 < value <= 3.5:
        return "Moderate"

    elif 3.5 < value <= 5:
        return "High"

    else:
        return np.nan


df["stress_group"] = df["stress_level"].apply(classify_stress)

group_order = ["Low", "Moderate", "High"]

df["stress_group"] = pd.Categorical(
    df["stress_group"],
    categories=group_order,
    ordered=True
)

print("\n1. STRESS GROUP DISTRIBUTION")
print("-" * 65)

print(df["stress_group"].value_counts().reindex(group_order))


# ---------------------------------------------------------
# 3. GROUP COMPARISON
# ---------------------------------------------------------

outcomes = [
    "productivity",
    "deep_work",
    "sleep_hours",
    "mood",
    "energy"
]

print("\n\n2. COMPARISON ACROSS STRESS GROUPS")
print("=" * 65)

summary = df.groupby(
    "stress_group",
    observed=False
)[outcomes].agg(["count", "mean", "median", "std"])

print(summary.round(3))

summary.to_csv("tables/stress_group_summary.csv")


# ---------------------------------------------------------
# 4. KRUSKAL-WALLIS TESTS
# ---------------------------------------------------------

print("\n\n3. KRUSKAL-WALLIS TESTS")
print("=" * 65)

test_results = []

for variable in outcomes:

    groups = []

    for group in group_order:

        values = df.loc[
            df["stress_group"] == group,
            variable
        ].dropna()

        groups.append(values)

    # Remove empty groups if any
    valid_groups = [g for g in groups if len(g) > 0]

    H, p = kruskal(*valid_groups)

    n = sum(len(g) for g in valid_groups)
    k = len(valid_groups)

    # Epsilon squared effect size
    epsilon_squared = (H - k + 1) / (n - k)

    epsilon_squared = max(0, epsilon_squared)

    test_results.append({
        "variable": variable,
        "H_statistic": H,
        "p_value": p,
        "epsilon_squared": epsilon_squared,
        "observations": n
    })

    print(f"\nVariable: {variable}")
    print(f"Observations: {n}")
    print(f"H statistic: {H:.4f}")
    print(f"P-value: {p:.6g}")
    print(f"Epsilon squared: {epsilon_squared:.4f}")


# ---------------------------------------------------------
# 5. HOLM CORRECTION
# ---------------------------------------------------------

print("\n\n4. RESULTS AFTER HOLM CORRECTION")
print("=" * 65)

results = pd.DataFrame(test_results)

reject, adjusted_p, _, _ = multipletests(
    results["p_value"],
    method="holm"
)

results["adjusted_p"] = adjusted_p
results["significant"] = reject

for _, row in results.iterrows():

    print(
        f"{row['variable']:15} | "
        f"Adjusted p={row['adjusted_p']:.6g} | "
        f"Significant: {row['significant']}"
    )

results.to_csv(
    "tables/stress_group_tests.csv",
    index=False
)


# ---------------------------------------------------------
# 6. GENERATE FIGURES
# ---------------------------------------------------------

print("\n\n5. GENERATING FIGURES")
print("-" * 65)

for variable in outcomes:

    plot_data = []

    for group in group_order:

        values = df.loc[
            df["stress_group"] == group,
            variable
        ].dropna()

        plot_data.append(values)

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        plot_data,
        tick_labels=group_order
    )

    plt.title(f"{variable.replace('_', ' ').title()} by Stress Group")
    plt.xlabel("Stress Group")
    plt.ylabel(variable.replace("_", " ").title())

    plt.tight_layout()

    plt.savefig(
        f"figures/stress_{variable}.png",
        dpi=300
    )

    plt.close()


# ---------------------------------------------------------
# 7. FINAL SUMMARY
# ---------------------------------------------------------

print("\n\n6. FINAL SUMMARY")
print("=" * 65)

print("\nCorrected stress group sizes:")

print(df["stress_group"].value_counts().reindex(group_order))

print("\nSignificant differences after Holm correction:")

significant_count = results["significant"].sum()

print(f"{significant_count} out of {len(outcomes)} outcomes")

print("\nFILES SAVED")
print("-" * 65)

print("tables/stress_group_summary.csv")
print("tables/stress_group_tests.csv")

for variable in outcomes:
    print(f"figures/stress_{variable}.png")

print("\nANALYSIS COMPLETE")
print("=" * 65)
