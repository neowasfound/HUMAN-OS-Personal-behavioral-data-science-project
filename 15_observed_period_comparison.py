import pandas as pd
import numpy as np
from scipy.stats import kruskal

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("15 - OBSERVED PERIOD COMPARISON")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

df["date"] = pd.to_datetime(df["date"])

# ---------------------------------------------------------
# 2. DEFINE OBSERVED PERIODS
# ---------------------------------------------------------

def assign_period(date):

    if date <= pd.Timestamp("2025-11-06"):
        return "Period 1: Aug-Nov 2025"

    elif date <= pd.Timestamp("2026-01-07"):
        return "Period 2: Dec-Jan"

    elif date <= pd.Timestamp("2026-03-18"):
        return "Period 3: March 2026"

    else:
        return "Period 4: May 2026"


df["period"] = df["date"].apply(assign_period)

period_order = [
    "Period 1: Aug-Nov 2025",
    "Period 2: Dec-Jan",
    "Period 3: March 2026",
    "Period 4: May 2026"
]

df["period"] = pd.Categorical(
    df["period"],
    categories=period_order,
    ordered=True
)

# ---------------------------------------------------------
# 3. PERIOD SAMPLE SIZES
# ---------------------------------------------------------

print("\n1. PERIOD SAMPLE SIZES")
print("-" * 70)

print(df["period"].value_counts().sort_index())

# ---------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------

variables = [
    "stress_level",
    "productivity",
    "deep_work",
    "mood",
    "energy"
]

print("\n2. DESCRIPTIVE STATISTICS")
print("-" * 70)

summary = df.groupby(
    "period",
    observed=False
)[variables].agg(
    ["count", "mean", "median", "std"]
)

print(summary.round(3).to_string())

summary.to_csv(
    "tables/observed_period_summary.csv"
)

# ---------------------------------------------------------
# 5. KRUSKAL-WALLIS TESTS
# ---------------------------------------------------------

print("\n3. KRUSKAL-WALLIS TESTS")
print("-" * 70)

results = []

for variable in variables:

    groups = []

    for period in period_order:

        values = df.loc[
            df["period"] == period,
            variable
        ].dropna()

        if len(values) > 0:
            groups.append(values)

    if len(groups) < 2:
        continue

    statistic, p_value = kruskal(*groups)

    n = sum(len(group) for group in groups)
    k = len(groups)

    epsilon_squared = (
        statistic - k + 1
    ) / (n - k)

    epsilon_squared = max(0, epsilon_squared)

    results.append({
        "variable": variable,
        "H_statistic": statistic,
        "p_value": p_value,
        "epsilon_squared": epsilon_squared,
        "observations": n
    })

    print(f"\n{variable}")
    print("H statistic:", round(statistic, 4))
    print("P-value:", round(p_value, 6))
    print("Epsilon squared:", round(epsilon_squared, 4))
    print("Observations:", n)

# ---------------------------------------------------------
# 6. HOLM CORRECTION
# ---------------------------------------------------------

print("\n4. MULTIPLE TESTING CORRECTION")
print("-" * 70)

results_df = pd.DataFrame(results)

# Holm adjustment
p_values = results_df["p_value"].values

order = np.argsort(p_values)

adjusted = np.empty(len(p_values))

for rank, index in enumerate(order):

    adjusted[index] = min(
        1,
        (len(p_values) - rank) * p_values[index]
    )

# Enforce monotonicity
for i in range(1, len(order)):

    adjusted[order[i]] = max(
        adjusted[order[i]],
        adjusted[order[i - 1]]
    )

results_df["holm_p_value"] = adjusted

print(
    results_df[
        [
            "variable",
            "p_value",
            "holm_p_value",
            "epsilon_squared"
        ]
    ].round(4).to_string(index=False)
)

# ---------------------------------------------------------
# 7. SAVE RESULTS
# ---------------------------------------------------------

results_df.to_csv(
    "tables/observed_period_tests.csv",
    index=False
)

print("\n5. FILES SAVED")
print("-" * 70)

print("tables/observed_period_summary.csv")
print("tables/observed_period_tests.csv")

print("\nOBSERVED PERIOD COMPARISON COMPLETE")
