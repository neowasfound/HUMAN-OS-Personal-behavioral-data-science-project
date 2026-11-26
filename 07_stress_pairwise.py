
import pandas as pd
import numpy as np
import scikit_posthocs as sp
import os

print("=" * 65)
print("HUMANOS INTELLIGENCE")
print("07 - STRESS PAIRWISE COMPARISONS")
print("=" * 65)


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

os.makedirs("tables", exist_ok=True)

df["stress_level"] = pd.to_numeric(
    df["stress_level"],
    errors="coerce"
)


# ---------------------------------------------------------
# 2. CREATE CORRECTED STRESS GROUPS
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


df["stress_group"] = df["stress_level"].apply(
    classify_stress
)

group_order = ["Low", "Moderate", "High"]

df["stress_group"] = pd.Categorical(
    df["stress_group"],
    categories=group_order,
    ordered=True
)

print("\n1. STRESS GROUP DISTRIBUTION")
print("-" * 65)

print(
    df["stress_group"]
    .value_counts()
    .reindex(group_order)
)


# ---------------------------------------------------------
# 3. DEFINE OUTCOMES
# ---------------------------------------------------------

outcomes = [
    "productivity",
    "deep_work",
    "mood"
]

comparisons = [
    ("Low", "Moderate"),
    ("Low", "High"),
    ("Moderate", "High")
]


# ---------------------------------------------------------
# 4. PAIRWISE DUNN TESTS
# ---------------------------------------------------------

print("\n\n2. PAIRWISE DUNN TESTS")
print("=" * 65)

all_results = []

for outcome in outcomes:

    df[outcome] = pd.to_numeric(
        df[outcome],
        errors="coerce"
    )

    temp = df[
        ["stress_group", outcome]
    ].dropna().copy()

    print("\n")
    print("=" * 65)
    print("OUTCOME:", outcome.upper())
    print("=" * 65)

    print("\nOBSERVATIONS PER GROUP")

    counts = (
        temp.groupby("stress_group", observed=False)[outcome]
        .count()
        .reindex(group_order)
    )

    print(counts)

    # Dunn's test with Holm adjustment
    result = sp.posthoc_dunn(
        temp,
        val_col=outcome,
        group_col="stress_group",
        p_adjust="holm"
    )

    print("\nPAIRWISE ADJUSTED P-VALUES")
    print(result.round(6))

    print("\nINDIVIDUAL COMPARISONS")

    for group1, group2 in comparisons:

        p_value = result.loc[group1, group2]

        all_results.append({
            "outcome": outcome,
            "comparison": f"{group1} vs {group2}",
            "adjusted_p": p_value,
            "significant": p_value < 0.05
        })

        print(
            f"{group1:10s} vs {group2:10s} | "
            f"Adjusted p = {p_value:.6f} | "
            f"Significant = {p_value < 0.05}"
        )


# ---------------------------------------------------------
# 5. SAVE RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    "tables/stress_pairwise_results.csv",
    index=False
)


# ---------------------------------------------------------
# 6. FINAL SUMMARY
# ---------------------------------------------------------

print("\n\n3. FINAL SUMMARY")
print("=" * 65)

for outcome in outcomes:

    subset = results_df[
        results_df["outcome"] == outcome
    ]

    significant = subset["significant"].sum()

    print(
        f"{outcome:15s}: "
        f"{significant} of {len(subset)} comparisons significant"
    )


print("\nFILES SAVED")
print("-" * 65)

print("tables/stress_pairwise_results.csv")

print("\nANALYSIS COMPLETE")
print("=" * 65)
