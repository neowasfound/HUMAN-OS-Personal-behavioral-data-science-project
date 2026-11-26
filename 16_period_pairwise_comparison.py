
import pandas as pd
import scikit_posthocs as sp

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("16 - PERIOD PAIRWISE COMPARISONS")
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
        return "Period 1"

    elif date <= pd.Timestamp("2026-01-07"):
        return "Period 2"

    elif date <= pd.Timestamp("2026-03-18"):
        return "Period 3"

    else:
        return "Period 4"


df["period"] = df["date"].apply(assign_period)

period_order = [
    "Period 1",
    "Period 2",
    "Period 3",
    "Period 4"
]

variables = [
    "stress_level",
    "productivity",
    "deep_work"
]

all_results = []

# ---------------------------------------------------------
# 3. DUNN PAIRWISE COMPARISONS
# ---------------------------------------------------------

for variable in variables:

    print("\n" + "=" * 70)
    print(variable.upper())
    print("=" * 70)

    data = df[
        ["period", variable]
    ].dropna()

    result = sp.posthoc_dunn(
        data,
        val_col=variable,
        group_col="period",
        p_adjust="holm"
    )

    result = result.reindex(
        index=period_order,
        columns=period_order
    )

    print("\nAdjusted pairwise p-values:")
    print(result.round(6).to_string())

    # -----------------------------------------------------
    # EXTRACT UNIQUE PAIRWISE COMPARISONS
    # -----------------------------------------------------

    for i in range(len(period_order)):

        for j in range(i + 1, len(period_order)):

            period_a = period_order[i]
            period_b = period_order[j]

            p_value = result.loc[
                period_a,
                period_b
            ]

            all_results.append({
                "variable": variable,
                "period_a": period_a,
                "period_b": period_b,
                "holm_p_value": p_value,
                "significant": p_value < 0.05
            })

# ---------------------------------------------------------
# 4. COMBINED RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 70)
print("4. SIGNIFICANT PAIRWISE DIFFERENCES")
print("=" * 70)

print(
    results_df[
        results_df["significant"]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# 5. SAVE RESULTS
# ---------------------------------------------------------

results_df.to_csv(
    "tables/period_pairwise_comparisons.csv",
    index=False
)

print("\nResults saved:")
print("tables/period_pairwise_comparisons.csv")

print("\nPAIRWISE COMPARISONS COMPLETE")
