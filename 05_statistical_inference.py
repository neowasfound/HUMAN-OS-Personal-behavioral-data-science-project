
import pandas as pd
import numpy as np
from scipy import stats
import os

print("=" * 65)
print("HUMANOS INTELLIGENCE")
print("05 - COMPARATIVE STATISTICAL INFERENCE")
print("=" * 65)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

os.makedirs("tables", exist_ok=True)

outcomes = ["productivity", "deep_work"]

predictors = [
    "mood",
    "energy",
    "sleep_hours",
    "stress_level",
    "meals_quality_num",
    "social_contact_num",
    "exercise_yn"
]

for col in outcomes + predictors:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nDATA OVERVIEW")
print("-" * 65)
print("Observations:", len(df))
print("Outcomes:", outcomes)
print("Behavioural factors:", predictors)


# ---------------------------------------------------------
# 2. HOLM CORRECTION FUNCTION
# ---------------------------------------------------------

def holm_correction(p_values):

    p_values = np.array(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(n)

    running_max = 0

    for rank, index in enumerate(order):
        adjusted_p = (n - rank) * p_values[index]
        running_max = max(running_max, adjusted_p)
        adjusted[index] = min(running_max, 1.0)

    return adjusted


def format_p(p):

    if pd.isna(p):
        return "NA"

    if p < 0.001:
        return "<0.001"

    return f"{p:.4f}"


# ---------------------------------------------------------
# 3. SPEARMAN CORRELATIONS
# ---------------------------------------------------------

print("\n\n1. SPEARMAN CORRELATION ANALYSIS")
print("=" * 65)

correlation_results = []

for outcome in outcomes:

    print(f"\nOUTCOME: {outcome.upper()}")
    print("-" * 65)

    for predictor in predictors:

        temp = df[[predictor, outcome]].dropna()

        if len(temp) < 3:
            continue

        correlation, p_value = stats.spearmanr(
            temp[predictor],
            temp[outcome]
        )

        correlation_results.append({
            "outcome": outcome,
            "predictor": predictor,
            "observations": len(temp),
            "spearman_r": correlation,
            "p_value": p_value
        })

        print(
            f"{predictor:22s} "
            f"n={len(temp):3d} | "
            f"r={correlation:7.4f} | "
            f"p={format_p(p_value)}"
        )


# Apply Holm correction to the 14 correlation tests

correlation_df = pd.DataFrame(correlation_results)

if len(correlation_df) > 0:

    correlation_df["holm_p"] = holm_correction(
        correlation_df["p_value"]
    )

    correlation_df["significant"] = (
        correlation_df["holm_p"] < 0.05
    )

    correlation_df.to_csv(
        "tables/inference_correlations.csv",
        index=False
    )

    print("\n\nCORRELATION RESULTS AFTER HOLM CORRECTION")
    print("-" * 65)

    for _, row in correlation_df.iterrows():

        print(
            f"{row['outcome']:15s} vs "
            f"{row['predictor']:22s} | "
            f"r={row['spearman_r']:7.4f} | "
            f"adjusted p={format_p(row['holm_p'])} | "
            f"Significant: {row['significant']}"
        )


# ---------------------------------------------------------
# 4. EXERCISE VS NO EXERCISE
# ---------------------------------------------------------

print("\n\n2. EXERCISE GROUP COMPARISONS")
print("=" * 65)

exercise_results = []

for outcome in outcomes:

    temp = df[["exercise_yn", outcome]].dropna()

    exercise_group = temp.loc[
        temp["exercise_yn"] == 1, outcome
    ]

    no_exercise_group = temp.loc[
        temp["exercise_yn"] == 0, outcome
    ]

    if len(exercise_group) == 0 or len(no_exercise_group) == 0:
        continue

    u_stat, p_value = stats.mannwhitneyu(
        exercise_group,
        no_exercise_group,
        alternative="two-sided"
    )

    n1 = len(exercise_group)
    n0 = len(no_exercise_group)

    rank_biserial = (2 * u_stat / (n1 * n0)) - 1

    exercise_results.append({
        "outcome": outcome,
        "exercise_n": n1,
        "no_exercise_n": n0,
        "exercise_mean": exercise_group.mean(),
        "no_exercise_mean": no_exercise_group.mean(),
        "exercise_median": exercise_group.median(),
        "no_exercise_median": no_exercise_group.median(),
        "U_statistic": u_stat,
        "rank_biserial": rank_biserial,
        "p_value": p_value
    })

    print(f"\nOutcome: {outcome}")
    print(f"Exercise days:     n={n1}")
    print(f"No exercise days:  n={n0}")

    print(
        f"Exercise mean:     {exercise_group.mean():.3f}"
    )
    print(
        f"No exercise mean:  {no_exercise_group.mean():.3f}"
    )

    print(
        f"Exercise median:   {exercise_group.median():.3f}"
    )
    print(
        f"No exercise median:{no_exercise_group.median():.3f}"
    )

    print(f"U statistic:       {u_stat:.3f}")
    print(f"Rank-biserial:     {rank_biserial:.3f}")
    print(f"P-value:           {format_p(p_value)}")


exercise_df = pd.DataFrame(exercise_results)

if len(exercise_df) > 0:

    exercise_df["holm_p"] = holm_correction(
        exercise_df["p_value"]
    )

    exercise_df["significant"] = (
        exercise_df["holm_p"] < 0.05
    )

    exercise_df.to_csv(
        "tables/inference_exercise.csv",
        index=False
    )


# ---------------------------------------------------------
# 5. WEEKDAY COMPARISONS
# ---------------------------------------------------------

print("\n\n3. WEEKDAY COMPARISONS")
print("=" * 65)

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_results = []

if "day_of_week" in df.columns:

    for outcome in outcomes:

        groups = []

        for day in weekday_order:

            values = df.loc[
                df["day_of_week"].str.capitalize() == day,
                outcome
            ].dropna()

            if len(values) > 0:
                groups.append(values)

        if len(groups) < 2:
            continue

        h_stat, p_value = stats.kruskal(*groups)

        total_n = sum(len(group) for group in groups)
        k = len(groups)

        epsilon_squared = max(
            0,
            (h_stat - k + 1) / (total_n - k)
        )

        weekday_results.append({
            "outcome": outcome,
            "observations": total_n,
            "groups": k,
            "H_statistic": h_stat,
            "epsilon_squared": epsilon_squared,
            "p_value": p_value
        })

        print(f"\nOutcome: {outcome}")
        print(f"Observations: {total_n}")
        print(f"Weekday groups: {k}")
        print(f"Kruskal-Wallis H: {h_stat:.4f}")
        print(f"Epsilon squared: {epsilon_squared:.4f}")
        print(f"P-value: {format_p(p_value)}")


weekday_df = pd.DataFrame(weekday_results)

if len(weekday_df) > 0:

    weekday_df["holm_p"] = holm_correction(
        weekday_df["p_value"]
    )

    weekday_df["significant"] = (
        weekday_df["holm_p"] < 0.05
    )

    weekday_df.to_csv(
        "tables/inference_weekday.csv",
        index=False
    )


# ---------------------------------------------------------
# 6. FINAL SUMMARY
# ---------------------------------------------------------

print("\n\n4. SUMMARY")
print("=" * 65)

print("\nCorrelation tests:")
if len(correlation_df) > 0:
    print(
        "Significant after Holm correction:",
        int(correlation_df["significant"].sum()),
        "out of",
        len(correlation_df)
    )

print("\nExercise comparisons:")
if len(exercise_df) > 0:
    print(
        "Significant after Holm correction:",
        int(exercise_df["significant"].sum()),
        "out of",
        len(exercise_df)
    )

print("\nWeekday comparisons:")
if len(weekday_df) > 0:
    print(
        "Significant after Holm correction:",
        int(weekday_df["significant"].sum()),
        "out of",
        len(weekday_df)
    )

print("\n\nFILES SAVED")
print("-" * 65)
print("tables/inference_correlations.csv")
print("tables/inference_exercise.csv")
print("tables/inference_weekday.csv")

print("\nANALYSIS COMPLETE")
print("=" * 65)
