
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("13 - STRESS RELATIONSHIP SENSITIVITY ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

os.makedirs("tables", exist_ok=True)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in ["stress_level", "productivity", "deep_work"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["date", "stress_level", "productivity"]
).copy()

df["month"] = df["date"].dt.to_period("M")

# ---------------------------------------------------------
# 2. OVERALL CORRELATION
# ---------------------------------------------------------

print("\n1. OVERALL CORRELATION")
print("-" * 70)

def calculate_correlation(data, x, y):

    temp = data[[x, y]].dropna()

    if len(temp) < 3:
        return len(temp), np.nan, np.nan

    rho, p = spearmanr(temp[x], temp[y])

    return len(temp), rho, p


n, rho, p = calculate_correlation(
    df, "stress_level", "productivity"
)

print("Observations:", n)
print(f"Spearman correlation: {rho:.4f}")
print(f"P-value: {p:.6g}")


# ---------------------------------------------------------
# 3. EXCLUDE DECEMBER 2025
# ---------------------------------------------------------

print("\n\n2. SENSITIVITY ANALYSIS: EXCLUDING DECEMBER")
print("=" * 70)

without_december = df[
    df["month"] != pd.Period("2025-12")
].copy()

n_dec, rho_dec, p_dec = calculate_correlation(
    without_december,
    "stress_level",
    "productivity"
)

print("Observations:", n_dec)
print(f"Spearman correlation: {rho_dec:.4f}")
print(f"P-value: {p_dec:.6g}")

print("\nChange in correlation:")
print(f"{rho_dec - rho:+.4f}")


# ---------------------------------------------------------
# 4. MONTHLY CORRELATIONS
# ---------------------------------------------------------

print("\n\n3. MONTHLY CORRELATIONS")
print("=" * 70)

monthly_results = []

for month, group in df.groupby("month"):

    n, rho, p = calculate_correlation(
        group,
        "stress_level",
        "productivity"
    )

    if n < 10:
        print(f"\n{month}: Skipped (only {n} observations)")
        continue

    monthly_results.append({
        "month": str(month),
        "observations": n,
        "spearman_rho": rho,
        "p_value": p
    })

    print(f"\nMonth: {month}")
    print("Observations:", n)
    print(f"Spearman correlation: {rho:.4f}")
    print(f"P-value: {p:.6g}")

monthly_df = pd.DataFrame(monthly_results)

monthly_df.to_csv(
    "tables/monthly_stress_correlations.csv",
    index=False
)


# ---------------------------------------------------------
# 5. LEAVE-ONE-MONTH-OUT ANALYSIS
# ---------------------------------------------------------

print("\n\n4. LEAVE-ONE-MONTH-OUT ANALYSIS")
print("=" * 70)

sensitivity_results = []

for month in sorted(df["month"].dropna().unique()):

    temp = df[df["month"] != month]

    n, rho, p = calculate_correlation(
        temp,
        "stress_level",
        "productivity"
    )

    sensitivity_results.append({
        "excluded_month": str(month),
        "observations": n,
        "spearman_rho": rho,
        "p_value": p
    })

    print(
        f"Exclude {month} | "
        f"N={n} | "
        f"rho={rho:.4f} | "
        f"p={p:.6g}"
    )

sensitivity_df = pd.DataFrame(sensitivity_results)

sensitivity_df.to_csv(
    "tables/leave_one_month_out.csv",
    index=False
)


# ---------------------------------------------------------
# 6. FINAL SUMMARY
# ---------------------------------------------------------

print("\n\n5. FINAL SUMMARY")
print("=" * 70)

print(f"Original correlation: {rho:.4f}" if False else
      f"Original correlation: {calculate_correlation(df, 'stress_level', 'productivity')[1]:.4f}")

print(f"Excluding December:   {rho_dec:.4f}")

print("\nFILES SAVED")
print("-" * 70)

print("tables/monthly_stress_correlations.csv")
print("tables/leave_one_month_out.csv")

print("\nANALYSIS COMPLETE")
print("=" * 70)
