import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import OLSInfluence

import os

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("21 - DEEP WORK REGRESSION DIAGNOSTICS")
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
    "deep_work",
    "stress_level",
    "social_contact_num",
    "mood",
    "energy",
    "sleep_hours",
    "exercise_yn"
]

df = df[variables].copy()

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.sort_values("date").reset_index(drop=True)

print("\n1. DATA OVERVIEW")
print("-" * 70)

print("Complete observations:", len(df))
print("First observation:", df["date"].min().date())
print("Last observation:", df["date"].max().date())

# ---------------------------------------------------------
# 2. DEFINE MODELS
# ---------------------------------------------------------

models = {
    "Model 2 - Stress and social contact": [
        "stress_level",
        "social_contact_num"
    ],

    "Model 3 - Full model": [
        "stress_level",
        "social_contact_num",
        "mood",
        "energy",
        "sleep_hours",
        "exercise_yn"
    ]
}

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

diagnostic_results = []

# ---------------------------------------------------------
# 3. FIT MODELS
# ---------------------------------------------------------

print("\n2. HC3 ROBUST REGRESSION RESULTS")
print("-" * 70)

for name, predictors in models.items():

    X = sm.add_constant(df[predictors])
    y = df["deep_work"]

    ordinary_model = sm.OLS(y, X).fit()

    robust_model = ordinary_model.get_robustcov_results(
        cov_type="HC3"
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("R-squared:", round(ordinary_model.rsquared, 4))
    print("Adjusted R-squared:", round(ordinary_model.rsquared_adj, 4))

    robust_table = pd.DataFrame({
        "coefficient": robust_model.params,
        "robust_SE": robust_model.bse,
        "t_value": robust_model.tvalues,
        "p_value": robust_model.pvalues,
        "CI_lower": robust_model.conf_int()[:, 0],
        "CI_upper": robust_model.conf_int()[:, 1]
    }, index=X.columns)

    print("\nHC3 ROBUST COEFFICIENTS")
    print(robust_table.round(4))

    filename = (
        name.split(" - ")[0].lower().replace(" ", "_")
        + "_hc3_results.csv"
    )

    robust_table.to_csv("tables/" + filename)

    # -----------------------------------------------------
    # 4. BREUSCH-PAGAN TEST
    # -----------------------------------------------------

    bp = het_breuschpagan(
        ordinary_model.resid,
        ordinary_model.model.exog
    )

    print("\nBREUSCH-PAGAN TEST")
    print("LM statistic:", round(bp[0], 4))
    print("LM p-value:", round(bp[1], 4))
    print("F statistic:", round(bp[2], 4))
    print("F p-value:", round(bp[3], 4))

    # -----------------------------------------------------
    # 5. DURBIN-WATSON TEST
    # -----------------------------------------------------

    dw = durbin_watson(ordinary_model.resid)

    print("\nDURBIN-WATSON STATISTIC")
    print(round(dw, 4))

    print(
        "Note: Observations are ordered by date, "
        "but tracking gaps exist."
    )

    diagnostic_results.append({
        "model": name,
        "R_squared": ordinary_model.rsquared,
        "adjusted_R_squared": ordinary_model.rsquared_adj,
        "BP_LM_p_value": bp[1],
        "BP_F_p_value": bp[3],
        "Durbin_Watson": dw
    })

    # -----------------------------------------------------
    # 6. CALCULATE RESIDUALS AND INFLUENCE
    # -----------------------------------------------------

    residuals = ordinary_model.resid
    fitted = ordinary_model.fittedvalues

    influence = OLSInfluence(ordinary_model)
    cooks_distance = influence.cooks_distance[0]

    threshold = 4 / len(df)

    # -----------------------------------------------------
    # 7. RESIDUALS VS FITTED
    # -----------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.scatter(fitted, residuals, alpha=0.7)

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Fitted deep-work score")
    plt.ylabel("Residuals")
    plt.title(name + " - Residuals vs Fitted")

    plt.tight_layout()

    filename = (
        name.split(" - ")[0].lower().replace(" ", "_")
        + "_residuals_fitted.png"
    )

    plt.savefig("figures/" + filename, dpi=300)
    plt.close()

    # -----------------------------------------------------
    # 8. NORMAL Q-Q PLOT
    # -----------------------------------------------------

    plt.figure(figsize=(8, 6))

    sm.qqplot(
        residuals,
        line="s",
        ax=plt.gca()
    )

    plt.title(name + " - Normal Q-Q Plot")

    plt.tight_layout()

    filename = (
        name.split(" - ")[0].lower().replace(" ", "_")
        + "_qq_plot.png"
    )

    plt.savefig("figures/" + filename, dpi=300)
    plt.close()

    # -----------------------------------------------------
    # 9. RESIDUALS OVER TIME
    # -----------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.scatter(
        df["date"],
        residuals,
        alpha=0.7
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.title(name + " - Residuals Over Time")

    plt.xticks(rotation=45)

    plt.tight_layout()

    filename = (
        name.split(" - ")[0].lower().replace(" ", "_")
        + "_residuals_over_time.png"
    )

    plt.savefig("figures/" + filename, dpi=300)
    plt.close()

    # -----------------------------------------------------
    # 10. COOK'S DISTANCE
    # -----------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.stem(
        range(len(cooks_distance)),
        cooks_distance
    )

    plt.axhline(
        y=threshold,
        linestyle="--"
    )

    plt.xlabel("Observation number (chronological order)")
    plt.ylabel("Cook's distance")
    plt.title(name + " - Influential Observations")

    plt.tight_layout()

    filename = (
        name.split(" - ")[0].lower().replace(" ", "_")
        + "_cooks_distance.png"
    )

    plt.savefig("figures/" + filename, dpi=300)
    plt.close()

    # -----------------------------------------------------
    # 11. IDENTIFY INFLUENTIAL OBSERVATIONS
    # -----------------------------------------------------

    influential = pd.DataFrame({
        "date": df["date"],
        "deep_work": df["deep_work"],
        "fitted": fitted,
        "residual": residuals,
        "cooks_distance": cooks_distance
    })

    influential = influential[
        influential["cooks_distance"] > threshold
    ].sort_values(
        "cooks_distance",
        ascending=False
    )

    print("\nPOTENTIALLY INFLUENTIAL OBSERVATIONS")
    print("Cook's distance threshold:", round(threshold, 4))

    if len(influential) == 0:
        print("No observations exceeded the threshold.")
    else:
        print(influential.round(4).to_string(index=False))

    filename = (
        name.split(" - ")[0].lower().replace(" ", "_")
        + "_influential_observations.csv"
    )

    influential.to_csv(
        "tables/" + filename,
        index=False
    )

# ---------------------------------------------------------
# 12. DIAGNOSTIC SUMMARY
# ---------------------------------------------------------

diagnostics_df = pd.DataFrame(diagnostic_results)

print("\n3. DIAGNOSTIC SUMMARY")
print("-" * 70)

print(diagnostics_df.round(4).to_string(index=False))

diagnostics_df.to_csv(
    "tables/deep_work_diagnostic_summary.csv",
    index=False
)

# ---------------------------------------------------------
# 13. FINISH
# ---------------------------------------------------------

print("\n4. GENERATED FILES")
print("-" * 70)

print("HC3 results saved in tables/")
print("Influential observations saved in tables/")
print("Diagnostic summary saved in tables/")
print("Diagnostic figures saved in figures/")

print("\nDEEP WORK REGRESSION DIAGNOSTICS COMPLETE")
