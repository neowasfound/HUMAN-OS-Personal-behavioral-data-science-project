import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("20 - DEEP WORK REGRESSION")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

variables = [
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

print("\n1. DATA OVERVIEW")
print("-" * 70)

print("Complete observations:", len(df))

# ---------------------------------------------------------
# 2. DEFINE MODELS
# ---------------------------------------------------------

models = {
    "Model 1 - Stress": [
        "stress_level"
    ],

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

results = {}

# ---------------------------------------------------------
# 3. FIT REGRESSION MODELS
# ---------------------------------------------------------

print("\n2. REGRESSION RESULTS")
print("-" * 70)

for name, predictors in models.items():

    X = sm.add_constant(df[predictors])
    y = df["deep_work"]

    model = sm.OLS(y, X).fit()

    results[name] = model

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Observations:", int(model.nobs))
    print("R-squared:", round(model.rsquared, 4))
    print("Adjusted R-squared:", round(model.rsquared_adj, 4))
    print("F-statistic:", round(model.fvalue, 4))
    print("F-test p-value:", model.f_pvalue)

    print("\nCoefficients:")
    print(model.params.round(4))

    print("\nP-values:")
    print(model.pvalues.round(4))

    print("\nConfidence intervals:")
    print(model.conf_int().round(4))

# ---------------------------------------------------------
# 4. MODEL COMPARISON
# ---------------------------------------------------------

print("\n3. MODEL COMPARISON")
print("-" * 70)

comparison = []

for name, model in results.items():

    comparison.append({
        "model": name,
        "observations": int(model.nobs),
        "R_squared": model.rsquared,
        "adjusted_R_squared": model.rsquared_adj,
        "AIC": model.aic,
        "BIC": model.bic
    })

comparison_df = pd.DataFrame(comparison)

print(comparison_df.round(4).to_string(index=False))

comparison_df.to_csv(
    "tables/deep_work_model_comparison.csv",
    index=False
)

# ---------------------------------------------------------
# 5. MULTICOLLINEARITY
# ---------------------------------------------------------

print("\n4. MULTICOLLINEARITY - VIF")
print("-" * 70)

X = sm.add_constant(
    df[models["Model 3 - Full model"]]
)

vif_results = []

for i, column in enumerate(X.columns):

    if column == "const":
        continue

    vif_results.append({
        "variable": column,
        "VIF": variance_inflation_factor(
            X.values,
            i
        )
    })

vif_df = pd.DataFrame(vif_results)

print(vif_df.round(4).to_string(index=False))

vif_df.to_csv(
    "tables/deep_work_vif.csv",
    index=False
)

# ---------------------------------------------------------
# 6. HC3 ROBUST STANDARD ERRORS
# ---------------------------------------------------------

print("\n5. HC3 ROBUST REGRESSION RESULTS")
print("-" * 70)

for name in [
    "Model 2 - Stress and social contact",
    "Model 3 - Full model"
]:

    model = results[name]

    robust = model.get_robustcov_results(
        cov_type="HC3"
    )

    print("\n" + name)

    robust_table = pd.DataFrame({
        "coefficient": robust.params,
        "robust_SE": robust.bse,
        "p_value": robust.pvalues,
        "CI_lower": robust.conf_int()[:, 0],
        "CI_upper": robust.conf_int()[:, 1]
    }, index=model.model.exog_names)

    print(robust_table.round(4))

    filename = name.split(" - ")[0].lower().replace(" ", "_")

    robust_table.to_csv(
        "tables/" + filename + "_hc3_results.csv"
    )

# ---------------------------------------------------------
# 7. SAVE REGRESSION SUMMARIES
# ---------------------------------------------------------

with open(
    "analysis/deep_work_regression_summary.txt",
    "w"
) as file:

    for name, model in results.items():

        file.write("\n" + "=" * 70 + "\n")
        file.write(name + "\n")
        file.write("=" * 70 + "\n")
        file.write(model.summary().as_text())
        file.write("\n\n")

# ---------------------------------------------------------
# 8. FINISH
# ---------------------------------------------------------

print("\n6. FILES SAVED")
print("-" * 70)

print("tables/deep_work_model_comparison.csv")
print("tables/deep_work_vif.csv")
print("tables/model_2_hc3_results.csv")
print("tables/model_3_hc3_results.csv")
print("analysis/deep_work_regression_summary.txt")

print("\nDEEP WORK REGRESSION COMPLETE")
