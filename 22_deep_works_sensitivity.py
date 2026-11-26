
import pandas as pd
import numpy as np
import statsmodels.api as sm

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("22 - DEEP WORK INFLUENCE SENSITIVITY")
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
    "social_contact_num"
]

df = df[variables].copy()

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.sort_values("date").reset_index(drop=True)

print("\n1. DATA OVERVIEW")
print("-" * 70)

print("Complete observations:", len(df))

# ---------------------------------------------------------
# 2. DEFINE MODEL
# ---------------------------------------------------------

predictors = [
    "stress_level",
    "social_contact_num"
]

# Influential dates identified in Script 21
influential_dates = pd.to_datetime([
    "2025-09-03",
    "2025-09-02",
    "2025-10-14"
])

# ---------------------------------------------------------
# 3. FUNCTION TO FIT HC3 MODEL
# ---------------------------------------------------------

def fit_model(data, scenario):

    X = sm.add_constant(data[predictors])
    y = data["deep_work"]

    model = sm.OLS(y, X).fit()

    robust = model.get_robustcov_results(
        cov_type="HC3"
    )

    results = {
        "scenario": scenario,
        "observations": int(model.nobs),
        "R_squared": model.rsquared,
        "adjusted_R_squared": model.rsquared_adj
    }

    for i, variable in enumerate(X.columns):

        results[variable + "_coef"] = robust.params[i]
        results[variable + "_SE"] = robust.bse[i]
        results[variable + "_p"] = robust.pvalues[i]

        results[variable + "_CI_lower"] = robust.conf_int()[i, 0]
        results[variable + "_CI_upper"] = robust.conf_int()[i, 1]

    return results


# ---------------------------------------------------------
# 4. ORIGINAL MODEL
# ---------------------------------------------------------

print("\n2. ORIGINAL MODEL")
print("-" * 70)

results = []

original = fit_model(
    df,
    "Original dataset"
)

results.append(original)

# ---------------------------------------------------------
# 5. REMOVE EACH INFLUENTIAL OBSERVATION
# ---------------------------------------------------------

print("\n3. INDIVIDUAL INFLUENCE SENSITIVITY")
print("-" * 70)

for date in influential_dates:

    reduced_df = df[
        df["date"] != date
    ].copy()

    result = fit_model(
        reduced_df,
        "Exclude " + str(date.date())
    )

    results.append(result)

# ---------------------------------------------------------
# 6. REMOVE ALL THREE TOGETHER
# ---------------------------------------------------------

print("\n4. COMBINED INFLUENCE SENSITIVITY")
print("-" * 70)

combined_df = df[
    ~df["date"].isin(influential_dates)
].copy()

combined = fit_model(
    combined_df,
    "Exclude all three"
)

results.append(combined)

# ---------------------------------------------------------
# 7. DISPLAY RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n5. MODEL COMPARISON")
print("-" * 70)

print(
    results_df.round(4).to_string(index=False)
)

# ---------------------------------------------------------
# 8. SAVE RESULTS
# ---------------------------------------------------------

results_df.to_csv(
    "tables/deep_work_influence_sensitivity.csv",
    index=False
)

print("\n6. OUTPUT")
print("-" * 70)

print(
    "Saved: tables/deep_work_influence_sensitivity.csv"
)

print("\nDEEP WORK INFLUENCE SENSITIVITY COMPLETE")
