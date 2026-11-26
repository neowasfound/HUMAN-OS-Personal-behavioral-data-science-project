
import pandas as pd
import numpy as np
import statsmodels.api as sm

print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("19 - PRODUCTIVITY INFLUENCE SENSITIVITY")
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
    "productivity",
    "stress_level",
    "social_contact",
    "social_contact_num"
]

df = df[variables].copy()

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------
# 2. CHECK SOCIAL CONTACT CODING
# ---------------------------------------------------------

print("\n1. SOCIAL CONTACT CODING")
print("-" * 70)

print(
    pd.crosstab(
        df["social_contact"],
        df["social_contact_num"],
        margins=True
    )
)

print("\nSocial contact numerical values:")
print(df["social_contact_num"].value_counts().sort_index())

# ---------------------------------------------------------
# 3. DEFINE MODEL
# ---------------------------------------------------------

predictors = [
    "stress_level",
    "social_contact_num"
]

y_variable = "productivity"

# Influential dates identified in Script 18
influential_dates = [
    "2025-09-02",
    "2025-10-14",
    "2026-01-06",
    "2026-01-07"
]

influential_dates = pd.to_datetime(influential_dates)

# ---------------------------------------------------------
# 4. FUNCTION TO FIT MODEL
# ---------------------------------------------------------

def fit_model(data, scenario):

    X = sm.add_constant(data[predictors])
    y = data[y_variable]

    model = sm.OLS(y, X).fit()
    robust = model.get_robustcov_results(cov_type="HC3")

    results = {
        "scenario": scenario,
        "observations": int(model.nobs),
        "R_squared": model.rsquared,
        "adjusted_R_squared": model.rsquared_adj
    }

    for i, variable in enumerate(X.columns):

        results[variable + "_coef"] = robust.params[i]
        results[variable + "_p"] = robust.pvalues[i]

    return results


# ---------------------------------------------------------
# 5. ORIGINAL MODEL
# ---------------------------------------------------------

print("\n2. ORIGINAL MODEL")
print("-" * 70)

results = []

original = fit_model(df, "Original dataset")
results.append(original)

# ---------------------------------------------------------
# 6. REMOVE EACH INFLUENTIAL OBSERVATION INDIVIDUALLY
# ---------------------------------------------------------

print("\n3. INDIVIDUAL INFLUENCE SENSITIVITY")
print("-" * 70)

for date in influential_dates:

    reduced_df = df[df["date"] != date].copy()

    result = fit_model(
        reduced_df,
        "Exclude " + str(date.date())
    )

    results.append(result)

# ---------------------------------------------------------
# 7. REMOVE ALL FOUR TOGETHER
# ---------------------------------------------------------

print("\n4. COMBINED INFLUENCE SENSITIVITY")
print("-" * 70)

combined_df = df[
    ~df["date"].isin(influential_dates)
].copy()

combined = fit_model(
    combined_df,
    "Exclude all four"
)

results.append(combined)

# ---------------------------------------------------------
# 8. DISPLAY RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n5. MODEL COMPARISON")
print("-" * 70)

print(results_df.round(4).to_string(index=False))

# ---------------------------------------------------------
# 9. SAVE RESULTS
# ---------------------------------------------------------

results_df.to_csv(
    "tables/productivity_influence_sensitivity.csv",
    index=False
)

print("\n6. OUTPUT")
print("-" * 70)

print("Saved: tables/productivity_influence_sensitivity.csv")

print("\nINFLUENCE SENSITIVITY ANALYSIS COMPLETE")
