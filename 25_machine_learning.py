
import pandas as pd
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("25 - PRODUCTIVITY MACHINE LEARNING")
print("=" * 70)


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/humanos_cleaned.csv",
    na_values=["<NA>", "nan", "NaN", ""]
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------
# 2. SELECT VARIABLES
# ---------------------------------------------------------

predictors = [
    "stress_level",
    "mood",
    "energy",
    "sleep_hours",
    "exercise_yn",
    "social_contact_num"
]

target = "productivity"

data = df[
    ["date"] + predictors + [target]
].copy()

data = data.dropna(
    subset=[target]
).reset_index(drop=True)

X = data[predictors]
y = data[target]


print("\n1. DATA OVERVIEW")
print("-" * 70)

print("Observations:", len(data))
print("Predictors:", len(predictors))
print("Start date:", data["date"].min().date())
print("End date:", data["date"].max().date())


# ---------------------------------------------------------
# 3. DEFINE MODELS
# ---------------------------------------------------------

models = {

    "Dummy Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyRegressor(strategy="mean"))
    ]),

    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=5,
            random_state=42
        ))
    ])

}


# ---------------------------------------------------------
# 4. TIME-AWARE VALIDATION
# ---------------------------------------------------------

splitter = TimeSeriesSplit(n_splits=5)

results = []
fold_results = []


print("\n2. CROSS-VALIDATION")
print("-" * 70)


for name, model in models.items():

    print("\n", name)
    print("-" * 70)

    fold_mae = []
    fold_rmse = []
    fold_r2 = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(X), start=1
    ):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        train_dates = data["date"].iloc[train_index]
        test_dates = data["date"].iloc[test_index]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        r2 = r2_score(
            y_test,
            predictions
        )

        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)

        fold_results.append({

            "Model": name,
            "Fold": fold,

            "Training Observations": len(train_index),
            "Testing Observations": len(test_index),

            "Training Start": train_dates.min().date(),
            "Training End": train_dates.max().date(),

            "Testing Start": test_dates.min().date(),
            "Testing End": test_dates.max().date(),

            "MAE": mae,
            "RMSE": rmse,
            "R2": r2

        })

        print(
            "\nFold", fold,
            "| Training:", len(train_index),
            "| Testing:", len(test_index)
        )

        print(
            "Training period:",
            train_dates.min().date(),
            "to",
            train_dates.max().date()
        )

        print(
            "Testing period:",
            test_dates.min().date(),
            "to",
            test_dates.max().date()
        )

        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2:", round(r2, 4))


    # -----------------------------------------------------
    # 5. AVERAGE MODEL PERFORMANCE
    # -----------------------------------------------------

    results.append({

        "Model": name,
        "MAE": sum(fold_mae) / len(fold_mae),
        "RMSE": sum(fold_rmse) / len(fold_rmse),
        "R2": sum(fold_r2) / len(fold_r2)

    })


# ---------------------------------------------------------
# 6. MODEL COMPARISON
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

fold_df = pd.DataFrame(fold_results)


print("\n3. MODEL COMPARISON")
print("-" * 70)

print(
    results_df.to_string(index=False)
)


# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------------------------------------

print("\n4. RANDOM FOREST FEATURE IMPORTANCE")
print("-" * 70)

forest_model = models["Random Forest"]

forest_model.fit(X, y)

forest = forest_model.named_steps["model"]

importance = pd.DataFrame({

    "Feature": predictors,
    "Importance": forest.feature_importances_

})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(
    importance.to_string(index=False)
)


# ---------------------------------------------------------
# 8. SAVE RESULTS
# ---------------------------------------------------------

os.makedirs("tables", exist_ok=True)

results_df.to_csv(
    "tables/25_ml_model_comparison.csv",
    index=False
)

fold_df.to_csv(
    "tables/25_ml_fold_results.csv",
    index=False
)

importance.to_csv(
    "tables/25_ml_feature_importance.csv",
    index=False
)


print("\n5. OUTPUT")
print("-" * 70)

print("Model comparison:")
print("tables/25_ml_model_comparison.csv")

print("Fold results:")
print("tables/25_ml_fold_results.csv")

print("Feature importance:")
print("tables/25_ml_feature_importance.csv")


print("\n" + "=" * 70)
print("MACHINE LEARNING ANALYSIS COMPLETE")
print("=" * 70)
