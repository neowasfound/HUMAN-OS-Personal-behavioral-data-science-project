
import pandas as pd
import matplotlib.pyplot as plt
import os


print("=" * 70)
print("HUMANOS INTELLIGENCE")
print("26 - MACHINE LEARNING VISUALISATION")
print("=" * 70)


# ---------------------------------------------------------
# 1. LOAD RESULTS
# ---------------------------------------------------------

results = pd.read_csv(
    "tables/25_ml_model_comparison.csv"
)

folds = pd.read_csv(
    "tables/25_ml_fold_results.csv"
)

importance = pd.read_csv(
    "tables/25_ml_feature_importance.csv"
)


os.makedirs("figures", exist_ok=True)


# ---------------------------------------------------------
# 2. MODEL PERFORMANCE COMPARISON
# ---------------------------------------------------------

print("\n1. MODEL PERFORMANCE COMPARISON")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ["MAE", "RMSE", "R2"]

for i, metric in enumerate(metrics):

    axes[i].bar(
        results["Model"],
        results[metric]
    )

    axes[i].set_title(metric)
    axes[i].set_ylabel("Score")
    axes[i].tick_params(
        axis="x",
        rotation=25
    )

    axes[i].axhline(
        0,
        linestyle="--",
        linewidth=0.8
    )

plt.suptitle(
    "Machine Learning Model Comparison"
)

plt.tight_layout()

plt.savefig(
    "figures/25_ml_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 3. R2 ACROSS VALIDATION FOLDS
# ---------------------------------------------------------

print("2. VALIDATION FOLD PERFORMANCE")


fig, ax = plt.subplots(figsize=(10, 6))

for model in folds["Model"].unique():

    model_data = folds[
        folds["Model"] == model
    ]

    ax.plot(
        model_data["Fold"],
        model_data["R2"],
        marker="o",
        label=model
    )

ax.axhline(
    0,
    linestyle="--",
    linewidth=1
)

ax.set_title(
    "R-squared Across Validation Folds"
)

ax.set_xlabel("Validation Fold")
ax.set_ylabel("R-squared")

ax.legend()

plt.tight_layout()

plt.savefig(
    "figures/26_ml_fold_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 4. RANDOM FOREST FEATURE IMPORTANCE
# ---------------------------------------------------------

print("3. FEATURE IMPORTANCE")


fig, ax = plt.subplots(figsize=(9, 6))

importance = importance.sort_values(
    "Importance",
    ascending=True
)

ax.barh(
    importance["Feature"],
    importance["Importance"]
)

ax.set_title(
    "Random Forest Feature Importance"
)

ax.set_xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "figures/27_ml_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 5. OUTPUT
# ---------------------------------------------------------

print("\n4. SAVED FIGURES")
print("-" * 70)

print("figures/25_ml_model_comparison.png")
print("figures/26_ml_fold_performance.png")
print("figures/27_ml_feature_importance.png")


print("\n" + "=" * 70)
print("VISUALISATION COMPLETE")
print("=" * 70)
