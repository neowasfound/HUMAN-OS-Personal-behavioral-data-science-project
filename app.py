
from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

app = Flask(__name__)

# =========================================================
# 1. PROJECT PATHS
# =========================================================

DATA_PATH = "data/processed/humanos_cleaned.csv"
TABLES_PATH = "tables"


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["date"])
df = df.sort_values("date").reset_index(drop=True)


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def average(column):

    if column in df.columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if not values.empty:
            return round(values.mean(), 2)

    return None

def load_ml_results(filename):

    path = os.path.join(TABLES_PATH, filename)

    if os.path.exists(path):
        return pd.read_csv(path)

    return None


def chart_html(fig):

    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn"
    )


# =========================================================
# 4. DASHBOARD INSIGHTS
# =========================================================

def get_dashboard_insights():

    # PRODUCTIVITY RECORDS

    productivity_data = df[["date", "productivity"]].copy()

    productivity_data["productivity"] = pd.to_numeric(
        productivity_data["productivity"],
        errors="coerce"
    )

    productivity_data = productivity_data.dropna(
        subset=["productivity"]
    )

    if not productivity_data.empty:

        highest_record = productivity_data.loc[
            productivity_data["productivity"].idxmax()
        ]

        lowest_record = productivity_data.loc[
            productivity_data["productivity"].idxmin()
        ]

        highest_productivity = highest_record["productivity"]

        highest_productivity_date = (
            highest_record["date"].strftime("%d %b %Y")
        )

        lowest_productivity = lowest_record["productivity"]

        lowest_productivity_date = (
            lowest_record["date"].strftime("%d %b %Y")
        )

    else:

        highest_productivity = None
        highest_productivity_date = "Unavailable"

        lowest_productivity = None
        lowest_productivity_date = "Unavailable"


    # STRONGEST SPEARMAN ASSOCIATION

    candidate_variables = {
        "stress_level": "Stress",
        "social_contact_num": "Social Contact",
        "mood": "Mood",
        "energy": "Energy",
        "sleep_hours": "Sleep",
        "exercise_yn": "Exercise",
        "meals_quality_num": "Meal Quality"
    }

    associations = []

    for column, label in candidate_variables.items():

        if column not in df.columns:
            continue

        paired_data = pd.DataFrame({
            "productivity": pd.to_numeric(
                df["productivity"],
                errors="coerce"
            ),
            "predictor": pd.to_numeric(
                df[column],
                errors="coerce"
            )
        }).dropna()

        if len(paired_data) < 3:
            continue

        correlation = paired_data["productivity"].corr(
            paired_data["predictor"],
            method="spearman"
        )

        if pd.notna(correlation):

            associations.append({
                "label": label,
                "rho": round(correlation, 3),
                "n": len(paired_data),
                "direction": (
                    "Positive"
                    if correlation > 0
                    else "Negative"
                    if correlation < 0
                    else "No association"
                )
            })

    if associations:

        strongest_association = max(
            associations,
            key=lambda item: abs(item["rho"])
        )

    else:

        strongest_association = {
            "label": "Unavailable",
            "rho": 0,
            "n": 0,
            "direction": "Unavailable"
        }


    return {
        "highest_productivity": highest_productivity,
        "highest_productivity_date": highest_productivity_date,
        "lowest_productivity": lowest_productivity,
        "lowest_productivity_date": lowest_productivity_date,
        "strongest_association": strongest_association
    }


# =========================================================
# 5. DASHBOARD
# =========================================================

@app.route("/")
def index():

    insights = get_dashboard_insights()

    return render_template(
        "index.html",
        observations=len(df),
        avg_productivity=average("productivity"),
        avg_stress=average("stress_level"),
        avg_deep_work=average("deep_work"),
        **insights
    )


# =========================================================
# 6. DATA EXPLORATION
# =========================================================

@app.route("/analysis")
def analysis():

    filtered_df = df.copy()

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if start_date:
        filtered_df = filtered_df[
            filtered_df["date"] >= pd.to_datetime(start_date)
        ]

    if end_date:
        filtered_df = filtered_df[
            filtered_df["date"] <= pd.to_datetime(end_date)
        ]

    # PRODUCTIVITY OVER TIME

    productivity_chart = px.line(
        filtered_df,
        x="date",
        y="productivity",
        title="Productivity Over Time",
        markers=True
    )

    productivity_chart.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Productivity"
    )

    # STRESS OVER TIME

    stress_chart = px.line(
        filtered_df,
        x="date",
        y="stress_level",
        title="Stress Over Time",
        markers=True
    )

    stress_chart.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Stress Level"
    )

    # MOOD AND ENERGY

    mood_energy = filtered_df.melt(
        id_vars=["date"],
        value_vars=["mood", "energy"],
        var_name="Variable",
        value_name="Score"
    )

    mood_energy_chart = px.line(
        mood_energy,
        x="date",
        y="Score",
        color="Variable",
        title="Mood and Energy Over Time",
        markers=True
    )

    mood_energy_chart.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Score"
    )

    # STRESS VS PRODUCTIVITY

    stress_productivity_chart = px.scatter(
        filtered_df,
        x="stress_level",
        y="productivity",
        title="Stress vs Productivity",
        trendline="ols"
    )

    stress_productivity_chart.update_layout(
        template="plotly_dark",
        xaxis_title="Stress Level",
        yaxis_title="Productivity"
    )

    return render_template(
        "analysis.html",
        observations=len(filtered_df),
        start_date=start_date,
        end_date=end_date,
        productivity_chart=chart_html(productivity_chart),
        stress_chart=chart_html(stress_chart),
        mood_energy_chart=chart_html(mood_energy_chart),
        stress_productivity_chart=chart_html(
            stress_productivity_chart
        )
    )


# =========================================================
# 7. STATISTICAL ANALYSIS
# =========================================================

@app.route("/statistics")
def statistics():

    return render_template("statistics.html")


# =========================================================
# 8. MACHINE LEARNING
# =========================================================

@app.route("/machine-learning")
def machine_learning():

    comparison = load_ml_results(
        "25_ml_model_comparison.csv"
    )

    importance = load_ml_results(
        "25_ml_feature_importance.csv"
    )

    folds = load_ml_results(
        "25_ml_fold_results.csv"
    )

    results_available = (
        comparison is not None
        and importance is not None
        and folds is not None
        and not comparison.empty
        and not importance.empty
        and not folds.empty
    )

    if not results_available:

        return render_template(
            "machine_learning.html",
            results_available=False
        )

    # MODEL COMPARISON

    comparison_chart = px.bar(
        comparison,
        x="Model",
        y=["MAE", "RMSE"],
        barmode="group",
        title="Model Prediction Error Comparison"
    )

    comparison_chart.update_layout(
        template="plotly_dark",
        xaxis_title="Model",
        yaxis_title="Error"
    )

    # CROSS-VALIDATION PERFORMANCE

    fold_chart = px.line(
        folds,
        x="Fold",
        y="R2",
        color="Model",
        markers=True,
        title="R² Across Validation Folds"
    )

    fold_chart.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="R² = 0"
    )

    fold_chart.update_layout(
        template="plotly_dark",
        xaxis_title="Validation Fold",
        yaxis_title="R²"
    )

    # FEATURE IMPORTANCE

    importance_chart = px.bar(
        importance.sort_values(
            "Importance",
            ascending=True
        ),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )

    importance_chart.update_layout(
        template="plotly_dark",
        xaxis_title="Importance",
        yaxis_title="Feature"
    )

    # BEST MODEL

    best_model = comparison.loc[
        comparison["MAE"].idxmin()
    ].to_dict()

    results = comparison.to_dict(
        orient="records"
    )

    return render_template(
        "machine_learning.html",
        results_available=True,
        results=results,
        best_model=best_model,
        comparison_chart=chart_html(comparison_chart),
        fold_chart=chart_html(fold_chart),
        importance_chart=chart_html(importance_chart)
    )


# =========================================================
# 9. ABOUT
# =========================================================

@app.route("/about")
def about():

    return render_template("about.html")


# =========================================================
# 10. RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)
