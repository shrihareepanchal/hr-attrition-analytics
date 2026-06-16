"""
Evaluation pipeline for attrition models.

Generates ROC/PR curves, SHAP explanations, survival curves,
fairness visualization, and cost-of-attrition analysis.
"""

import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)
import shap

from src.models.attrition_classifier import AttritionModel
from src.models.survival_analysis import KaplanMeierAnalysis, CoxPHAnalysis
from src.models.fairness_audit import compute_fairness_metrics
from src.models.train import get_feature_columns, SURVIVAL_FEATURES


# Replacement cost multipliers by role level (multiple of annual salary)
REPLACEMENT_COST_MULTIPLIERS = {
    "IC1": 0.5, "IC2": 0.75, "IC3": 1.0, "IC4": 1.25, "IC5": 1.5,
    "Manager": 1.5, "Director": 2.0, "VP": 2.5,
}


def plot_roc_curves(df: pd.DataFrame, models_dir: Path, figures_dir: Path):
    """Plot ROC curves for all trained classifiers."""
    feature_cols = joblib.load(models_dir / "feature_columns.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")
    X = df[feature_cols].values
    y = df["attrition"].values

    model_names = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for name in model_names:
        model_path = models_dir / f"{name}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        X_input = scaler.transform(X) if name == "logistic_regression" else X
        y_proba = model.predict_proba(X_input)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        label = f"{name.replace('_', ' ').title()} (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, label=label, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Attrition Classifiers")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=150)
    plt.close()
    print("  Saved ROC curves")


def plot_pr_curves(df: pd.DataFrame, models_dir: Path, figures_dir: Path):
    """Plot Precision-Recall curves for all trained classifiers."""
    feature_cols = joblib.load(models_dir / "feature_columns.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")
    X = df[feature_cols].values
    y = df["attrition"].values

    model_names = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for name in model_names:
        model_path = models_dir / f"{name}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        X_input = scaler.transform(X) if name == "logistic_regression" else X
        y_proba = model.predict_proba(X_input)[:, 1]
        precision, recall, _ = precision_recall_curve(y, y_proba)
        pr_auc = auc(recall, precision)
        label = f"{name.replace('_', ' ').title()} (AUC={pr_auc:.3f})"
        ax.plot(recall, precision, label=label, linewidth=2)

    baseline = y.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.4, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Attrition Classifiers")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "pr_curves.png", dpi=150)
    plt.close()
    print("  Saved PR curves")


def plot_confusion_matrices(df: pd.DataFrame, models_dir: Path, figures_dir: Path):
    """Plot confusion matrices for all classifiers."""
    feature_cols = joblib.load(models_dir / "feature_columns.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")
    X = df[feature_cols].values
    y = df["attrition"].values

    model_names = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for ax, name in zip(axes, model_names):
        model_path = models_dir / f"{name}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        X_input = scaler.transform(X) if name == "logistic_regression" else X
        y_pred = model.predict(X_input)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"])
        ax.set_title(name.replace("_", " ").title())
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.suptitle("Confusion Matrices", fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices.png", dpi=150)
    plt.close()
    print("  Saved confusion matrices")


def plot_shap_analysis(df: pd.DataFrame, models_dir: Path, figures_dir: Path):
    """Generate SHAP feature importance and summary plots."""
    feature_cols = joblib.load(models_dir / "feature_columns.joblib")
    X = df[feature_cols].values
    X_df = df[feature_cols]

    best_model = joblib.load(models_dir / "lightgbm.joblib")

    explainer = shap.TreeExplainer(best_model.model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be a list of two arrays
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Summary bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_df, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (LightGBM)")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_df, show=False, max_display=20)
    plt.title("SHAP Summary Plot (LightGBM)")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved SHAP plots")


def plot_survival_curves(df: pd.DataFrame, figures_dir: Path):
    """Plot Kaplan-Meier survival curves by department and tenure bucket."""
    km = KaplanMeierAnalysis()

    for group_col in ["department", "tenure_bucket"]:
        if group_col not in df.columns:
            continue

        km.fit_by_group(df, group_col=group_col)

        fig, ax = plt.subplots(figsize=(10, 6))
        for name, kmf in km.fitters.items():
            kmf.plot_survival_function(ax=ax)

        ax.set_xlabel("Months")
        ax.set_ylabel("Survival Probability")
        ax.set_title(f"Kaplan-Meier Survival Curves by {group_col.replace('_', ' ').title()}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f"km_survival_{group_col}.png", dpi=150)
        plt.close()

    medians = km.get_median_survival_times()
    print(f"  Saved survival curves | Median survival times: {medians}")


def plot_fairness_dashboard(df: pd.DataFrame, models_dir: Path, figures_dir: Path):
    """Generate fairness visualization plots."""
    feature_cols = joblib.load(models_dir / "feature_columns.joblib")
    X = df[feature_cols].values

    best_model = joblib.load(models_dir / "lightgbm.joblib")
    y_pred = best_model.predict(X)
    y_true = df["attrition"].values

    sensitive_features = pd.DataFrame({
        "gender": df["gender"],
        "age_group": df["age_group"].astype(str),
    })

    metrics = compute_fairness_metrics(y_true, y_pred, sensitive_features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (attr, attr_metrics) in zip(axes, metrics.items()):
        fairness_scores = {
            "Dem. Parity\nRatio": attr_metrics["demographic_parity_ratio"],
            "Eq. Odds\nDiff (inv)": 1 - abs(attr_metrics["equalized_odds_difference"]),
            "Disparate\nImpact": attr_metrics["disparate_impact_ratio"],
        }

        colors = []
        for val in fairness_scores.values():
            if val >= 0.90:
                colors.append("#2ecc71")
            elif val >= 0.80:
                colors.append("#f39c12")
            else:
                colors.append("#e74c3c")

        bars = ax.bar(fairness_scores.keys(), fairness_scores.values(), color=colors, edgecolor="black", alpha=0.8)
        ax.axhline(y=0.80, color="red", linestyle="--", alpha=0.5, label="4/5 Rule Threshold")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Fairness Metrics: {attr.replace('_', ' ').title()}")
        ax.set_ylabel("Score")
        ax.legend()

        for bar, val in zip(bars, fairness_scores.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.suptitle("DEI Fairness Audit", fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / "fairness_dashboard.png", dpi=150)
    plt.close()
    print("  Saved fairness dashboard")


def compute_cost_of_attrition(df: pd.DataFrame, results_dir: Path) -> dict:
    """Compute estimated cost of attrition by department and role level."""
    attrited = df[df["attrition"] == 1].copy()

    if "role_level" not in attrited.columns:
        print("  Skipping cost analysis: role_level column missing")
        return {}

    attrited["replacement_cost"] = attrited.apply(
        lambda row: row["salary"] * REPLACEMENT_COST_MULTIPLIERS.get(row["role_level"], 1.0),
        axis=1,
    )

    # By department
    dept_costs = attrited.groupby("department").agg(
        headcount_lost=("employee_id", "count"),
        total_replacement_cost=("replacement_cost", "sum"),
        avg_replacement_cost=("replacement_cost", "mean"),
        avg_salary_lost=("salary", "mean"),
    ).round(0).to_dict(orient="index")

    # By role level
    level_costs = attrited.groupby("role_level").agg(
        headcount_lost=("employee_id", "count"),
        total_replacement_cost=("replacement_cost", "sum"),
        avg_replacement_cost=("replacement_cost", "mean"),
    ).round(0).to_dict(orient="index")

    cost_report = {
        "total_attrition_cost": round(attrited["replacement_cost"].sum(), 0),
        "total_headcount_lost": len(attrited),
        "avg_cost_per_departure": round(attrited["replacement_cost"].mean(), 0),
        "by_department": dept_costs,
        "by_role_level": level_costs,
    }

    cost_path = results_dir / "cost_of_attrition.json"
    with open(cost_path, "w") as f:
        json.dump(cost_report, f, indent=2, default=str)

    print(f"  Total attrition cost: ${cost_report['total_attrition_cost']:,.0f}")
    print(f"  Avg cost per departure: ${cost_report['avg_cost_per_departure']:,.0f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    dept_summary = attrited.groupby("department")["replacement_cost"].sum().sort_values(ascending=True)
    dept_summary.plot(kind="barh", ax=ax1, color="#3498db", edgecolor="black")
    ax1.set_xlabel("Total Replacement Cost ($)")
    ax1.set_title("Attrition Cost by Department")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    level_summary = attrited.groupby("role_level")["replacement_cost"].sum().sort_values(ascending=True)
    level_summary.plot(kind="barh", ax=ax2, color="#e67e22", edgecolor="black")
    ax2.set_xlabel("Total Replacement Cost ($)")
    ax2.set_title("Attrition Cost by Role Level")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.suptitle("Cost of Attrition Analysis", fontsize=14)
    plt.tight_layout()
    figures_dir = results_dir.parent / "figures"
    plt.savefig(figures_dir / "cost_of_attrition.png", dpi=150)
    plt.close()

    return cost_report


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "hr_features.csv"
    models_dir = project_root / "models"
    figures_dir = project_root / "results" / "figures"
    results_dir = project_root / "results" / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HR Attrition Model Evaluation Pipeline")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records\n")

    print("Generating evaluation plots...")
    plot_roc_curves(df, models_dir, figures_dir)
    plot_pr_curves(df, models_dir, figures_dir)
    plot_confusion_matrices(df, models_dir, figures_dir)
    plot_shap_analysis(df, models_dir, figures_dir)
    plot_survival_curves(df, figures_dir)
    plot_fairness_dashboard(df, models_dir, figures_dir)

    print("\nComputing cost of attrition...")
    cost_report = compute_cost_of_attrition(df, results_dir)

    # Generate classification reports
    feature_cols = joblib.load(models_dir / "feature_columns.joblib")
    X = df[feature_cols].values
    y = df["attrition"].values

    reports = {}
    for name in ["logistic_regression", "random_forest", "xgboost", "lightgbm"]:
        model_path = models_dir / f"{name}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        scaler = joblib.load(models_dir / "scaler.joblib")
        X_input = scaler.transform(X) if name == "logistic_regression" else X
        y_pred = model.predict(X_input)
        reports[name] = classification_report(y, y_pred, target_names=["Stay", "Leave"], output_dict=True)

    eval_path = results_dir / "evaluation_report.json"
    with open(eval_path, "w") as f:
        json.dump({"classification_reports": reports, "cost_analysis": cost_report}, f, indent=2, default=str)

    print(f"\nEvaluation report saved to: {eval_path}")
    print("Evaluation pipeline complete.")


if __name__ == "__main__":
    main()
