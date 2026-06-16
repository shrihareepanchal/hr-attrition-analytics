"""
Training pipeline for attrition prediction models.

Trains all classifiers with stratified cross-validation, generates
flight risk scores, and saves model artifacts and metrics.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from src.models.attrition_classifier import AttritionModel, get_all_models
from src.models.survival_analysis import CoxPHAnalysis, prepare_survival_data
from src.models.fairness_audit import generate_fairness_report


# Columns to exclude from model features
NON_FEATURE_COLS = [
    "employee_id", "attrition", "attrition_date", "event_observed",
    "duration_months", "gender", "age_group", "tenure_bucket",
    "role_level", "department",
]

SURVIVAL_FEATURES = [
    "age", "tenure_months", "salary", "performance_rating",
    "satisfaction_score", "engagement_score", "overtime_flag",
    "promotion_last_3y", "work_hours_weekly", "remote_days",
    "last_raise_pct", "stock_options", "training_hours_annual",
    "salary_vs_dept_median", "engagement_composite",
    "risk_flag_count", "role_level_encoded",
]


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature columns by excluding non-feature columns."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS and df[c].dtype in ("int64", "float64", "int32", "float32", "uint8")]


def train_classifiers(df: pd.DataFrame, models_dir: Path, results_dir: Path) -> dict:
    """Train all attrition classifiers with stratified cross-validation."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["attrition"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = {}

    models = get_all_models()

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        # Use scaled features for logistic regression, raw for tree models
        X_train = X_scaled if model_name == "logistic_regression" else X

        # Stratified cross-validation
        cv_results = cross_validate(
            model.model, X_train, y,
            cv=skf,
            scoring=["roc_auc", "average_precision", "f1", "precision", "recall"],
            return_train_score=False,
            n_jobs=-1,
        )

        cv_metrics = {
            "roc_auc": round(cv_results["test_roc_auc"].mean(), 4),
            "roc_auc_std": round(cv_results["test_roc_auc"].std(), 4),
            "pr_auc": round(cv_results["test_average_precision"].mean(), 4),
            "f1": round(cv_results["test_f1"].mean(), 4),
            "precision": round(cv_results["test_precision"].mean(), 4),
            "recall": round(cv_results["test_recall"].mean(), 4),
        }

        # Fit on full data for artifact saving and downstream use
        model.fit(X_train, y)
        y_pred = model.predict(X_train)
        y_proba = model.predict_proba(X_train)[:, 1]

        train_metrics = {
            "train_roc_auc": round(roc_auc_score(y, y_proba), 4),
            "train_f1": round(f1_score(y, y_pred), 4),
        }

        all_metrics[model_name] = {**cv_metrics, **train_metrics}

        # Save model artifact
        joblib.dump(model, models_dir / f"{model_name}.joblib")

        # Feature importance
        importance = model.get_feature_importance(feature_cols)
        top_features = dict(list(importance.items())[:15])
        all_metrics[model_name]["top_features"] = top_features

        print(f"  CV ROC-AUC: {cv_metrics['roc_auc']:.4f} (+/- {cv_metrics['roc_auc_std']:.4f})")
        print(f"  CV F1: {cv_metrics['f1']:.4f}")

    # Save scaler
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(feature_cols, models_dir / "feature_columns.joblib")

    return all_metrics


def train_survival_model(df: pd.DataFrame, models_dir: Path) -> dict:
    """Train Cox PH survival model."""
    print("\nTraining Cox PH survival model...")

    available_features = [c for c in SURVIVAL_FEATURES if c in df.columns]
    survival_df = prepare_survival_data(
        df, available_features,
        duration_col="duration_months",
        event_col="event_observed",
    )

    cox = CoxPHAnalysis(penalizer=0.01)
    cox.fit(survival_df, duration_col="duration_months", event_col="event_observed")

    c_index = cox.get_concordance_index()
    hazard_ratios = cox.get_hazard_ratios()

    print(f"  C-Index: {c_index:.4f}")
    print(f"  Top hazard ratios:")
    for feat, hr in hazard_ratios["hazard_ratio"].head(5).items():
        print(f"    {feat}: {hr:.3f}")

    joblib.dump(cox, models_dir / "cox_ph_model.joblib")

    return {
        "c_index": round(c_index, 4),
        "top_hazard_ratios": hazard_ratios["hazard_ratio"].head(10).round(4).to_dict(),
    }


def compute_flight_risk_scores(df: pd.DataFrame, models_dir: Path) -> pd.DataFrame:
    """Compute composite flight risk scores combining model predictions and indicators."""
    feature_cols = joblib.load(models_dir / "feature_columns.joblib")
    X = df[feature_cols].values

    # Load best classifier (LightGBM)
    best_model = joblib.load(models_dir / "lightgbm.joblib")
    model_prob = best_model.predict_proba(X)[:, 1]

    # Load Cox PH for survival-based risk
    cox_model = joblib.load(models_dir / "cox_ph_model.joblib")
    available_features = [c for c in SURVIVAL_FEATURES if c in df.columns]
    survival_df = prepare_survival_data(
        df, available_features,
        duration_col="duration_months",
        event_col="event_observed",
    )
    cox_hazard = cox_model.predict_hazard(survival_df.drop(columns=["duration_months", "event_observed"])).iloc[-1].values

    # Normalize components to 0-1
    hazard_norm = (cox_hazard - cox_hazard.min()) / (cox_hazard.max() - cox_hazard.min() + 1e-8)

    # Risk flag count normalized
    risk_flags_norm = df["risk_flag_count"].values / df["risk_flag_count"].max() if "risk_flag_count" in df.columns else np.zeros(len(df))

    # Composite flight risk score (0-100)
    flight_risk = (
        model_prob * 0.50 +
        hazard_norm * 0.30 +
        risk_flags_norm * 0.20
    ) * 100

    df = df.copy()
    df["flight_risk_score"] = np.round(flight_risk, 1)
    df["flight_risk_tier"] = pd.cut(
        df["flight_risk_score"],
        bins=[0, 25, 50, 75, 100],
        labels=["Low", "Medium", "High", "Critical"],
    )

    return df


def run_fairness_audit(df: pd.DataFrame, models_dir: Path) -> dict:
    """Run fairness audit on the best model predictions."""
    feature_cols = joblib.load(models_dir / "feature_columns.joblib")
    X = df[feature_cols].values

    best_model = joblib.load(models_dir / "lightgbm.joblib")
    y_pred = best_model.predict(X)
    y_true = df["attrition"].values

    sensitive_features = pd.DataFrame({
        "gender": df["gender"],
        "age_group": df["age_group"].astype(str),
    })

    report = generate_fairness_report(y_true, y_pred, sensitive_features)
    print("\nFairness Audit Summary:")
    for attr, metrics in report["fairness_metrics"].items():
        print(f"  {attr}:")
        print(f"    Demographic Parity Ratio: {metrics['demographic_parity_ratio']:.4f}")
        print(f"    Equalized Odds Diff: {metrics['equalized_odds_difference']:.4f}")
        print(f"    Disparate Impact: {metrics['disparate_impact_ratio']:.4f}")
        print(f"    Passes 4/5 Rule: {metrics['passes_four_fifths_rule']}")

    return report


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "hr_features.csv"
    models_dir = project_root / "models"
    results_dir = project_root / "results" / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HR Attrition Model Training Pipeline")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    print(f"Attrition rate: {df['attrition'].mean():.1%}")

    # 1. Train classifiers
    classifier_metrics = train_classifiers(df, models_dir, results_dir)

    # 2. Train survival model
    survival_metrics = train_survival_model(df, models_dir)

    # 3. Compute flight risk scores
    df_scored = compute_flight_risk_scores(df, models_dir)
    df_scored.to_csv(project_root / "data" / "processed" / "hr_scored.csv", index=False)
    print(f"\nFlight Risk Distribution:")
    print(df_scored["flight_risk_tier"].value_counts().to_string())

    # 4. Fairness audit
    fairness_report = run_fairness_audit(df, models_dir)

    # 5. Save all metrics
    all_results = {
        "classifier_metrics": classifier_metrics,
        "survival_metrics": survival_metrics,
        "fairness_report": {
            k: v for k, v in fairness_report.items() if k != "fairness_metrics"
        },
        "flight_risk_stats": {
            "mean_score": round(df_scored["flight_risk_score"].mean(), 2),
            "median_score": round(df_scored["flight_risk_score"].median(), 2),
            "critical_count": int((df_scored["flight_risk_tier"] == "Critical").sum()),
            "high_count": int((df_scored["flight_risk_tier"] == "High").sum()),
        },
    }

    metrics_path = results_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nMetrics saved to: {metrics_path}")
    print("Training pipeline complete.")


if __name__ == "__main__":
    main()
