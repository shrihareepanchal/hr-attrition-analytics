"""
DEI fairness auditing for attrition models.

Measures demographic parity, equalized odds, and disparate impact
across protected attributes (gender, age group) using Fairlearn.
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score, recall_score, precision_score


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.DataFrame,
) -> dict:
    """Compute comprehensive fairness metrics across sensitive attributes."""
    results = {}

    for col in sensitive_features.columns:
        sf = sensitive_features[col]

        # Metric frame for group-level analysis
        metric_frame = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "selection_rate": selection_rate,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "recall": recall_score,
                "precision": precision_score,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf,
        )

        # Summary metrics
        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
        dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sf)
        eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sf)

        # Disparate impact ratio (4/5 rule)
        group_selection_rates = metric_frame.by_group["selection_rate"]
        min_rate = group_selection_rates.min()
        max_rate = group_selection_rates.max()
        disparate_impact = min_rate / max_rate if max_rate > 0 else 0.0

        results[col] = {
            "demographic_parity_difference": round(dp_diff, 4),
            "demographic_parity_ratio": round(dp_ratio, 4),
            "equalized_odds_difference": round(eo_diff, 4),
            "disparate_impact_ratio": round(disparate_impact, 4),
            "group_metrics": metric_frame.by_group.round(4).to_dict(),
            "overall_metrics": metric_frame.overall.to_dict(),
            "passes_four_fifths_rule": disparate_impact >= 0.80,
        }

    return results


def generate_bias_recommendations(fairness_results: dict) -> list:
    """Generate actionable bias mitigation recommendations based on audit."""
    recommendations = []

    for attribute, metrics in fairness_results.items():
        # Check demographic parity
        if abs(metrics["demographic_parity_difference"]) > 0.05:
            recommendations.append({
                "attribute": attribute,
                "issue": "Demographic Parity Violation",
                "severity": "HIGH" if abs(metrics["demographic_parity_difference"]) > 0.10 else "MEDIUM",
                "detail": f"Demographic parity difference of {metrics['demographic_parity_difference']:.3f} "
                          f"exceeds 0.05 threshold for {attribute}.",
                "recommendation": (
                    f"Consider rebalancing training data across {attribute} groups, "
                    f"applying threshold adjustment, or using fairness-constrained "
                    f"optimization (e.g., ExponentiatedGradient with DemographicParity constraint)."
                ),
            })

        # Check equalized odds
        if abs(metrics["equalized_odds_difference"]) > 0.08:
            recommendations.append({
                "attribute": attribute,
                "issue": "Equalized Odds Violation",
                "severity": "HIGH" if abs(metrics["equalized_odds_difference"]) > 0.15 else "MEDIUM",
                "detail": f"Equalized odds difference of {metrics['equalized_odds_difference']:.3f} "
                          f"exceeds 0.08 threshold for {attribute}.",
                "recommendation": (
                    f"Review error rates by {attribute} subgroup. Consider post-processing "
                    f"calibration (ThresholdOptimizer) to equalize TPR/FPR across groups."
                ),
            })

        # Check 4/5 rule
        if not metrics["passes_four_fifths_rule"]:
            recommendations.append({
                "attribute": attribute,
                "issue": "Disparate Impact (4/5 Rule Violation)",
                "severity": "HIGH",
                "detail": f"Disparate impact ratio of {metrics['disparate_impact_ratio']:.3f} "
                          f"falls below the 0.80 (4/5 rule) threshold for {attribute}.",
                "recommendation": (
                    f"This may indicate discriminatory impact. Review feature selection for "
                    f"proxies of {attribute}. Consider removing correlated features or "
                    f"applying in-processing fairness constraints."
                ),
            })

    if not recommendations:
        recommendations.append({
            "attribute": "all",
            "issue": "No Significant Bias Detected",
            "severity": "LOW",
            "detail": "All fairness metrics are within acceptable thresholds.",
            "recommendation": "Continue monitoring fairness metrics in production.",
        })

    return recommendations


def generate_fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.DataFrame,
) -> dict:
    """Generate a complete fairness audit report."""
    metrics = compute_fairness_metrics(y_true, y_pred, sensitive_features)
    recommendations = generate_bias_recommendations(metrics)

    report = {
        "fairness_metrics": metrics,
        "recommendations": recommendations,
        "summary": {
            "attributes_audited": list(sensitive_features.columns),
            "n_samples": len(y_true),
            "n_recommendations": len(recommendations),
            "has_violations": any(r["severity"] in ("HIGH", "MEDIUM") for r in recommendations),
        },
    }

    return report
