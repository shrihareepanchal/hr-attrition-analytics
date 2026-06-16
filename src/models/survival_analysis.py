"""
Survival analysis for employee attrition.

Implements Kaplan-Meier estimation and Cox Proportional Hazards modeling
to predict time-to-attrition and identify hazard ratios for key factors.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index


class KaplanMeierAnalysis:
    """Kaplan-Meier survival curve estimation by cohorts."""

    def __init__(self):
        self.fitters = {}

    def fit_by_group(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration_months",
        event_col: str = "event_observed",
        group_col: str = "department",
    ) -> dict:
        """Fit Kaplan-Meier curves for each group in group_col."""
        self.fitters = {}
        for group_name, group_df in df.groupby(group_col):
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=group_df[duration_col],
                event_observed=group_df[event_col],
                label=str(group_name),
            )
            self.fitters[str(group_name)] = kmf
        return self.fitters

    def get_median_survival_times(self) -> dict:
        """Return median survival time for each fitted group."""
        return {
            name: float(kmf.median_survival_time_)
            for name, kmf in self.fitters.items()
        }

    def get_survival_function(self, group_name: str) -> pd.DataFrame:
        """Return the survival function for a specific group."""
        if group_name not in self.fitters:
            raise KeyError(f"Group '{group_name}' not found. Available: {list(self.fitters.keys())}")
        return self.fitters[group_name].survival_function_


class CoxPHAnalysis:
    """Cox Proportional Hazards model for attrition."""

    def __init__(self, penalizer: float = 0.01):
        self.model = CoxPHFitter(penalizer=penalizer)
        self.is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration_months",
        event_col: str = "event_observed",
    ) -> "CoxPHAnalysis":
        """Fit Cox PH model."""
        self.duration_col = duration_col
        self.event_col = event_col
        self.model.fit(
            df,
            duration_col=duration_col,
            event_col=event_col,
        )
        self.is_fitted = True
        return self

    def get_hazard_ratios(self) -> pd.DataFrame:
        """Return hazard ratios with confidence intervals."""
        summary = self.model.summary
        result = pd.DataFrame({
            "hazard_ratio": np.exp(summary["coef"]),
            "hr_lower_95": np.exp(summary["coef lower 95%"]),
            "hr_upper_95": np.exp(summary["coef upper 95%"]),
            "p_value": summary["p"],
        })
        return result.sort_values("hazard_ratio", ascending=False)

    def predict_survival_function(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict survival function for new data."""
        return self.model.predict_survival_function(df)

    def predict_median_survival(self, df: pd.DataFrame) -> pd.Series:
        """Predict median survival time for new data."""
        return self.model.predict_median(df)

    def predict_hazard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict cumulative hazard for new data."""
        return self.model.predict_cumulative_hazard(df)

    def get_concordance_index(self) -> float:
        """Return the concordance index (C-index) of the fitted model."""
        return self.model.concordance_index_

    def get_summary(self) -> pd.DataFrame:
        """Return full model summary."""
        return self.model.summary


def prepare_survival_data(
    df: pd.DataFrame,
    feature_cols: list,
    duration_col: str = "duration_months",
    event_col: str = "event_observed",
) -> pd.DataFrame:
    """Prepare dataframe for survival analysis, keeping only needed columns."""
    cols = feature_cols + [duration_col, event_col]
    survival_df = df[cols].copy()
    survival_df = survival_df.dropna()
    return survival_df
