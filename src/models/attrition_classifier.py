"""
Attrition classification models.

Provides a unified interface for Logistic Regression, Random Forest,
XGBoost, and LightGBM classifiers with fit/predict/predict_proba methods.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class AttritionModel:
    """Wrapper for attrition classification models with a unified API."""

    MODELS = {
        "logistic_regression": {
            "class": LogisticRegression,
            "params": {
                "max_iter": 1000,
                "C": 1.0,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "random_state": 42,
            },
        },
        "random_forest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 300,
                "max_depth": 12,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        "xgboost": {
            "class": XGBClassifier,
            "params": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 3,
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        "lightgbm": {
            "class": LGBMClassifier,
            "params": {
                "n_estimators": 300,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "is_unbalance": True,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            },
        },
    }

    def __init__(self, model_name: str):
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.MODELS.keys())}")
        self.model_name = model_name
        config = self.MODELS[model_name]
        self.model = config["class"](**config["params"])
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AttritionModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list) -> dict:
        if self.model_name == "logistic_regression":
            importances = np.abs(self.model.coef_[0])
        else:
            importances = self.model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def get_all_models() -> dict:
    """Return a dictionary of all attrition models."""
    return {name: AttritionModel(name) for name in AttritionModel.MODELS}
