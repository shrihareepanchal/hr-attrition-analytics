"""
FastAPI service for real-time attrition prediction and flight risk scoring.

Endpoints:
  - POST /predict/attrition  — Predict attrition probability for an employee
  - POST /score/flight-risk  — Compute composite flight risk score
  - GET  /health             — Service health check
"""

import os
import numpy as np
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="HR Attrition Analytics API",
    description="ML-powered attrition prediction and flight risk scoring",
    version="1.0.0",
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# Lazy-loaded model cache
_model_cache = {}


def _load_artifact(name: str):
    """Load and cache a model artifact."""
    if name not in _model_cache:
        path = MODELS_DIR / name
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"Model artifact not found: {name}. Run training first.")
        _model_cache[name] = joblib.load(path)
    return _model_cache[name]


class EmployeeInput(BaseModel):
    age: int = Field(..., ge=18, le=70)
    department: str = Field("Engineering", description="Department: Engineering, Finance, HR, Marketing, Operations, Sales")
    tenure_months: int = Field(..., ge=0)
    salary: float = Field(..., gt=0)
    performance_rating: int = Field(..., ge=1, le=5)
    satisfaction_score: float = Field(..., ge=1.0, le=5.0)
    engagement_score: float = Field(..., ge=1.0, le=5.0)
    work_hours_weekly: int = Field(..., ge=20, le=80)
    overtime_flag: int = Field(..., ge=0, le=1)
    remote_days: int = Field(..., ge=0, le=5)
    commute_distance: float = Field(..., ge=0)
    num_direct_reports: int = Field(0, ge=0)
    team_size: int = Field(10, ge=1)
    manager_tenure: int = Field(24, ge=0)
    promotion_last_3y: int = Field(0, ge=0, le=5)
    last_raise_pct: float = Field(3.0, ge=0)
    stock_options: int = Field(0, ge=0, le=3)
    training_hours_annual: int = Field(20, ge=0)
    num_projects: int = Field(4, ge=0)
    peer_rating: float = Field(3.5, ge=1.0, le=5.0)
    skip_level_meeting_freq: int = Field(1, ge=0)
    time_since_last_promo: int = Field(24, ge=0)
    salary_percentile: int = Field(50, ge=0, le=100)
    role_level_encoded: int = Field(3, ge=1, le=8)
    model_name: Optional[str] = Field("lightgbm", description="Model to use for prediction")


class AttritionResponse(BaseModel):
    attrition_probability: float
    prediction: str
    model_used: str
    confidence: float


class FlightRiskResponse(BaseModel):
    flight_risk_score: float
    risk_tier: str
    attrition_probability: float
    risk_factors: list


class HealthResponse(BaseModel):
    status: str
    models_loaded: list
    version: str


def _employee_to_feature_vector(employee: EmployeeInput) -> np.ndarray:
    """Convert employee input to a feature vector matching trained model columns."""
    feature_cols = _load_artifact("feature_columns.joblib")

    # Build a dict of all available raw features from input
    raw_features = employee.model_dump(exclude={"model_name", "department"})

    # Compute derived features
    raw_features["salary_vs_dept_median"] = 1.0  # Approximation without full dataset
    raw_features["salary_vs_level_median"] = 1.0
    raw_features["salary_vs_team_median"] = 1.0
    raw_features["comp_gap_k"] = 0.0
    raw_features["raise_per_perf_point"] = round(raw_features["last_raise_pct"] / max(raw_features["performance_rating"], 1), 2)

    stock_value_map = {0: 0, 1: 5000, 2: 15000, 3: 30000}
    raw_features["estimated_total_comp"] = raw_features["salary"] + stock_value_map.get(raw_features["stock_options"], 0)

    # Engagement composite
    sat_norm = (raw_features["satisfaction_score"] - 1) / 4
    eng_norm = (raw_features["engagement_score"] - 1) / 4
    peer_norm = (raw_features["peer_rating"] - 1) / 4
    training_norm = raw_features["training_hours_annual"] / 80
    skip_norm = raw_features["skip_level_meeting_freq"] / 4
    raw_features["engagement_composite"] = round(sat_norm * 0.30 + eng_norm * 0.30 + peer_norm * 0.15 + training_norm * 0.15 + skip_norm * 0.10, 3)
    raw_features["sat_eng_gap"] = round(raw_features["satisfaction_score"] - raw_features["engagement_score"], 2)

    # Manager proxies
    raw_features["manager_tenure_ratio"] = round(raw_features["manager_tenure"] / max(raw_features["tenure_months"], 1), 2)
    raw_features["team_avg_satisfaction"] = raw_features["satisfaction_score"]
    raw_features["team_avg_performance"] = float(raw_features["performance_rating"])
    raw_features["team_attrition_rate"] = 0.15
    raw_features["span_of_control"] = round(raw_features["num_direct_reports"] / max(raw_features["team_size"], 1), 2)

    # Flight risk flags
    raw_features["stagnation_flag"] = int(raw_features["promotion_last_3y"] == 0 and raw_features["tenure_months"] > 24 and raw_features["last_raise_pct"] < 3.0)
    raw_features["burnout_flag"] = int(raw_features["work_hours_weekly"] > 48 and raw_features["overtime_flag"] == 1 and raw_features["satisfaction_score"] < 3.5)
    raw_features["comp_dissatisfaction_flag"] = int(raw_features["salary_percentile"] < 35 and raw_features["last_raise_pct"] < 2.5)
    raw_features["high_perf_at_risk"] = int(raw_features["performance_rating"] >= 4 and raw_features["satisfaction_score"] < 3.5 and raw_features["stock_options"] == 0)
    raw_features["isolation_flag"] = int(raw_features["skip_level_meeting_freq"] == 0 and raw_features["engagement_score"] < 3.0)
    raw_features["commute_burden_flag"] = int(raw_features["commute_distance"] > 25 and raw_features["remote_days"] <= 1)

    risk_flags = ["stagnation_flag", "burnout_flag", "comp_dissatisfaction_flag", "high_perf_at_risk", "isolation_flag", "commute_burden_flag"]
    raw_features["risk_flag_count"] = sum(raw_features[f] for f in risk_flags)

    # Department one-hot encoding from input
    valid_depts = ["Engineering", "Finance", "HR", "Marketing", "Operations", "Sales"]
    for dept in valid_depts:
        raw_features[f"dept_{dept}"] = int(employee.department == dept)

    # Assemble feature vector in correct column order
    feature_vector = []
    for col in feature_cols:
        feature_vector.append(raw_features.get(col, 0.0))

    return np.array([feature_vector])


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Service health check."""
    available_models = []
    for name in ["logistic_regression", "random_forest", "xgboost", "lightgbm"]:
        if (MODELS_DIR / f"{name}.joblib").exists():
            available_models.append(name)
    return HealthResponse(
        status="healthy",
        models_loaded=available_models,
        version="1.0.0",
    )


@app.post("/predict/attrition", response_model=AttritionResponse)
def predict_attrition(employee: EmployeeInput):
    """Predict attrition probability for an employee."""
    model_name = employee.model_name or "lightgbm"
    valid_models = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    if model_name not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {valid_models}")

    model = _load_artifact(f"{model_name}.joblib")
    X = _employee_to_feature_vector(employee)

    if model_name == "logistic_regression":
        scaler = _load_artifact("scaler.joblib")
        X = scaler.transform(X)

    proba = model.predict_proba(X)[0]
    attrition_prob = float(proba[1])
    prediction = "High Risk" if attrition_prob >= 0.5 else "Low Risk"
    confidence = float(max(proba))

    return AttritionResponse(
        attrition_probability=round(attrition_prob, 4),
        prediction=prediction,
        model_used=model_name,
        confidence=round(confidence, 4),
    )


@app.post("/score/flight-risk", response_model=FlightRiskResponse)
def score_flight_risk(employee: EmployeeInput):
    """Compute composite flight risk score for an employee."""
    model = _load_artifact("lightgbm.joblib")
    X = _employee_to_feature_vector(employee)

    proba = model.predict_proba(X)[0]
    attrition_prob = float(proba[1])

    # Compute risk factors
    risk_factors = []
    if employee.satisfaction_score < 3.0:
        risk_factors.append("Low satisfaction score")
    if employee.engagement_score < 3.0:
        risk_factors.append("Low engagement score")
    if employee.promotion_last_3y == 0 and employee.tenure_months > 24:
        risk_factors.append("No recent promotion despite tenure")
    if employee.overtime_flag == 1 and employee.work_hours_weekly > 48:
        risk_factors.append("Burnout risk: excessive overtime")
    if employee.salary_percentile < 30:
        risk_factors.append("Below-market compensation")
    if employee.stock_options == 0 and employee.performance_rating >= 4:
        risk_factors.append("High performer without equity")
    if employee.last_raise_pct < 2.0:
        risk_factors.append("Minimal recent raise")
    if employee.commute_distance > 25 and employee.remote_days <= 1:
        risk_factors.append("Long commute with limited remote work")

    # Composite score
    risk_flag_count = len(risk_factors)
    risk_norm = min(risk_flag_count / 6, 1.0)
    flight_risk_score = round((attrition_prob * 0.60 + risk_norm * 0.40) * 100, 1)

    if flight_risk_score >= 75:
        risk_tier = "Critical"
    elif flight_risk_score >= 50:
        risk_tier = "High"
    elif flight_risk_score >= 25:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    return FlightRiskResponse(
        flight_risk_score=flight_risk_score,
        risk_tier=risk_tier,
        attrition_probability=round(attrition_prob, 4),
        risk_factors=risk_factors,
    )
