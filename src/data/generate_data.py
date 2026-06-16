"""
Generate realistic synthetic HR/employee data for attrition analytics.

Produces 5,000 employee records spanning 3 years with realistic
attrition patterns driven by satisfaction, compensation, promotion
history, overtime, and manager quality.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_EMPLOYEES = 5000
OBSERVATION_MONTHS = 36  # 3 years


def _salary_by_level(role_level: str, department: str, rng: np.random.Generator) -> float:
    base_salaries = {
        "IC1": 55_000, "IC2": 72_000, "IC3": 92_000, "IC4": 115_000, "IC5": 140_000,
        "Manager": 120_000, "Director": 160_000, "VP": 210_000,
    }
    dept_multipliers = {
        "Engineering": 1.15, "Sales": 1.05, "Marketing": 1.00,
        "HR": 0.95, "Finance": 1.05, "Operations": 0.95,
    }
    base = base_salaries[role_level] * dept_multipliers[department]
    return round(base * rng.uniform(0.85, 1.15), -2)


def _compute_attrition_probability(row: pd.Series) -> float:
    """Compute attrition probability based on realistic HR drivers."""
    prob = 0.08  # base annual attrition rate (~8%)

    # Low satisfaction is the strongest driver
    if row["satisfaction_score"] < 3.0:
        prob += 0.15
    elif row["satisfaction_score"] < 4.0:
        prob += 0.05

    # Low engagement compounds with low satisfaction
    if row["engagement_score"] < 3.0:
        prob += 0.10

    # No promotion in 3 years for tenured employees
    if row["promotion_last_3y"] == 0 and row["tenure_months"] > 24:
        prob += 0.12

    # Chronic overtime
    if row["overtime_flag"] == 1:
        prob += 0.08

    # Extreme work hours
    if row["work_hours_weekly"] > 50:
        prob += 0.06

    # Low compensation relative to level
    if row["salary_percentile"] < 25:
        prob += 0.10
    elif row["salary_percentile"] < 40:
        prob += 0.04

    # Poor performance rating (involuntary attrition)
    if row["performance_rating"] <= 2:
        prob += 0.12

    # Low last raise
    if row["last_raise_pct"] < 2.0:
        prob += 0.05

    # Long commute with no remote days
    if row["commute_distance"] > 30 and row["remote_days"] == 0:
        prob += 0.06

    # High performers with no stock options tend to leave
    if row["performance_rating"] >= 4 and row["stock_options"] == 0:
        prob += 0.07

    # Very short tenure (< 6 months) — probationary churn
    if row["tenure_months"] < 6:
        prob += 0.04

    # Very long tenure dampens attrition
    if row["tenure_months"] > 60:
        prob -= 0.06

    # Senior roles have lower attrition
    if row["role_level"] in ("Director", "VP"):
        prob -= 0.05

    return np.clip(prob, 0.02, 0.70)


def generate_hr_data(seed: int = SEED, n_employees: int = N_EMPLOYEES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
    dept_weights = [0.30, 0.20, 0.12, 0.10, 0.13, 0.15]
    role_levels = ["IC1", "IC2", "IC3", "IC4", "IC5", "Manager", "Director", "VP"]
    role_weights = [0.15, 0.25, 0.22, 0.15, 0.08, 0.08, 0.05, 0.02]
    genders = ["Male", "Female", "Non-Binary"]
    gender_weights = [0.48, 0.46, 0.06]

    records = []
    for i in range(n_employees):
        emp_id = f"EMP-{i + 1:05d}"
        department = rng.choice(departments, p=dept_weights)
        role_level = rng.choice(role_levels, p=role_weights)
        gender = rng.choice(genders, p=gender_weights)

        # Age correlated with role level
        age_base = {"IC1": 24, "IC2": 27, "IC3": 30, "IC4": 34, "IC5": 38,
                     "Manager": 33, "Director": 40, "VP": 45}
        age = int(np.clip(rng.normal(age_base[role_level], 4), 21, 62))

        tenure_months = int(np.clip(rng.exponential(30), 1, OBSERVATION_MONTHS * 3))
        salary = _salary_by_level(role_level, department, rng)

        # Performance and satisfaction with some correlation
        performance_base = rng.normal(3.5, 0.8)
        performance_rating = int(np.clip(round(performance_base), 1, 5))
        satisfaction_base = performance_base * 0.4 + rng.normal(2.1, 0.7)
        satisfaction_score = round(np.clip(satisfaction_base, 1.0, 5.0), 1)
        engagement_score = round(np.clip(satisfaction_score * 0.6 + rng.normal(1.5, 0.6), 1.0, 5.0), 1)

        work_hours_weekly = int(np.clip(rng.normal(42, 6), 35, 65))
        overtime_flag = 1 if work_hours_weekly > 45 else 0
        remote_days = int(rng.choice([0, 1, 2, 3, 5], p=[0.15, 0.20, 0.30, 0.25, 0.10]))
        commute_distance = round(rng.exponential(15), 1)

        num_direct_reports = 0
        if role_level in ("Manager", "Director", "VP"):
            num_direct_reports = int(rng.integers(3, 15))
        team_size = int(rng.integers(4, 25))
        manager_tenure = int(np.clip(rng.exponential(24), 3, 120))

        promotion_last_3y = int(rng.choice([0, 1, 2], p=[0.55, 0.35, 0.10]))
        time_since_last_promo = int(np.clip(rng.exponential(18), 0, 72)) if promotion_last_3y > 0 else int(rng.integers(36, 84))
        last_raise_pct = round(np.clip(rng.normal(3.5, 2.0), 0.0, 15.0), 1)
        stock_options = int(rng.choice([0, 1, 2, 3], p=[0.40, 0.30, 0.20, 0.10]))
        training_hours_annual = int(np.clip(rng.normal(25, 12), 0, 80))
        num_projects = int(np.clip(rng.poisson(4), 1, 12))
        peer_rating = round(np.clip(rng.normal(3.8, 0.6), 1.0, 5.0), 1)
        skip_level_meeting_freq = int(rng.choice([0, 1, 2, 4], p=[0.30, 0.35, 0.25, 0.10]))

        records.append({
            "employee_id": emp_id,
            "age": age,
            "gender": gender,
            "department": department,
            "role_level": role_level,
            "tenure_months": tenure_months,
            "salary": salary,
            "promotion_last_3y": promotion_last_3y,
            "performance_rating": performance_rating,
            "satisfaction_score": satisfaction_score,
            "engagement_score": engagement_score,
            "work_hours_weekly": work_hours_weekly,
            "overtime_flag": overtime_flag,
            "remote_days": remote_days,
            "commute_distance": commute_distance,
            "num_direct_reports": num_direct_reports,
            "team_size": team_size,
            "manager_tenure": manager_tenure,
            "last_raise_pct": last_raise_pct,
            "stock_options": stock_options,
            "training_hours_annual": training_hours_annual,
            "num_projects": num_projects,
            "peer_rating": peer_rating,
            "skip_level_meeting_freq": skip_level_meeting_freq,
            "time_since_last_promo": time_since_last_promo,
        })

    df = pd.DataFrame(records)

    # Compute salary percentile within department + role_level
    df["salary_percentile"] = df.groupby(["department", "role_level"])["salary"].rank(pct=True).mul(100).round(0).astype(int)

    # Compute attrition labels
    df["attrition_prob"] = df.apply(_compute_attrition_probability, axis=1)
    df["attrition"] = (rng.random(len(df)) < df["attrition_prob"]).astype(int)

    # Generate attrition date for survival analysis (months from observation start)
    df["event_observed"] = df["attrition"]
    df["duration_months"] = OBSERVATION_MONTHS  # default: survived full observation
    attrited_mask = df["attrition"] == 1
    n_attrited = attrited_mask.sum()
    df.loc[attrited_mask, "duration_months"] = rng.integers(1, OBSERVATION_MONTHS, size=n_attrited)
    df["attrition_date"] = pd.NaT
    observation_start = pd.Timestamp("2023-01-01")
    df.loc[attrited_mask, "attrition_date"] = df.loc[attrited_mask, "duration_months"].apply(
        lambda m: observation_start + pd.DateOffset(months=int(m))
    )

    # Drop helper column
    df = df.drop(columns=["attrition_prob"])

    return df


def main():
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic HR data...")
    df = generate_hr_data()

    output_path = output_dir / "hr_employee_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} employee records")
    print(f"Attrition rate: {df['attrition'].mean():.1%}")
    print(f"Saved to: {output_path}")
    print(f"\nDepartment breakdown:")
    print(df["department"].value_counts().to_string())
    print(f"\nAttrition by department:")
    print(df.groupby("department")["attrition"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
