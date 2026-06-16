"""
HR-specific feature engineering pipeline.

Transforms raw employee data into ML-ready features with domain-specific
composite scores, compensation ratios, manager quality proxies, and
flight risk indicators.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def compute_compensation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute salary-relative features."""
    # Salary vs department median
    dept_median = df.groupby("department")["salary"].transform("median")
    df["salary_vs_dept_median"] = round(df["salary"] / dept_median, 3)

    # Salary vs role-level median
    level_median = df.groupby("role_level")["salary"].transform("median")
    df["salary_vs_level_median"] = round(df["salary"] / level_median, 3)

    # Salary vs team (department + role_level) median
    team_median = df.groupby(["department", "role_level"])["salary"].transform("median")
    df["salary_vs_team_median"] = round(df["salary"] / team_median, 3)

    # Compensation gap: difference from team median in thousands
    df["comp_gap_k"] = round((df["salary"] - team_median) / 1000, 1)

    # Raise-to-performance ratio
    df["raise_per_perf_point"] = round(
        df["last_raise_pct"] / df["performance_rating"].clip(lower=1), 2
    )

    # Total compensation proxy (salary + stock options value estimate)
    stock_value_map = {0: 0, 1: 5000, 2: 15000, 3: 30000}
    df["estimated_total_comp"] = df["salary"] + df["stock_options"].map(stock_value_map)

    return df


def compute_engagement_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engagement composite score from multiple signals."""
    # Normalize components to 0-1 scale
    sat_norm = (df["satisfaction_score"] - 1) / 4
    eng_norm = (df["engagement_score"] - 1) / 4
    peer_norm = (df["peer_rating"] - 1) / 4
    training_norm = df["training_hours_annual"] / df["training_hours_annual"].max()
    skip_norm = df["skip_level_meeting_freq"] / df["skip_level_meeting_freq"].max()

    # Weighted composite
    df["engagement_composite"] = round(
        sat_norm * 0.30 + eng_norm * 0.30 + peer_norm * 0.15 +
        training_norm * 0.15 + skip_norm * 0.10, 3
    )

    # Engagement trend proxy (satisfaction - engagement gap)
    df["sat_eng_gap"] = round(df["satisfaction_score"] - df["engagement_score"], 2)

    return df


def compute_manager_quality_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Proxy features for manager/team quality."""
    # Manager tenure relative to employee tenure
    df["manager_tenure_ratio"] = round(
        df["manager_tenure"] / df["tenure_months"].clip(lower=1), 2
    )

    # Team-level aggregates
    dept_level_group = df.groupby(["department", "role_level"])
    df["team_avg_satisfaction"] = dept_level_group["satisfaction_score"].transform("mean").round(2)
    df["team_avg_performance"] = dept_level_group["performance_rating"].transform("mean").round(2)
    df["team_attrition_rate"] = dept_level_group["attrition"].transform("mean").round(3)

    # Manager span of control (for managers)
    df["span_of_control"] = df["num_direct_reports"] / df["team_size"].clip(lower=1)
    df["span_of_control"] = df["span_of_control"].round(2)

    return df


def compute_flight_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Binary and ordinal flight risk indicator features."""
    # Stagnation flag: no promotion + long tenure + low raise
    df["stagnation_flag"] = (
        (df["promotion_last_3y"] == 0) &
        (df["tenure_months"] > 24) &
        (df["last_raise_pct"] < 3.0)
    ).astype(int)

    # Burnout flag: high hours + overtime + low satisfaction
    df["burnout_flag"] = (
        (df["work_hours_weekly"] > 48) &
        (df["overtime_flag"] == 1) &
        (df["satisfaction_score"] < 3.5)
    ).astype(int)

    # Compensation dissatisfaction: low pay percentile + low raise
    df["comp_dissatisfaction_flag"] = (
        (df["salary_percentile"] < 35) &
        (df["last_raise_pct"] < 2.5)
    ).astype(int)

    # High performer at risk: good rating + low satisfaction + no stock
    df["high_perf_at_risk"] = (
        (df["performance_rating"] >= 4) &
        (df["satisfaction_score"] < 3.5) &
        (df["stock_options"] == 0)
    ).astype(int)

    # Isolation flag: no skip-level meetings + low engagement
    df["isolation_flag"] = (
        (df["skip_level_meeting_freq"] == 0) &
        (df["engagement_score"] < 3.0)
    ).astype(int)

    # Commute burden: long commute + no remote
    df["commute_burden_flag"] = (
        (df["commute_distance"] > 25) &
        (df["remote_days"] <= 1)
    ).astype(int)

    # Composite risk indicator count
    risk_flags = [
        "stagnation_flag", "burnout_flag", "comp_dissatisfaction_flag",
        "high_perf_at_risk", "isolation_flag", "commute_burden_flag"
    ]
    df["risk_flag_count"] = df[risk_flags].sum(axis=1)

    return df


def compute_tenure_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize tenure into HR-standard buckets."""
    bins = [0, 6, 12, 24, 36, 60, 120, 999]
    labels = ["0-6mo", "6-12mo", "1-2yr", "2-3yr", "3-5yr", "5-10yr", "10yr+"]
    df["tenure_bucket"] = pd.cut(df["tenure_months"], bins=bins, labels=labels, right=True)
    return df


def compute_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Age grouping for fairness analysis."""
    bins = [20, 30, 40, 50, 65]
    labels = ["20-29", "30-39", "40-49", "50+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode role_level, one-hot encode remaining categoricals."""
    role_order = {"IC1": 1, "IC2": 2, "IC3": 3, "IC4": 4, "IC5": 5,
                  "Manager": 6, "Director": 7, "VP": 8}
    df["role_level_encoded"] = df["role_level"].map(role_order)

    df = pd.get_dummies(df, columns=["department"], prefix="dept", dtype=int)

    return df


def run_feature_engineering(input_path: str, output_path: str) -> pd.DataFrame:
    """Execute the full feature engineering pipeline."""
    df = pd.read_csv(input_path)

    df = compute_compensation_features(df)
    df = compute_engagement_composite(df)
    df = compute_manager_quality_proxies(df)
    df = compute_flight_risk_indicators(df)
    df = compute_tenure_buckets(df)
    df = compute_age_groups(df)
    df = encode_categoricals(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "raw" / "hr_employee_data.csv"
    output_path = project_root / "data" / "processed" / "hr_features.csv"

    print("Running feature engineering pipeline...")
    df = run_feature_engineering(str(input_path), str(output_path))

    print(f"Processed {len(df)} records with {len(df.columns)} features")
    print(f"New features added: {len(df.columns)} total columns")
    print(f"Risk flag distribution:\n{df['risk_flag_count'].value_counts().sort_index().to_string()}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
