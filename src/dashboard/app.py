"""
Streamlit dashboard for HR Attrition Analytics.

Provides interactive visualizations for flight risk heatmaps, individual
employee risk profiles, retention recommendations, and DEI fairness metrics.
"""

import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="HR Attrition Analytics",
    page_icon="👥",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


@st.cache_data
def load_data():
    """Load scored employee data."""
    scored_path = DATA_DIR / "hr_scored.csv"
    features_path = DATA_DIR / "hr_features.csv"
    if scored_path.exists():
        return pd.read_csv(scored_path)
    elif features_path.exists():
        return pd.read_csv(features_path)
    else:
        st.error("No data found. Run `make data` and `make train` first.")
        st.stop()


@st.cache_resource
def load_model(name: str):
    """Load a trained model artifact."""
    path = MODELS_DIR / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data
def load_metrics():
    """Load training metrics."""
    path = RESULTS_DIR / "reports" / "training_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def sidebar():
    """Render sidebar navigation."""
    st.sidebar.title("👥 HR Analytics")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Flight Risk Heatmap", "Employee Risk Profile",
         "Retention Recommender", "DEI Fairness Dashboard"],
    )
    return page


def page_overview(df: pd.DataFrame):
    """Overview page with key metrics."""
    st.title("📊 Workforce Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", f"{len(df):,}")
    col2.metric("Attrition Rate", f"{df['attrition'].mean():.1%}")

    if "flight_risk_score" in df.columns:
        critical = (df["flight_risk_tier"] == "Critical").sum()
        high = (df["flight_risk_tier"] == "High").sum()
        col3.metric("Critical Risk", critical)
        col4.metric("High Risk", high)
    else:
        col3.metric("Avg Satisfaction", f"{df['satisfaction_score'].mean():.2f}")
        col4.metric("Avg Engagement", f"{df['engagement_score'].mean():.2f}")

    st.markdown("---")

    # Attrition by department
    col_left, col_right = st.columns(2)

    with col_left:
        dept_attrition = df.groupby("department")["attrition"].mean().reset_index()
        dept_attrition.columns = ["Department", "Attrition Rate"]
        fig = px.bar(
            dept_attrition, x="Department", y="Attrition Rate",
            color="Attrition Rate", color_continuous_scale="Reds",
            title="Attrition Rate by Department",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        if "role_level" in df.columns:
            level_attrition = df.groupby("role_level")["attrition"].mean().reset_index()
            level_attrition.columns = ["Role Level", "Attrition Rate"]
            fig = px.bar(
                level_attrition, x="Role Level", y="Attrition Rate",
                color="Attrition Rate", color_continuous_scale="Reds",
                title="Attrition Rate by Role Level",
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    # Model performance summary
    metrics = load_metrics()
    if metrics and "classifier_metrics" in metrics:
        st.subheader("Model Performance")
        perf_data = []
        for model_name, m in metrics["classifier_metrics"].items():
            perf_data.append({
                "Model": model_name.replace("_", " ").title(),
                "ROC-AUC": m.get("roc_auc", 0),
                "PR-AUC": m.get("pr_auc", 0),
                "F1": m.get("f1", 0),
                "Precision": m.get("precision", 0),
                "Recall": m.get("recall", 0),
            })
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)


def page_flight_risk_heatmap(df: pd.DataFrame):
    """Flight risk heatmap visualization."""
    st.title("🔥 Flight Risk Heatmap")

    if "flight_risk_score" not in df.columns:
        st.warning("Flight risk scores not available. Run `make train` first.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        dept_filter = st.multiselect("Department", df["department"].unique(), default=df["department"].unique())
    with col2:
        risk_filter = st.multiselect("Risk Tier", ["Critical", "High", "Medium", "Low"], default=["Critical", "High"])

    filtered = df[df["department"].isin(dept_filter) & df["flight_risk_tier"].isin(risk_filter)]

    # Heatmap: department x role_level
    if "role_level" in filtered.columns:
        pivot = filtered.groupby(["department", "role_level"])["flight_risk_score"].mean().reset_index()
        pivot_wide = pivot.pivot(index="department", columns="role_level", values="flight_risk_score")

        fig = px.imshow(
            pivot_wide,
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            title="Average Flight Risk Score: Department × Role Level",
            labels={"color": "Risk Score"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Top at-risk employees table
    st.subheader(f"Top At-Risk Employees ({len(filtered)} matching)")
    display_cols = ["employee_id", "department", "role_level", "tenure_months",
                    "satisfaction_score", "flight_risk_score", "flight_risk_tier"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    top_risk = filtered.nlargest(20, "flight_risk_score")[display_cols]
    st.dataframe(top_risk, use_container_width=True, hide_index=True)

    # Distribution
    fig = px.histogram(
        filtered, x="flight_risk_score", color="flight_risk_tier",
        nbins=50, title="Flight Risk Score Distribution",
        color_discrete_map={"Critical": "#e74c3c", "High": "#f39c12", "Medium": "#3498db", "Low": "#2ecc71"},
    )
    st.plotly_chart(fig, use_container_width=True)


def page_employee_risk_profile(df: pd.DataFrame):
    """Individual employee risk profile."""
    st.title("🔍 Employee Risk Profile")

    if "employee_id" not in df.columns:
        st.warning("Employee data not available.")
        return

    emp_id = st.selectbox("Select Employee", df["employee_id"].sort_values().unique())
    emp = df[df["employee_id"] == emp_id].iloc[0]

    # Profile header
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Department", emp.get("department", "N/A"))
    col2.metric("Role Level", emp.get("role_level", "N/A"))
    col3.metric("Tenure", f"{emp.get('tenure_months', 0)} months")

    if "flight_risk_score" in emp.index:
        col4.metric("Flight Risk", f"{emp['flight_risk_score']:.0f}/100",
                     delta=emp.get("flight_risk_tier", ""))

    st.markdown("---")

    # Key metrics radar chart
    categories = ["Satisfaction", "Engagement", "Performance", "Peer Rating", "Work-Life"]
    values = [
        emp.get("satisfaction_score", 3) / 5,
        emp.get("engagement_score", 3) / 5,
        emp.get("performance_rating", 3) / 5,
        emp.get("peer_rating", 3) / 5,
        1 - min(emp.get("work_hours_weekly", 40), 60) / 60,
    ]
    values.append(values[0])  # close the polygon
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill="toself", name=emp_id))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Employee Profile: {emp_id}",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detail table
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Compensation")
        st.write(f"- **Salary**: ${emp.get('salary', 0):,.0f}")
        st.write(f"- **Salary Percentile**: {emp.get('salary_percentile', 0)}th")
        st.write(f"- **Last Raise**: {emp.get('last_raise_pct', 0)}%")
        st.write(f"- **Stock Options**: Level {emp.get('stock_options', 0)}")

    with col_right:
        st.subheader("Risk Indicators")
        flags = {
            "Stagnation": emp.get("stagnation_flag", 0),
            "Burnout": emp.get("burnout_flag", 0),
            "Comp Dissatisfaction": emp.get("comp_dissatisfaction_flag", 0),
            "High Performer at Risk": emp.get("high_perf_at_risk", 0),
            "Isolation": emp.get("isolation_flag", 0),
            "Commute Burden": emp.get("commute_burden_flag", 0),
        }
        for flag_name, flag_val in flags.items():
            icon = "🔴" if flag_val else "🟢"
            st.write(f"- {icon} {flag_name}")


def page_retention_recommender(df: pd.DataFrame):
    """Data-driven retention intervention recommendations."""
    st.title("💡 Retention Recommender")

    if "flight_risk_score" not in df.columns:
        st.warning("Flight risk scores not available. Run `make train` first.")
        return

    # Filter to high/critical risk employees
    at_risk = df[df["flight_risk_tier"].isin(["Critical", "High"])].copy()
    st.info(f"**{len(at_risk)}** employees in Critical or High risk tiers")

    if len(at_risk) == 0:
        st.success("No high-risk employees found!")
        return

    # Generate recommendations based on risk flags
    recommendations = []
    for _, row in at_risk.iterrows():
        rec = {"employee_id": row["employee_id"], "department": row.get("department", ""),
               "risk_score": row["flight_risk_score"], "interventions": []}

        if row.get("stagnation_flag", 0):
            rec["interventions"].append("📈 Career path discussion + promotion consideration")
        if row.get("burnout_flag", 0):
            rec["interventions"].append("⏰ Workload rebalancing + mandatory PTO")
        if row.get("comp_dissatisfaction_flag", 0):
            rec["interventions"].append("💰 Compensation adjustment / retention bonus")
        if row.get("high_perf_at_risk", 0):
            rec["interventions"].append("🎯 Stock option grant + stretch project assignment")
        if row.get("isolation_flag", 0):
            rec["interventions"].append("🤝 Mentorship pairing + skip-level check-ins")
        if row.get("commute_burden_flag", 0):
            rec["interventions"].append("🏠 Remote work flexibility increase")
        if not rec["interventions"]:
            rec["interventions"].append("📋 Conduct stay interview to identify concerns")

        recommendations.append(rec)

    rec_df = pd.DataFrame(recommendations)
    rec_df["interventions"] = rec_df["interventions"].apply(lambda x: "\n".join(x))

    # Summary by intervention type
    st.subheader("Intervention Counts")
    intervention_counts = {}
    for recs in recommendations:
        for intervention in recs["interventions"].split("\n") if isinstance(recs["interventions"], str) else recs["interventions"]:
            intervention_counts[intervention] = intervention_counts.get(intervention, 0) + 1

    if intervention_counts:
        count_df = pd.DataFrame([
            {"Intervention": k, "Count": v}
            for k, v in sorted(intervention_counts.items(), key=lambda x: -x[1])
        ])
        fig = px.bar(count_df, x="Count", y="Intervention", orientation="h",
                     title="Most Common Recommended Interventions", color="Count",
                     color_continuous_scale="Blues")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("Individual Recommendations")
    st.dataframe(
        rec_df.sort_values("risk_score", ascending=False).head(30),
        use_container_width=True, hide_index=True,
    )

    # Cost impact estimation
    st.subheader("Estimated Retention ROI")
    cost_path = RESULTS_DIR / "reports" / "cost_of_attrition.json"
    if cost_path.exists():
        with open(cost_path) as f:
            cost_data = json.load(f)
        avg_cost = cost_data.get("avg_cost_per_departure", 80000)
        st.write(f"- Average replacement cost per departure: **${avg_cost:,.0f}**")
        st.write(f"- Potential savings from retaining {len(at_risk)} at-risk employees: **${avg_cost * len(at_risk):,.0f}**")


def page_dei_fairness(df: pd.DataFrame):
    """DEI fairness dashboard."""
    st.title("⚖️ DEI Fairness Dashboard")

    # Load fairness metrics if available
    metrics = load_metrics()
    fairness_report_path = RESULTS_DIR / "reports" / "training_metrics.json"

    # Attrition rates by gender
    col1, col2 = st.columns(2)

    with col1:
        if "gender" in df.columns:
            gender_attrition = df.groupby("gender").agg(
                count=("attrition", "count"),
                attrition_rate=("attrition", "mean"),
                avg_satisfaction=("satisfaction_score", "mean"),
            ).reset_index()
            gender_attrition.columns = ["Gender", "Count", "Attrition Rate", "Avg Satisfaction"]

            fig = px.bar(gender_attrition, x="Gender", y="Attrition Rate",
                         color="Gender", title="Attrition Rate by Gender",
                         text_auto=".1%")
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(gender_attrition, use_container_width=True, hide_index=True)

    with col2:
        if "age_group" in df.columns:
            age_attrition = df.groupby("age_group").agg(
                count=("attrition", "count"),
                attrition_rate=("attrition", "mean"),
                avg_satisfaction=("satisfaction_score", "mean"),
            ).reset_index()
            age_attrition.columns = ["Age Group", "Count", "Attrition Rate", "Avg Satisfaction"]

            fig = px.bar(age_attrition, x="Age Group", y="Attrition Rate",
                         color="Age Group", title="Attrition Rate by Age Group",
                         text_auto=".1%")
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(age_attrition, use_container_width=True, hide_index=True)

    # Compensation equity
    st.markdown("---")
    st.subheader("Compensation Equity Analysis")

    if "gender" in df.columns and "salary" in df.columns:
        fig = px.box(df, x="gender", y="salary", color="gender",
                     title="Salary Distribution by Gender",
                     points="suspectedoutliers")
        st.plotly_chart(fig, use_container_width=True)

    if "gender" in df.columns and "role_level" in df.columns:
        pay_gap = df.groupby(["role_level", "gender"])["salary"].median().unstack()
        if "Male" in pay_gap.columns and "Female" in pay_gap.columns:
            pay_gap["Pay Gap %"] = ((pay_gap["Male"] - pay_gap["Female"]) / pay_gap["Male"] * 100).round(1)
            st.write("**Pay Gap by Role Level** (Male median - Female median as % of Male median)")
            st.dataframe(pay_gap.round(0), use_container_width=True)

    # Fairness metrics from training results
    if metrics and "fairness_report" in metrics:
        st.markdown("---")
        st.subheader("Model Fairness Audit Results")
        fr = metrics["fairness_report"]
        if "recommendations" in fr:
            for rec in fr["recommendations"]:
                severity_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(rec.get("severity", ""), "⚪")
                st.write(f"{severity_color} **{rec.get('issue', '')}** ({rec.get('attribute', '')})")
                st.write(f"  {rec.get('detail', '')}")
                st.write(f"  *Recommendation*: {rec.get('recommendation', '')}")
                st.write("")


def main():
    page = sidebar()
    df = load_data()

    if page == "Overview":
        page_overview(df)
    elif page == "Flight Risk Heatmap":
        page_flight_risk_heatmap(df)
    elif page == "Employee Risk Profile":
        page_employee_risk_profile(df)
    elif page == "Retention Recommender":
        page_retention_recommender(df)
    elif page == "DEI Fairness Dashboard":
        page_dei_fairness(df)


if __name__ == "__main__":
    main()
