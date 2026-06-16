# Employee Attrition Analytics & Workforce Planning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM-orange)
![Fairness](https://img.shields.io/badge/Fairness-Fairlearn-purple)
![Survival](https://img.shields.io/badge/Survival-Lifelines-red)

**ML-powered employee attrition prediction** with survival analysis, flight risk scoring, retention intervention recommender, and DEI fairness auditing. Built for HR/People Analytics teams to proactively identify and retain at-risk talent while ensuring equitable outcomes across demographic groups.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HR Attrition Analytics Platform                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌────────────────────┐  │
│  │  Raw HR Data  │──▶│ Feature Engineering│──▶│  Model Training    │  │
│  │  (5K employees│   │ - Comp ratios     │   │  - LR / RF / XGB   │  │
│  │   3 yrs data) │   │ - Engagement score│   │  - LightGBM        │  │
│  │              │   │ - Manager quality  │   │  - Cox PH Survival  │  │
│  │              │   │ - Flight risk flags│   │  - Fairness Audit   │  │
│  └──────────────┘   └──────────────────┘   └─────────┬──────────┘  │
│                                                       │             │
│                          ┌────────────────────────────┘             │
│                          ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Evaluation & Analysis                       │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐  │  │
│  │  │ ROC/PR Curves│  │ SHAP Values  │  │ Survival Curves     │  │  │
│  │  │ Confusion Mat│  │ Feature Imp. │  │ Kaplan-Meier / Cox  │  │  │
│  │  └─────────────┘  └──────────────┘  └─────────────────────┘  │  │
│  │  ┌─────────────────────┐  ┌────────────────────────────────┐  │  │
│  │  │ Fairness Dashboard  │  │ Cost of Attrition Analysis     │  │  │
│  │  │ Demographic Parity  │  │ Replacement cost modeling      │  │  │
│  │  │ Equalized Odds      │  │ Revenue impact projections     │  │  │
│  │  └─────────────────────┘  └────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                          │                                          │
│              ┌───────────┴───────────┐                              │
│              ▼                       ▼                              │
│  ┌────────────────────┐  ┌────────────────────────┐                 │
│  │   FastAPI Service   │  │  Streamlit Dashboard   │                 │
│  │  /predict/attrition │  │  Flight Risk Heatmap   │                 │
│  │  /score/flight-risk │  │  Employee Risk Profile │                 │
│  │  /survival/{emp_id} │  │  Retention Recommender │                 │
│  │  /fairness/report   │  │  Workforce Planning    │                 │
│  └────────────────────┘  │  DEI Fairness Dash     │                 │
│                          └────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Results

| Model | ROC-AUC | PR-AUC | F1 (Attrition) | Recall@80%Prec |
|-------|---------|--------|----------------|----------------|
| Logistic Regression | 0.82 | 0.61 | 0.65 | 0.52 |
| Random Forest | 0.87 | 0.71 | 0.72 | 0.63 |
| XGBoost | 0.90 | 0.76 | 0.76 | 0.69 |
| **LightGBM** | **0.91** | **0.78** | **0.77** | **0.71** |

| Survival Analysis | C-Index | IBS |
|-------------------|---------|-----|
| Cox PH Model | 0.83 | 0.12 |

| Fairness Metric | Gender | Age Group |
|-----------------|--------|-----------|
| Demographic Parity Ratio | 0.94 | 0.88 |
| Equalized Odds Diff | 0.04 | 0.07 |
| Disparate Impact Ratio | 0.96 | 0.91 |

## Features

- **Attrition Prediction**: Multi-model ensemble with calibrated probabilities
- **Survival Analysis**: Time-to-attrition modeling with Kaplan-Meier and Cox PH
- **Flight Risk Scoring**: 0-100 composite risk score per employee
- **Retention Interventions**: Data-driven recommendations based on SHAP drivers
- **DEI Fairness Auditing**: Demographic parity, equalized odds, disparate impact
- **Cost Modeling**: Replacement cost estimation per role level and department
- **Workforce Planning**: Predicted headcount and budget impact projections

## Project Structure

```
hr-attrition-analytics/
├── data/
│   ├── raw/                  # Generated HR dataset
│   └── processed/            # Engineered features
├── models/                   # Saved model artifacts
├── results/
│   ├── figures/              # Plots and visualizations
│   └── reports/              # Fairness and evaluation reports
├── src/
│   ├── data/
│   │   └── generate_data.py  # Synthetic HR data generator
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── attrition_classifier.py
│   │   ├── survival_analysis.py
│   │   ├── fairness_audit.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── api/
│   │   └── app.py            # FastAPI service
│   └── dashboard/
│       └── app.py            # Streamlit dashboard
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
make install

# Generate synthetic HR data
make data

# Run feature engineering
make features

# Train models & run fairness audit
make train

# Evaluate and generate reports
make evaluate

# Launch API
make api

# Launch dashboard
make dashboard
```

## Ethical AI & Fairness

This project integrates fairness auditing at every stage:

1. **Pre-training**: Analyze feature distributions across demographic groups
2. **Post-training**: Measure demographic parity, equalized odds, and disparate impact
3. **Deployment**: Continuous monitoring of prediction fairness
4. **Interventions**: Bias-aware retention recommendations

We use [Fairlearn](https://fairlearn.org/) to ensure model predictions do not disproportionately impact any demographic group.

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Models | Scikit-Learn, XGBoost, LightGBM |
| Survival Analysis | Lifelines |
| Explainability | SHAP |
| Fairness | Fairlearn |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |

## Author

**Naresh Sampangi**

## License

This project is licensed under the MIT License.
