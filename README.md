# HR Employee Attrition Analysis & Prediction

A Streamlit dashboard for exploring employee attrition patterns and predicting attrition risk using a Random Forest classifier. The app uses a synthetic HR dataset (~1,200 records) with realistic correlations — no external data download required.

## Features

- **Overview** — Dataset summary, attrition rate, and key metric cards
- **EDA** — Interactive Plotly charts (attrition by department, role, overtime, satisfaction; correlation heatmap; income distribution)
- **Prediction Model** — Tuned Random Forest with GridSearch, threshold optimization, before/after metrics comparison, confusion matrix, and feature importance
- **Predict for an Employee** — Input form for live attrition risk scoring with probability gauge

## Project Structure

```
hr-attrition-app/
├── app.py              # Streamlit dashboard
├── data_generator.py   # Synthetic HR data generation
├── model.py            # Random Forest training & prediction
├── requirements.txt    # Pinned dependencies
├── data/
│   └── hr_employees.csv  # Auto-generated on first run
└── README.md
```

## Prerequisites

- Python 3.10 or later
- pip

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/nishanttyagi28/hr-attrition-app.git
   cd hr-attrition-app
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Run

```bash
streamlit run app.py
```

The app opens in your browser at **http://localhost:8501**.

On first launch, synthetic employee data is generated and saved to `data/hr_employees.csv`. Subsequent runs load the cached CSV automatically. The first model training run takes ~1–2 minutes while GridSearch completes; results are cached afterward.

## Dataset

The synthetic dataset includes ~1,200 employee records with columns such as Age, Department, JobRole, MonthlyIncome, YearsAtCompany, JobSatisfaction, OverTime, and Attrition (Yes/No). Features are generated from a latent risk score so satisfaction, overtime, income, and tenure correlate naturally. Attrition outcomes use a logistic model with strong coefficients — for example, low satisfaction combined with overtime and lower income increases attrition probability (~22% overall rate).

## Model Performance

The Random Forest classifier is tuned via **GridSearchCV** (`n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`) and a **validation-set threshold** optimized for precision–recall balance. Evaluated on a held-out 25% test set:

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Accuracy | 72.7% | **88.0%** | +15.3% |
| Precision | 41.8% | **71.2%** | +29.4% |
| Recall | 62.1% | **77.6%** | +15.5% |
| F1 Score | 50.0% | **74.3%** | +24.3% |

**Before:** Default Random Forest with `class_weight='balanced'` and a 0.50 threshold (57 false positives on the test set).

**After:** Best params — `n_estimators=300`, `max_depth=10`, `min_samples_leaf=8`, `class_weight={0:1, 1:1.5}`, threshold **0.52** (21 false positives on the test set).

Improvements came from cleaner synthetic data correlations, hyperparameter tuning, and raising the classification threshold to reduce false positives.