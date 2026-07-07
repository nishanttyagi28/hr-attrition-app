# HR Employee Attrition Analysis & Prediction

A Streamlit dashboard for exploring employee attrition patterns and predicting attrition risk using a Random Forest classifier. The app uses a synthetic HR dataset (~1,200 records) with realistic correlations — no external data download required.

## Features

- **Overview** — Dataset summary, attrition rate, and key metric cards
- **EDA** — Interactive Plotly charts (attrition by department, role, overtime, satisfaction; correlation heatmap; income distribution)
- **Prediction Model** — Random Forest evaluation with accuracy, precision, recall, F1, confusion matrix, and feature importance
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

On first launch, synthetic employee data is generated and saved to `data/hr_employees.csv`. Subsequent runs load the cached CSV automatically.

## Dataset

The synthetic dataset includes 1,000–1,500 employee records with columns such as Age, Department, JobRole, MonthlyIncome, YearsAtCompany, JobSatisfaction, OverTime, and Attrition (Yes/No). Attrition outcomes are modeled with realistic correlations — for example, low satisfaction combined with overtime and lower income increases attrition probability.