"""Generate synthetic HR employee data with realistic attrition correlations."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "hr_employees.csv"

DEPARTMENTS = {
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development": ["Research Scientist", "Laboratory Technician", "Manufacturing Director", "Research Director"],
    "Human Resources": ["Human Resources", "Manager"],
}

EDUCATION_LEVELS = [
    "Below College",
    "College",
    "Bachelor",
    "Master",
    "Doctor",
]

MARITAL_STATUS = ["Single", "Married", "Divorced"]
BUSINESS_TRAVEL = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]


def _attrition_probability(row: dict) -> float:
    """Compute attrition probability from employee features."""
    score = -4.3

    if row["JobSatisfaction"] <= 2:
        score += 1.2
    if row["WorkLifeBalance"] <= 2:
        score += 0.9
    if row["OverTime"] == "Yes":
        score += 1.1
    if row["MonthlyIncome"] < 3500:
        score += 0.9
    elif row["MonthlyIncome"] < 5000:
        score += 0.35
    if row["DistanceFromHome"] > 15:
        score += 0.7
    elif row["DistanceFromHome"] > 10:
        score += 0.35
    if row["YearsAtCompany"] < 2:
        score += 0.8
    elif row["YearsAtCompany"] < 5:
        score += 0.25
    if row["YearsInCurrentRole"] >= row["YearsAtCompany"] - 1 and row["YearsAtCompany"] > 3:
        score -= 0.6
    if row["PerformanceRating"] >= 4:
        score -= 0.5
    if row["NumCompaniesWorked"] >= 4:
        score += 0.6
    if row["Age"] < 30:
        score += 0.4
    if row["JobRole"] in ("Sales Representative", "Laboratory Technician"):
        score += 0.35
    if row["BusinessTravel"] == "Travel_Frequently":
        score += 0.45
    if row["MaritalStatus"] == "Single":
        score += 0.25

    return 1 / (1 + np.exp(-score))


def generate_hr_data(n_records: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic HR dataset with correlated attrition outcomes."""
    rng = np.random.default_rng(seed)

    records = []
    for _ in range(n_records):
        age = int(rng.integers(22, 61))
        department = rng.choice(list(DEPARTMENTS.keys()))
        job_role = rng.choice(DEPARTMENTS[department])
        total_working_years = int(rng.integers(max(0, age - 22), min(age - 18, 40) + 1))
        years_at_company = int(rng.integers(0, min(total_working_years + 1, 25)))
        years_in_current_role = int(rng.integers(0, min(years_at_company + 1, 15)))
        num_companies = int(rng.integers(1, min(total_working_years + 2, 10)))

        base_income = {
            "Sales Executive": 4500,
            "Sales Representative": 3200,
            "Manager": 7500,
            "Research Scientist": 5500,
            "Laboratory Technician": 3800,
            "Manufacturing Director": 9000,
            "Research Director": 8500,
            "Human Resources": 4200,
        }[job_role]
        monthly_income = int(rng.normal(base_income, base_income * 0.15))
        monthly_income = max(2500, monthly_income)

        distance = int(rng.integers(1, 31))
        job_satisfaction = int(rng.integers(1, 5))
        work_life_balance = int(rng.integers(1, 5))
        overtime = rng.choice(["Yes", "No"], p=[0.28, 0.72])
        performance = int(rng.choice([3, 4], p=[0.7, 0.3]))
        education = rng.choice(EDUCATION_LEVELS, p=[0.05, 0.15, 0.45, 0.28, 0.07])
        marital = rng.choice(MARITAL_STATUS, p=[0.35, 0.55, 0.10])
        travel = rng.choice(BUSINESS_TRAVEL, p=[0.15, 0.65, 0.20])

        row = {
            "Age": age,
            "Department": department,
            "JobRole": job_role,
            "MonthlyIncome": monthly_income,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "DistanceFromHome": distance,
            "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life_balance,
            "OverTime": overtime,
            "NumCompaniesWorked": num_companies,
            "TotalWorkingYears": total_working_years,
            "PerformanceRating": performance,
            "Education": education,
            "MaritalStatus": marital,
            "BusinessTravel": travel,
        }
        prob = _attrition_probability(row)
        row["Attrition"] = "Yes" if rng.random() < prob else "No"
        records.append(row)

    return pd.DataFrame(records)


def load_or_generate_data(force_regenerate: bool = False) -> pd.DataFrame:
    """Load cached CSV or generate synthetic data on first run."""
    if DATA_PATH.exists() and not force_regenerate:
        return pd.read_csv(DATA_PATH)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_hr_data()
    df.to_csv(DATA_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load_or_generate_data(force_regenerate=True)
    print(f"Generated {len(df)} records -> {DATA_PATH}")
    print(f"Attrition rate: {df['Attrition'].eq('Yes').mean():.1%}")