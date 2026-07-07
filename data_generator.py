"""Generate synthetic HR employee data with realistic attrition correlations."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "hr_employees.csv"
DATA_VERSION = "v2"

DEPARTMENTS = {
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development": [
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Research Director",
    ],
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


def _clip_rating(value: float) -> int:
    return int(np.clip(round(value), 1, 4))


def _attrition_probability(row: dict) -> float:
    """Compute attrition probability from employee features (stronger coefficients)."""
    score = -4.5

    if row["JobSatisfaction"] == 1:
        score += 2.0
    elif row["JobSatisfaction"] == 2:
        score += 1.3
    if row["WorkLifeBalance"] == 1:
        score += 1.6
    elif row["WorkLifeBalance"] == 2:
        score += 1.0
    if row["OverTime"] == "Yes":
        score += 1.5
    if row["MonthlyIncome"] < 3500:
        score += 1.2
    elif row["MonthlyIncome"] < 5000:
        score += 0.5
    if row["DistanceFromHome"] > 15:
        score += 1.0
    elif row["DistanceFromHome"] > 10:
        score += 0.5
    if row["YearsAtCompany"] < 2:
        score += 1.1
    elif row["YearsAtCompany"] < 5:
        score += 0.4
    if row["YearsInCurrentRole"] >= row["YearsAtCompany"] - 1 and row["YearsAtCompany"] > 3:
        score -= 0.8
    if row["PerformanceRating"] >= 4:
        score -= 0.7
    if row["NumCompaniesWorked"] >= 4:
        score += 0.8
    if row["Age"] < 30:
        score += 0.5
    if row["JobRole"] in ("Sales Representative", "Laboratory Technician"):
        score += 0.5
    if row["BusinessTravel"] == "Travel_Frequently":
        score += 0.6
    if row["MaritalStatus"] == "Single":
        score += 0.35

    return 1 / (1 + np.exp(-score))


def generate_hr_data(n_records: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic HR dataset with correlated features and attrition outcomes."""
    rng = np.random.default_rng(seed)

    records = []
    for _ in range(n_records):
        # Latent risk drives correlated feature patterns
        risk = float(rng.normal(0, 1))

        age = int(rng.integers(22, 61))
        department = rng.choice(list(DEPARTMENTS.keys()))
        job_role = rng.choice(DEPARTMENTS[department])

        job_satisfaction = _clip_rating(3.2 - 0.9 * risk + rng.normal(0, 0.35))
        work_life_balance = _clip_rating(3.1 - 0.8 * risk + rng.normal(0, 0.35))
        overtime = "Yes" if risk > 0.6 else ("Yes" if risk > 0.1 and rng.random() < 0.35 else "No")

        total_working_years = int(rng.integers(max(0, age - 22), min(age - 18, 40) + 1))
        years_at_company = int(
            np.clip(
                round((total_working_years * 0.5) - 0.8 * risk + rng.normal(0, 1.5)),
                0,
                min(total_working_years, 25),
            )
        )
        years_in_current_role = int(rng.integers(0, min(years_at_company + 1, 15)))
        num_companies = int(
            np.clip(round(2 + 0.6 * risk + rng.normal(0, 0.8)), 1, min(total_working_years + 2, 10))
        )

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
        income_factor = 1.0 - 0.12 * risk
        monthly_income = int(max(2500, rng.normal(base_income * income_factor, base_income * 0.08)))

        distance = int(np.clip(round(10 + 4 * risk + rng.normal(0, 3)), 1, 30))
        performance = 3 if risk > 0.5 else int(rng.choice([3, 4], p=[0.55, 0.45]))
        education = rng.choice(EDUCATION_LEVELS, p=[0.05, 0.15, 0.45, 0.28, 0.07])
        marital = "Single" if risk > 0.4 and rng.random() < 0.55 else rng.choice(
            MARITAL_STATUS, p=[0.35, 0.55, 0.10]
        )
        travel = (
            "Travel_Frequently"
            if risk > 0.7
            else rng.choice(BUSINESS_TRAVEL, p=[0.15, 0.65, 0.20])
        )

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
    version_path = DATA_PATH.parent / ".data_version"
    stored_version = version_path.read_text().strip() if version_path.exists() else None

    if DATA_PATH.exists() and not force_regenerate and stored_version == DATA_VERSION:
        return pd.read_csv(DATA_PATH)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_hr_data()
    df.to_csv(DATA_PATH, index=False)
    version_path.write_text(DATA_VERSION)
    return df


if __name__ == "__main__":
    df = load_or_generate_data(force_regenerate=True)
    print(f"Generated {len(df)} records -> {DATA_PATH}")
    print(f"Attrition rate: {df['Attrition'].eq('Yes').mean():.1%}")