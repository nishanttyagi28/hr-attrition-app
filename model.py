"""Train and evaluate a Random Forest attrition classifier."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_FEATURES = [
    "Department",
    "JobRole",
    "OverTime",
    "Education",
    "MaritalStatus",
    "BusinessTravel",
]

NUMERIC_FEATURES = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "DistanceFromHome",
    "JobSatisfaction",
    "WorkLifeBalance",
    "NumCompaniesWorked",
    "TotalWorkingYears",
    "PerformanceRating",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Attrition"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERIC_FEATURES + cat_names


def train_model(df: pd.DataFrame) -> dict[str, Any]:
    """Train Random Forest and return model artifacts and metrics."""
    X = df[FEATURE_COLUMNS]
    y = (df[TARGET_COLUMN] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    importances = pipeline.named_steps["classifier"].feature_importances_
    feature_names = get_feature_names(pipeline)
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "feature_importance": importance_df,
        "feature_columns": FEATURE_COLUMNS,
    }


def predict_employee(pipeline: Pipeline, employee: dict) -> tuple[str, float]:
    """Predict attrition label and probability for a single employee."""
    input_df = pd.DataFrame([employee])
    prob = float(pipeline.predict_proba(input_df)[0, 1])
    label = "Yes" if prob >= 0.5 else "No"
    return label, prob