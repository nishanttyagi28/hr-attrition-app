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
from sklearn.model_selection import GridSearchCV, train_test_split
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

BASELINE_METRICS = {
    "accuracy": 0.727,
    "precision": 0.418,
    "recall": 0.621,
    "f1": 0.500,
    "threshold": 0.50,
    "note": "Default RF with class_weight='balanced', threshold=0.50",
}


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


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick threshold that balances precision and recall (favoring precision)."""
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.30, 0.76, 0.02):
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if precision < 0.50:
            continue

        score = 0.45 * precision + 0.35 * f1 + 0.20 * recall
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def train_model(df: pd.DataFrame) -> dict[str, Any]:
    """Train tuned Random Forest with GridSearch and threshold optimization."""
    X = df[FEATURE_COLUMNS]
    y = (df[TARGET_COLUMN] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    pipeline = build_pipeline()
    param_grid = {
        "classifier__n_estimators": [150, 200, 300],
        "classifier__max_depth": [8, 10, 12],
        "classifier__min_samples_leaf": [3, 5, 8],
        "classifier__class_weight": [None, "balanced", {0: 1, 1: 1.5}],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )
    grid_search.fit(X_fit, y_fit)

    best_pipeline = grid_search.best_estimator_
    val_probs = best_pipeline.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val.to_numpy(), val_probs)

    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (test_probs >= threshold).astype(int)

    importances = best_pipeline.named_steps["classifier"].feature_importances_
    feature_names = get_feature_names(best_pipeline)
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "pipeline": best_pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": test_probs,
        "threshold": threshold,
        "best_params": grid_search.best_params_,
        "metrics": {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "feature_importance": importance_df,
        "feature_columns": FEATURE_COLUMNS,
        "baseline_metrics": BASELINE_METRICS,
    }


def predict_employee(
    pipeline: Pipeline, employee: dict, threshold: float = 0.5
) -> tuple[str, float]:
    """Predict attrition label and probability for a single employee."""
    input_df = pd.DataFrame([employee])
    prob = float(pipeline.predict_proba(input_df)[0, 1])
    label = "Yes" if prob >= threshold else "No"
    return label, prob