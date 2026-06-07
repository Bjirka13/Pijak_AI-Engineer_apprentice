"""Manual MLflow training script for shipment delay risk prediction.

This file intentionally does not use mlflow.autolog. It logs parameters,
metrics, model artifacts, and visual artifacts manually to satisfy the MSML
advanced rubric.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "dataco_preprocessing"
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "mlflow"
MODEL_ARTIFACT_DIR = ARTIFACT_DIR / "model"
FIGURE_DIR = ARTIFACT_DIR / "figures"
REPORT_DIR = ARTIFACT_DIR / "reports"
RANDOM_STATE = 42
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def ensure_directories_exist(paths: list[Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def configure_mlflow() -> None:
    dagshub_repo = os.getenv("DAGSHUB_REPO")
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    if dagshub_repo and dagshub_username and dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        try:
            import dagshub

            owner, repo = dagshub_repo.split("/", maxsplit=1)
            dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
        except Exception:
            mlflow.set_tracking_uri(f"https://dagshub.com/{dagshub_repo}.mlflow")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif not dagshub_repo:
        # Use SQLite for local MLflow tracking by default.
        # The legacy file store backend is now in maintenance mode for MLflow 3+.
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "shipment-delay-risk")
    mlflow.set_experiment(experiment_name)


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    required = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    missing = [name for name in required if not (PROCESSED_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed data files: {missing}. Expected them in {PROCESSED_DIR}."
        )

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def compute_metrics(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> Path:
    path = FIGURE_DIR / "confusion_matrix.png"
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["On Time", "Late"])
    display.plot(cmap="Blues", values_format="d")
    plt.title("Shipment Delay Risk - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_classification_report(y_true: pd.Series, y_pred: np.ndarray) -> Path:
    path = REPORT_DIR / "classification_report.json"
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def save_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> Tuple[Path, Path]:
    csv_path = REPORT_DIR / "feature_importance.csv"
    png_path = FIGURE_DIR / "feature_importance.png"
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(csv_path, index=False)

    top = importance.head(20).sort_values("importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top 20 Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()
    return csv_path, png_path


def save_shap_summary(model: RandomForestClassifier, X_test: pd.DataFrame) -> Path:
    path = FIGURE_DIR / "shap_summary.png"
    sample_size = int(os.getenv("SHAP_SAMPLE_SIZE", "500"))
    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    class_values = shap_values[1] if isinstance(shap_values, list) else shap_values
    if hasattr(class_values, "ndim") and class_values.ndim == 3:
        class_values = class_values[:, :, 1]
    plt.figure()
    shap.summary_plot(class_values, sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def train_and_log() -> RandomForestClassifier:
    configure_mlflow()
    ensure_directories_exist([ARTIFACT_DIR, MODEL_ARTIFACT_DIR, FIGURE_DIR, REPORT_DIR, Path(MODEL_PATH).parent])

    X_train, X_test, y_train, y_test = load_processed_data()
    with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "manual-random-forest")):
        mlflow.log_params(RF_PARAMS)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("validation_strategy", "GroupShuffleSplit by customer_id")
        mlflow.log_param("feature_count", X_train.shape[1])
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train, y_train)

        train_metrics = compute_metrics(model, X_train, y_train)
        test_metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics({f"train_{key}": value for key, value in train_metrics.items()})
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

        y_test_pred = model.predict(X_test)
        artifact_paths = [
            save_confusion_matrix(y_test, y_test_pred),
            save_classification_report(y_test, y_test_pred),
            *save_feature_importance(model, list(X_train.columns)),
            save_shap_summary(model, X_test),
        ]
        for path in artifact_paths:
            mlflow.log_artifact(str(path), artifact_path="evaluation")

        joblib.dump(model, MODEL_PATH)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.head(3),
            registered_model_name=os.getenv("MLFLOW_REGISTERED_MODEL_NAME") or None,
        )
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_pickle")

    return model


if __name__ == "__main__":
    train_and_log()
