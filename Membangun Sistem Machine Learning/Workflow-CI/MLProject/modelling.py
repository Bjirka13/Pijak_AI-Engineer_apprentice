"""MSML Kriteria 2 modelling script with manual MLflow logging."""

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


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "namadataset_preprocessing"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = BASE_DIR / "models"
RUN_ID_PATH = BASE_DIR / "run_id.txt"

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "random_state": 42,
    "n_jobs": -1,
}


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
    else:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "shipment-delay-risk"))


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
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


def save_artifacts(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> list[Path]:
    figure_dir = ARTIFACT_DIR / "figures"
    report_dir = ARTIFACT_DIR / "reports"
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    paths: list[Path] = []

    cm_path = figure_dir / "confusion_matrix.png"
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred),
        display_labels=["On Time", "Late"],
    ).plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=160)
    plt.close()
    paths.append(cm_path)

    report_path = report_dir / "classification_report.json"
    report_path.write_text(
        json.dumps(classification_report(y_test, y_pred, output_dict=True, zero_division=0), indent=2),
        encoding="utf-8",
    )
    paths.append(report_path)

    importance = (
        pd.DataFrame({"feature": X_test.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_csv = report_dir / "feature_importance.csv"
    fi_png = figure_dir / "feature_importance.png"
    importance.to_csv(fi_csv, index=False)
    top = importance.head(20).sort_values("importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance"])
    plt.tight_layout()
    plt.savefig(fi_png, dpi=160)
    plt.close()
    paths.extend([fi_csv, fi_png])

    shap_path = figure_dir / "shap_summary.png"
    sample = X_test.sample(n=min(int(os.getenv("SHAP_SAMPLE_SIZE", "500")), len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    class_values = shap_values[1] if isinstance(shap_values, list) else shap_values
    if hasattr(class_values, "ndim") and class_values.ndim == 3:
        class_values = class_values[:, :, 1]
    shap.summary_plot(class_values, sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(shap_path, dpi=160, bbox_inches="tight")
    plt.close()
    paths.append(shap_path)

    return paths


def main() -> None:
    configure_mlflow()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "manual-random-forest")) as run:
        mlflow.log_params(RF_PARAMS)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("validation_strategy", "GroupShuffleSplit by customer_id")
        mlflow.log_param("feature_count", X_train.shape[1])
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train, y_train)

        mlflow.log_metrics({f"train_{k}": v for k, v in compute_metrics(model, X_train, y_train).items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in compute_metrics(model, X_test, y_test).items()})

        model_path = MODEL_DIR / "random_forest_model.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_test.head(3))
        mlflow.log_artifact(str(model_path), artifact_path="model_pickle")

        for artifact_path in save_artifacts(model, X_test, y_test):
            mlflow.log_artifact(str(artifact_path), artifact_path="evaluation")

        RUN_ID_PATH.write_text(run.info.run_id, encoding="utf-8")


if __name__ == "__main__":
    main()
