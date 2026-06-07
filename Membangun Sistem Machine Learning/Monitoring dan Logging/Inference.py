"""MSML Kriteria 4 inference service with Prometheus metrics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "Membangun_model" / "models" / "random_forest_model.pkl"
SCHEMA_PATH = BASE_DIR.parent / "Membangun_model" / "namadataset_preprocessing" / "X_train.csv"

prediction_requests_total = Counter("prediction_requests_total", "Total prediction requests.")
prediction_errors_total = Counter("prediction_errors_total", "Total failed prediction requests.")
prediction_latency_seconds = Histogram("prediction_latency_seconds", "Prediction latency in seconds.")
model_confidence_avg = Gauge("model_confidence_avg", "Latest model confidence.")
high_risk_predictions_total = Counter("high_risk_predictions_total", "Total high risk predictions.")
medium_risk_predictions_total = Counter("medium_risk_predictions_total", "Total medium risk predictions.")
low_risk_predictions_total = Counter("low_risk_predictions_total", "Total low risk predictions.")
cpu_usage_percent = Gauge("cpu_usage_percent", "CPU usage percent.")
memory_usage_percent = Gauge("memory_usage_percent", "Memory usage percent.")
disk_usage_percent = Gauge("disk_usage_percent", "Disk usage percent.")
model_predictions_total = Counter("model_predictions_total", "Total successful model predictions.")


class PredictionRequest(BaseModel):
    features: Dict[str, Any] = {}


app = FastAPI(title="Shipment Delay Risk Inference", version="1.0.0")
app.mount("/metrics", make_asgi_app())

_model = None
_feature_columns: list[str] | None = None


def update_resource_metrics() -> None:
    cpu_usage_percent.set(psutil.cpu_percent(interval=None))
    memory_usage_percent.set(psutil.virtual_memory().percent)
    disk_usage_percent.set(psutil.disk_usage("/").percent)


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def load_feature_columns() -> list[str]:
    global _feature_columns
    if _feature_columns is None:
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Feature schema not found: {SCHEMA_PATH}")
        _feature_columns = list(pd.read_csv(SCHEMA_PATH, nrows=1).columns)
    return _feature_columns


def build_feature_frame(features: Dict[str, Any]) -> pd.DataFrame:
    columns = load_feature_columns()
    row = {column: 0.0 for column in columns}
    for key, value in features.items():
        if key in row:
            row[key] = value
    return pd.DataFrame([row], columns=columns)


@app.get("/health")
def health() -> Dict[str, str]:
    update_resource_metrics()
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> Dict[str, Any]:
    prediction_requests_total.inc()
    start = time.perf_counter()
    try:
        model = load_model()
        X = build_feature_frame(request.features)
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X)[0, 1])
        else:
            probability = float(model.predict(X)[0])
        predicted_class = int(probability >= 0.5)
        risk_level = "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.4 else "LOW"

        model_predictions_total.inc()
        model_confidence_avg.set(max(probability, 1.0 - probability))
        if risk_level == "HIGH":
            high_risk_predictions_total.inc()
        elif risk_level == "MEDIUM":
            medium_risk_predictions_total.inc()
        else:
            low_risk_predictions_total.inc()

        return {
            "predicted_class": predicted_class,
            "risk_probability": probability,
            "risk_level": risk_level,
            "risk_score": int(round(probability * 100)),
        }
    except Exception as exc:
        prediction_errors_total.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        prediction_latency_seconds.observe(time.perf_counter() - start)
        update_resource_metrics()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
