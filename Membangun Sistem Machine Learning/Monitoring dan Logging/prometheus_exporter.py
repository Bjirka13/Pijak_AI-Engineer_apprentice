"""Prometheus metrics for shipment delay inference monitoring."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import psutil
from prometheus_client import Counter, Gauge, Histogram, start_http_server


prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total prediction requests received.",
)
prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total prediction requests that failed.",
)
prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
model_confidence_avg = Gauge(
    "model_confidence_avg",
    "Latest model confidence score calculated as max(probability, 1 - probability).",
)
high_risk_predictions_total = Counter(
    "high_risk_predictions_total",
    "Total predictions classified as high risk.",
)
low_risk_predictions_total = Counter(
    "low_risk_predictions_total",
    "Total predictions classified as low risk.",
)
cpu_usage_percent = Gauge(
    "cpu_usage_percent",
    "Current CPU usage percentage.",
)
memory_usage_percent = Gauge(
    "memory_usage_percent",
    "Current memory usage percentage.",
)
disk_usage_percent = Gauge(
    "disk_usage_percent",
    "Current disk usage percentage.",
)
model_predictions_total = Counter(
    "model_predictions_total",
    "Total model predictions completed successfully.",
)
medium_risk_predictions_total = Counter(
    "medium_risk_predictions_total",
    "Total predictions classified as medium risk.",
)


def update_resource_metrics() -> None:
    cpu_usage_percent.set(psutil.cpu_percent(interval=None))
    memory_usage_percent.set(psutil.virtual_memory().percent)
    disk_usage_percent.set(psutil.disk_usage("/").percent)


def record_prediction(probability: float, risk_level: str) -> None:
    model_predictions_total.inc()
    model_confidence_avg.set(max(float(probability), 1.0 - float(probability)))
    normalized_level = risk_level.upper()
    if normalized_level == "HIGH":
        high_risk_predictions_total.inc()
    elif normalized_level == "LOW":
        low_risk_predictions_total.inc()
    else:
        medium_risk_predictions_total.inc()


@contextmanager
def track_prediction_latency() -> Iterator[None]:
    prediction_requests_total.inc()
    start = time.perf_counter()
    try:
        yield
    except Exception:
        prediction_errors_total.inc()
        raise
    finally:
        prediction_latency_seconds.observe(time.perf_counter() - start)
        update_resource_metrics()


def start_metrics_server(port: int = 8001) -> None:
    start_http_server(port)
    update_resource_metrics()


if __name__ == "__main__":
    start_metrics_server()
    while True:
        update_resource_metrics()
        time.sleep(5)
