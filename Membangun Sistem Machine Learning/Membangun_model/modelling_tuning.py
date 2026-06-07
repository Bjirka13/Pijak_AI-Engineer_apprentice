"""Manual MLflow hyperparameter tuning script.

This is provided for Dicoding's advanced modeling requirement. It keeps the
search space intentionally small so CI remains practical.
"""

from __future__ import annotations

import os
from typing import Dict

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

from modelling import configure_mlflow, load_processed_data


RANDOM_STATE = 42


def ensure_directories_exist(paths: list[str]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


PARAM_GRID: Dict[str, list] = {
    "n_estimators": [100, 200],
    "max_depth": [10, 12],
    "min_samples_split": [10],
    "min_samples_leaf": [4],
}


def run_tuning() -> GridSearchCV:
    configure_mlflow()
    ensure_directories_exist(["artifacts/mlflow/reports"])
    X_train, X_test, y_train, y_test = load_processed_data()
    with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "manual-random-forest-tuning")):
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        search = GridSearchCV(
            estimator=base_model,
            param_grid=PARAM_GRID,
            scoring=make_scorer(f1_score, zero_division=0),
            cv=3,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)

        results = pd.DataFrame(search.cv_results_)
        results_path = "artifacts/mlflow/reports/tuning_results.csv"
        results.to_csv(results_path, index=False)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_f1", float(search.best_score_))
        mlflow.log_metric("test_f1", float(f1_score(y_test, search.best_estimator_.predict(X_test), zero_division=0)))
        mlflow.log_artifact(results_path, artifact_path="tuning")
    return search


if __name__ == "__main__":
    run_tuning()
