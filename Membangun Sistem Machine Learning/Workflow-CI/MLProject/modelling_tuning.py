"""MSML Kriteria 2 tuning script with manual MLflow logging."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd
from modelling import configure_mlflow, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts" / "tuning"

PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [10, 12],
    "min_samples_split": [10],
    "min_samples_leaf": [4],
}


def main() -> None:
    configure_mlflow()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "manual-random-forest-tuning")):
        search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid=PARAM_GRID,
            scoring=make_scorer(f1_score, zero_division=0),
            cv=3,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        result_path = ARTIFACT_DIR / "tuning_results.csv"
        pd.DataFrame(search.cv_results_).to_csv(result_path, index=False)
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_f1", float(search.best_score_))
        mlflow.log_metric("test_f1", float(f1_score(y_test, search.best_estimator_.predict(X_test), zero_division=0)))
        mlflow.log_artifact(str(result_path), artifact_path="tuning")


if __name__ == "__main__":
    main()
