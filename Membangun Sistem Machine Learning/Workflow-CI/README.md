# Workflow-CI

Repository ini disiapkan untuk Kriteria 3 MSML.

## Cara Menjalankan Lokal

Gunakan Python 3.11.

```bash
cd MLProject
python -m pip install --upgrade pip
pip install -r requirements.txt
mlflow run . -e main --env-manager=local
```

Secrets GitHub yang harus tersedia:

- `MLFLOW_TRACKING_URI` jika memakai tracking server eksplisit.
- `DAGSHUB_REPO`, format `owner/repo`.
- `DAGSHUB_USERNAME`.
- `DAGSHUB_TOKEN`.
- `DOCKERHUB_USERNAME`.
- `DOCKERHUB_TOKEN`.

Workflow menjalankan MLflow Project, menyimpan artifact melalui GitHub Actions,
membangun Docker image memakai `mlflow models build-docker`, lalu push ke Docker Hub.

## Output yang Diharapkan

- `MLProject/run_id.txt`
- `MLProject/models/random_forest_model.pkl`
- `MLProject/artifacts/figures/confusion_matrix.png`
- `MLProject/artifacts/figures/feature_importance.png`
- `MLProject/artifacts/figures/shap_summary.png`
- `MLProject/artifacts/reports/classification_report.json`
- Docker image: `<DOCKERHUB_USERNAME>/shipment-delay-risk`

Setelah workflow sukses, isi `Tautan ke Docker Hub.txt` dengan URL image Docker Hub.
