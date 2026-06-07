# Panduan Penggunaan Lokal dan Submission

Dokumen ini menjelaskan cara menjalankan project, membuat bukti submission, dan bagian yang perlu dikerjakan manual.

## Prasyarat Lokal

Gunakan Python 3.11. Saat pengecekan terakhir, mesin lokal hanya mendeteksi Python 3.14 sehingga instalasi `mlflow==2.19.0` tidak bisa diverifikasi sampai selesai.

Cek versi Python:

```powershell
py -0p
python --version
```

Jika Python 3.11 belum ada, install Python 3.11 lalu buat virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika menjalankan repository `Workflow-CI` secara terpisah:

```powershell
cd Workflow-CI\MLProject
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Kriteria 1: Eksperimen dan Preprocessing

Jalankan preprocessing dari folder repository Kriteria 1:

```powershell
cd Eksperimen_SML_Frans
python preprocessing\automate_Frans.py
```

Output yang harus ada:

- `dataco_preprocessing/X_train.csv`
- `dataco_preprocessing/X_test.csv`
- `dataco_preprocessing/y_train.csv`
- `dataco_preprocessing/y_test.csv`
- `preprocessing_artifacts/encoders/preprocessor.pkl`
- `preprocessing_artifacts/metadata/preprocessing_metadata.pkl`

Workflow otomatis tersedia di:

```text
Eksperimen_SML_Frans/.github/workflows/preprocessing.yml
```

Setelah repository Kriteria 1 dibuat public di GitHub, isi:

```text
SMSML_Frans/Eksperimen_SML_Frans.txt
```

## Kriteria 2: Membangun Model dengan MLflow

Jalankan dari root project:

```powershell
python automate_preprocessing.py
$env:SHAP_SAMPLE_SIZE="200"
python modelling.py
```

Atau jalankan dari folder final submission:

```powershell
cd SMSML_Frans\Membangun_model
$env:SHAP_SAMPLE_SIZE="200"
python modelling.py
```

Output yang perlu dicek:

- model tersimpan di folder `models/`
- MLflow run muncul di `mlruns/` atau DagsHub
- artifact evaluasi muncul, minimal confusion matrix, classification report, feature importance, dan SHAP summary
- metrics muncul: accuracy, precision, recall, f1, roc_auc

Jika menggunakan DagsHub, isi environment variable berikut sebelum training:

```powershell
$env:DAGSHUB_REPO="<owner>/<repo>"
$env:DAGSHUB_USERNAME="<username>"
$env:DAGSHUB_TOKEN="<token>"
```

Setelah run berhasil tersimpan online, isi:

```text
SMSML_Frans/Membangun_model/DagsHub.txt
```

Tambahkan screenshot asli sesuai:

```text
SMSML_Frans/Membangun_model/BUKTI_SCREENSHOT_WAJIB.md
```

## Kriteria 3: Workflow CI dan Docker Hub

Repository `Workflow-CI` sudah berisi workflow:

```text
Workflow-CI/.github/workflows/mlflow-ci.yml
```

Secrets GitHub yang perlu diisi di repository `Workflow-CI`:

- `MLFLOW_TRACKING_URI`, opsional jika memakai tracking server eksplisit
- `DAGSHUB_REPO`, format `owner/repo`
- `DAGSHUB_USERNAME`
- `DAGSHUB_TOKEN`
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Workflow akan:

1. install dependency
2. menjalankan `mlflow run MLProject -e main --env-manager=local`
3. upload training artifact
4. build Docker image dari model MLflow
5. push image ke Docker Hub

Setelah repository `Workflow-CI` dibuat public di GitHub, isi:

```text
SMSML_Frans/Workflow-CI.txt
```

Setelah image berhasil dipush, isi:

```text
Workflow-CI/Tautan ke Docker Hub.txt
```

## Kriteria 4: Monitoring dan Logging

Jalankan inference API dari root project:

```powershell
uvicorn inference:app --host 0.0.0.0 --port 8000
```

Cek health endpoint:

```powershell
curl.exe http://localhost:8000/health
```

Kirim request prediksi agar metric Prometheus bertambah:

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"region\":\"Southeast Asia\",\"shipping_mode\":\"Standard Class\",\"category\":\"Sporting Goods\",\"quantity\":1,\"days_for_shipment\":4,\"sales\":120.0,\"benefit_per_order\":10.0,\"customer_segment\":\"Consumer\",\"market\":\"Pacific Asia\"}"
```

Endpoint metric:

```text
http://localhost:8000/metrics
```

Jika memakai file Prometheus root:

```powershell
prometheus --config.file=prometheus.yml
```

Query yang perlu ditunjukkan tersedia di:

```text
SMSML_Frans/Monitoring dan Logging/4.bukti monitoring Prometheus/prometheus_queries.md
```

Dashboard Grafana yang perlu dibuat tersedia rencananya di:

```text
SMSML_Frans/Monitoring dan Logging/5.bukti monitoring Grafana/grafana_dashboard_plan.md
```

Alert rule tersedia di:

```text
SMSML_Frans/Monitoring dan Logging/6.bukti alerting Grafana/alert_rules.yml
```

Tambahkan screenshot asli sesuai:

```text
SMSML_Frans/Monitoring dan Logging/BUKTI_SCREENSHOT_WAJIB.md
```

## Hal yang Masih Perlu Anda Lakukan Manual

- Install Python 3.11 jika belum tersedia.
- Jalankan training MLflow penuh dengan Python 3.11.
- Buat repository public untuk `Eksperimen_SML_Frans` dan `Workflow-CI`.
- Isi GitHub Secrets untuk DagsHub dan Docker Hub.
- Jalankan GitHub Actions sampai sukses.
- Isi file tautan `.txt` yang masih berupa placeholder.
- Buat screenshot asli untuk MLflow/DagsHub, Prometheus, Grafana dashboard, Grafana alerting, dan inference service.
- ZIP folder `SMSML_Frans` setelah semua bukti dan tautan lengkap.
