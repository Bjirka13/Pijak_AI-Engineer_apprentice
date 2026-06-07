# SMSML_Frans

Folder final submission MSML.

## Isi Folder

- `Eksperimen_SML_Frans.txt`: tautan repository public untuk Kriteria 1.
- `Workflow-CI.txt`: tautan repository public untuk Kriteria 3.
- `Membangun_model/`: script modelling, data preprocessing, dependency, tautan DagsHub, dan bukti screenshot Kriteria 2.
- `Monitoring dan Logging/`: file inference, Prometheus, exporter, alert rules, dan bukti screenshot Kriteria 4.
- `PANDUAN_PENGGUNAAN_LOKAL.md`: panduan menjalankan project di lokal dan menyiapkan bukti submission.

## Checklist Sebelum ZIP

1. Isi `Eksperimen_SML_Frans.txt` dengan URL repository public Kriteria 1.
2. Isi `Workflow-CI.txt` dengan URL repository public Kriteria 3.
3. Isi `Membangun_model/DagsHub.txt` dengan URL DagsHub setelah run MLflow tersimpan online.
4. Isi `../Workflow-CI/Tautan ke Docker Hub.txt` dengan URL image Docker Hub setelah workflow berhasil push image.
5. Tambahkan screenshot asli untuk Kriteria 2 ke folder `Membangun_model`.
6. Tambahkan screenshot asli untuk Kriteria 4 ke folder `Monitoring dan Logging`.
7. Pastikan tidak membuat ZIP di dalam ZIP.

## Catatan Verifikasi

Preprocessing sudah dapat dijalankan di lokal. Training MLflow penuh perlu Python 3.11 karena dependency `mlflow==2.19.0` tidak berhasil dipasang pada Python 3.14 yang tersedia di mesin lokal saat dokumentasi ini dibuat.
