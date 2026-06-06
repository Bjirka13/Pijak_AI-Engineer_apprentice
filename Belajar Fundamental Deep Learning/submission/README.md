
# Klasifikasi Gambar Fruits 360

- **Penulis:** Frans Christiopan Hutapea
- **Email:** franshutapea05@gmail.com
- **ID Dicoding:** frans_pemula

Notebook proyek untuk mengklasifikasikan tiga kelas buah (Cherry, Peach, Plum) menggunakan subset dataset Fruits 360.

## Dataset

Classes:
- Cherry
- Peach
- Plum

Dataset:
Fruits 360

Total Images:
10061

Model:
CNN Sequential

Accuracy:
Train = 100%
Test = 100%

Export:
- SavedModel
- TFLite
- TensorFlowJS

## Ringkasan

Notebook ini melatih sebuah convolutional neural network (CNN) untuk mengklasifikasikan tiga kelas buah (Cherry, Peach, Plum). Isi notebook mencakup persiapan data, pra-pemrosesan, pelatihan model, evaluasi, visualisasi, serta ekspor model ke format SavedModel, TFLite, dan TFJS.

## Berkas yang Dihasilkan

- `saved_model/` — ekspor TensorFlow SavedModel
- `tflite/model.tflite` — model TFLite
- `tfjs_model/` — model untuk TensorFlow.js
- `label.txt` — daftar label kelas (satu per baris)

## Persyaratan

Pasang dependensi utama (disarankan menggunakan virtual environment):

```
pip install -r requirements.txt
```

Jika `requirements.txt` tidak tersedia, paket utama yang diperlukan adalah:

```
tensorflow
numpy
pandas
matplotlib
scikit-learn
seaborn
tensorflowjs  # opsional, untuk konversi TFJS
```

## Cara Menjalankan

1. Buka `notebook.ipynb` dan jalankan sel (cell) secara berurutan.
2. Notebook mengunduh dataset Fruits 360 (contoh menggunakan `kagglehub.dataset_download("moltean/fruits")`). Sesuaikan variabel `dataset_path` dan `filtered_test` jika dataset disimpan di lokasi lain.
3. Pengaturan pelatihan seperti ukuran gambar, batch size, dan epoch terdapat di notebook — sesuaikan bila diperlukan.

Langkah ekspor model yang ada di notebook:

- Ekspor SavedModel: `model.export("saved_model")`
- Konversi TFLite: menghasilkan `model.tflite`
- Konversi TFJS: notebook mengekspor SavedModel yang siap TFJS lalu menjalankan `tensorflowjs_converter` untuk membuat folder `tfjs_model/`

## Catatan

- Notebook menggunakan augmentasi data dan callback (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`).
- Arsitektur model merupakan CNN sederhana dengan 4 blok konvolusi dan lapisan dense akhir dengan softmax untuk 3 kelas.
- Jika menjalankan secara lokal di Windows, ubah jalur file (`/content/...`) pada notebook menjadi jalur Windows yang sesuai.

## Kontak

Jika ada pertanyaan, hubungi penulis di franshutapea05@gmail.com
