# NLP Assignment

## Deskripsi
Proyek ini adalah tugas NLP untuk analisis sentimen ulasan aplikasi. Data berasal dari file `Data/dataset_sentiment.csv`, lalu diproses dan dievaluasi di notebook `model_LR.ipynb`.

## Struktur Proyek

- `model_LR.ipynb` - Notebook utama yang mencakup impor library, pembersihan teks, pelabelan, ekstraksi fitur, pelatihan model, inferensi, dan evaluasi.
- `requirements.txt` - Daftar dependensi Python yang dibutuhkan untuk menjalankan proyek.
- `Data/dataset_sentiment.csv` - Dataset sentimen berisi ulasan aplikasi dengan kolom `app`, `review`, dan `score`.
- `Tools/scraper.ipynb` - Notebook tambahan untuk scraping ulasan aplikasi dari Google Play.

## Alur Notebook `model_LR.ipynb`

1. Import library dan unduh stopwords NLTK.
2. Load dataset dan lakukan pelabelan sentimen.
3. Pra-pemrosesan teks:
   - menghapus karakter unik
   - menghapus punctuation
   - menghapus stopwords Bahasa Indonesia
   - melakukan stemming dengan `Sastrawi`
4. Membagi data menjadi train/test.
5. Membangun beberapa skema pemodelan:
   - Skema 1: `TF-IDF + Logistic Regression`
   - Skema 2: `TF-IDF + Linear SVM`
   - Skema 3: `Tokenizer + LSTM`
6. Evaluasi model dengan metrik akurasi, classification report, dan confusion matrix.
7. Contoh inferensi sentimen pada teks baru.

## Dataset

Dataset memiliki kolom-kolom berikut:

- `app` - Nama aplikasi.
- `review` - Teks ulasan pengguna.
- `score` - Nilai rating/score.

Notebook juga membuat kolom `label` untuk klasifikasi sentimen negatif vs positif berdasarkan skor.

## Instalasi

1. Pastikan Python terpasang di lingkungan Anda.
2. Jalankan perintah berikut di direktori proyek:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

1. Buka `model_LR.ipynb` dengan Jupyter Notebook / JupyterLab / VS Code.
2. Jalankan sel-sel notebook secara berurutan.
3. Perhatikan bahwa notebook menggunakan `nltk.download('stopwords')` dan menginstal `Sastrawi` di dalam notebook.
4. Untuk eksperimen tambahan, buka `Tools/scraper.ipynb` untuk scraping data ulasan.

## Dependensi Utama

- `pandas`
- `numpy`
- `nltk`
- `scikit_learn`
- `tensorflow`
- `matplotlib`
- `seaborn`
- `Sastrawi`
- `tqdm`
- `google_play_scraper`

## Catatan

- Model utama di notebook menggunakan `LogisticRegression` dengan `TF-IDF` sebagai fitur.
- Notebook juga mengevaluasi `LinearSVC` dan model `LSTM` sebagai alternatif.
- Bagian inferensi menunjukkan prediksi sentimen (`positif` / `negatif`) untuk contoh ulasan baru.

---

Terima kasih telah menggunakan proyek ini untuk belajar dasar-dasar NLP dan analisis sentimen.