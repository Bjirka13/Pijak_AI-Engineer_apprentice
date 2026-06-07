# Eksperimen_SML_Frans

Repository Kriteria 1 MSML untuk eksperimen dan preprocessing dataset DataCo Supply Chain.

Isi utama:

- `dataco_raw/DataCoSupplyChainDataset.csv`: dataset mentah.
- `Eksperimen_Frans.ipynb`: notebook eksperimen.
- `preprocessing/automate_Frans.py`: automasi preprocessing.
- `dataco_preprocessing/`: dataset hasil preprocessing.
- `preprocessing_artifacts/`: preprocessor dan metadata preprocessing.
- `.github/workflows/preprocessing.yml`: workflow preprocessing otomatis.

Cara menjalankan:

```bash
pip install pandas==2.2.3 numpy==1.26.4 scikit-learn==1.6.1 joblib==1.4.2
python preprocessing/automate_Frans.py
```

Output yang dihasilkan:

- `dataco_preprocessing/X_train.csv`
- `dataco_preprocessing/X_test.csv`
- `dataco_preprocessing/y_train.csv`
- `dataco_preprocessing/y_test.csv`
- `preprocessing_artifacts/encoders/preprocessor.pkl`
- `preprocessing_artifacts/metadata/preprocessing_metadata.pkl`

Workflow otomatis tersedia di `.github/workflows/preprocessing.yml`.
