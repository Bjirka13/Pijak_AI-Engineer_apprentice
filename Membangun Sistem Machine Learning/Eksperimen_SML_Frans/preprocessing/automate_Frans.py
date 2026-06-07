"""Automated preprocessing for MSML Kriteria 1."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "dataco_raw" / "DataCoSupplyChainDataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "preprocessing" / "dataco_preprocessing"
ARTIFACT_DIR = PROJECT_ROOT / "preprocessing" / "dataco_preprocessing_artifacts"

TARGET = "late_delivery_risk"
GROUPBY_COLUMN = "customer_id"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOOTHING_FACTOR = 10

RAW_FEATURES = [
    "days_for_shipment_(scheduled)",
    "benefit_per_order",
    "category_name",
    "customer_id",
    "customer_segment",
    "market",
    "order_date_(dateorders)",
    "order_id",
    "order_item_quantity",
    "sales",
    "order_region",
    "shipping_mode",
]

CATEGORICAL_FEATURES = [
    "category_name",
    "customer_segment",
    "market",
    "order_region",
    "shipping_mode",
]

NUMERICAL_FEATURES = [
    "days_for_shipment_(scheduled)",
    "benefit_per_order",
    "order_item_quantity",
    "sales",
    "order_month",
    "order_dayofweek",
    "is_weekend_order",
    "shipping_mode_risk",
    "avg_delay_by_region",
    "category_delay_risk",
    "customer_order_count",
    "customer_late_rate",
]


def load_dataset() -> pd.DataFrame:
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(RAW_DATA_PATH, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode dataset: {RAW_DATA_PATH}")

    df.columns = df.columns.str.lower().str.replace(" ", "_")
    selected_columns = RAW_FEATURES + [TARGET]
    missing = sorted(set(selected_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[selected_columns].copy()


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order_date_(dateorders)"] = pd.to_datetime(df["order_date_(dateorders)"])
    df["order_month"] = df["order_date_(dateorders)"].dt.month
    df["order_dayofweek"] = df["order_date_(dateorders)"].dt.dayofweek
    df["is_weekend_order"] = df["order_dayofweek"].isin([5, 6]).astype(int)
    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df[GROUPBY_COLUMN]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def add_train_only_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    global_mean = float(train_df[TARGET].mean())

    shipping_mode_risk = train_df.groupby("shipping_mode")[TARGET].mean()
    region_risk = train_df.groupby("order_region")[TARGET].mean()
    category_risk = train_df.groupby("category_name")[TARGET].mean()
    customer_order_count = train_df.groupby("customer_id")["order_id"].count()
    customer_stats = train_df.groupby("customer_id")[TARGET].agg(n_transactions="size", raw_late_rate="mean")
    customer_stats["customer_late_rate"] = (
        customer_stats["n_transactions"] * customer_stats["raw_late_rate"]
        + SMOOTHING_FACTOR * global_mean
    ) / (customer_stats["n_transactions"] + SMOOTHING_FACTOR)

    train_df["shipping_mode_risk"] = train_df["shipping_mode"].map(shipping_mode_risk)
    test_df["shipping_mode_risk"] = test_df["shipping_mode"].map(shipping_mode_risk).fillna(global_mean)
    train_df["avg_delay_by_region"] = train_df["order_region"].map(region_risk)
    test_df["avg_delay_by_region"] = test_df["order_region"].map(region_risk).fillna(global_mean)
    train_df["category_delay_risk"] = train_df["category_name"].map(category_risk)
    test_df["category_delay_risk"] = test_df["category_name"].map(category_risk).fillna(global_mean)
    train_df["customer_order_count"] = train_df["customer_id"].map(customer_order_count)
    test_df["customer_order_count"] = test_df["customer_id"].map(customer_order_count).fillna(0)
    train_df["customer_late_rate"] = train_df["customer_id"].map(customer_stats["customer_late_rate"])
    test_df["customer_late_rate"] = test_df["customer_id"].map(customer_stats["customer_late_rate"]).fillna(global_mean)

    metadata = {
        "global_mean": global_mean,
        "shipping_mode_risk": shipping_mode_risk.to_dict(),
        "region_risk": region_risk.to_dict(),
        "category_risk": category_risk.to_dict(),
    }
    return train_df, test_df, metadata


def build_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    feature_names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat":
            encoder = transformer.named_steps["encoder"]
            feature_names.extend(encoder.get_feature_names_out(columns))
        else:
            feature_names.extend(columns)
    return feature_names


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "encoders").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "metadata").mkdir(parents=True, exist_ok=True)
    for output_file in OUTPUT_DIR.glob("*.csv"):
        output_file.unlink()

    df = add_temporal_features(load_dataset())
    train_df, test_df = split_dataset(df)
    train_df, test_df, global_means = add_train_only_features(train_df, test_df)

    drop_columns = [GROUPBY_COLUMN, "order_id", "order_date_(dateorders)"]
    train_df = train_df.drop(columns=[col for col in drop_columns if col in train_df.columns])
    test_df = test_df.drop(columns=[col for col in drop_columns if col in test_df.columns])

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    pd.DataFrame(X_train_processed, columns=feature_names).to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test_processed, columns=feature_names).to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)
    joblib.dump(preprocessor, ARTIFACT_DIR / "encoders" / "preprocessor.pkl")
    joblib.dump(
        {
            "feature_names": feature_names,
            "global_means": global_means,
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        ARTIFACT_DIR / "metadata" / "preprocessing_metadata.pkl",
    )


if __name__ == "__main__":
    main()
