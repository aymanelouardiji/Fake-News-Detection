from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ensure_directories


LABEL_MAPPING = {"FAKE": 1, "REAL": 0}
INVERSE_LABEL_MAPPING = {value: key for key, value in LABEL_MAPPING.items()}


def load_dataset(
    dataset_path: str | Path,
    text_column: str,
    label_column: str,
    language: str = "English",
) -> pd.DataFrame:
    ensure_directories()
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Place the CSV there before training."
        )

    dataframe = pd.read_csv(dataset_path)
    if text_column not in dataframe.columns:
        raise KeyError(f"Text column '{text_column}' not found in dataset.")
    if label_column not in dataframe.columns:
        raise KeyError(f"Label column '{label_column}' not found in dataset.")

    dataframe = dataframe[[text_column, label_column]].dropna().copy()
    dataframe[text_column] = dataframe[text_column].astype(str).str.strip()
    dataframe[label_column] = dataframe[label_column].astype(str).str.strip().str.upper()
    dataframe = dataframe[dataframe[text_column].ne("")]
    dataframe = dataframe[dataframe[label_column].isin(LABEL_MAPPING)]
    dataframe["label_id"] = dataframe[label_column].map(LABEL_MAPPING).astype(int)
    dataframe = dataframe.rename(columns={text_column: "text", label_column: "label"})

    print("=" * 80)
    print("Dataset summary")
    print(f"Path: {dataset_path}")
    print(f"Language: {language}")
    print(f"Dataset size: {len(dataframe)}")
    print("Class distribution:")
    print(dataframe["label"].value_counts(normalize=False).rename_axis("label"))
    print("\nSample rows:")
    print(dataframe.head(3).to_string(index=False))
    print("=" * 80)
    return dataframe.reset_index(drop=True)


def split_dataset(
    dataframe: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, Any]:
    texts = dataframe["text"]
    labels = dataframe["label_id"]

    x_temp, x_test, y_temp, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    validation_ratio = validation_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=validation_ratio,
        stratify=y_temp,
        random_state=random_state,
    )

    return {
        "train": (x_train.reset_index(drop=True), y_train.reset_index(drop=True)),
        "val": (x_val.reset_index(drop=True), y_val.reset_index(drop=True)),
        "test": (x_test.reset_index(drop=True), y_test.reset_index(drop=True)),
        "train_full": (
            pd.concat([x_train, x_val], ignore_index=True),
            pd.concat([y_train, y_val], ignore_index=True),
        ),
    }
