from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub
import pandas as pd

from src.utils import DATA_DIR, ensure_directories


def load_split(csv_path: Path, label: str) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    dataframe["label"] = label
    return dataframe


def prepare_kaggle_dataset(output_path: str | Path = DATA_DIR / "fake_news_dataset.csv") -> Path:
    ensure_directories()
    download_path = Path(kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset"))

    true_path = download_path / "True.csv"
    fake_path = download_path / "Fake.csv"
    if not true_path.exists() or not fake_path.exists():
        raise FileNotFoundError(
            f"Expected True.csv and Fake.csv inside '{download_path}', but they were not both found."
        )

    true_df = load_split(true_path, "REAL")
    fake_df = load_split(fake_path, "FAKE")
    combined = pd.concat([true_df, fake_df], ignore_index=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Downloaded dataset directory: {download_path}")
    print(f"Combined dataset saved to: {output_path}")
    print(f"Rows: {len(combined)}")
    print("Class distribution:")
    print(combined["label"].value_counts().to_string())
    print("Columns:")
    print(", ".join(combined.columns))

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Download and combine the Kaggle fake/real news dataset.")
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "fake_news_dataset.csv"),
        help="Combined CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_kaggle_dataset(output_path=args.output)


if __name__ == "__main__":
    main()
