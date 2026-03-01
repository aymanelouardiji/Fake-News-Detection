from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.preprocessing import load_tokenizer, tokenize_and_pad
from src.utils import FIGURES_DIR, METRICS_DIR, ensure_directories, save_json


def compute_metrics(y_true, y_pred, training_time: float) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "training_time_seconds": round(training_time, 4),
    }


def save_confusion_matrix(y_true, y_pred, labels: list[str], output_path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_ml_model(model, x_test, y_test, training_time: float, model_name: str) -> dict[str, Any]:
    predictions = model.predict(x_test)
    metrics = compute_metrics(y_test, predictions, training_time)
    metrics["model_name"] = model_name
    save_confusion_matrix(
        y_test,
        predictions,
        labels=["REAL", "FAKE"],
        output_path=FIGURES_DIR / f"{model_name}_confusion_matrix.png",
        title=f"{model_name} Confusion Matrix",
    )
    return metrics


def evaluate_dl_model(
    model_path: Path,
    tokenizer_path: Path,
    x_test,
    y_test,
    max_length: int,
    training_time: float,
    model_name: str,
) -> dict[str, Any]:
    from tensorflow.keras.models import load_model

    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model(model_path)
    x_test_seq = tokenize_and_pad(tokenizer, x_test, max_length=max_length)
    probabilities = model.predict(x_test_seq, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    metrics = compute_metrics(y_test, predictions, training_time)
    metrics["model_name"] = model_name
    save_confusion_matrix(
        y_test,
        predictions,
        labels=["REAL", "FAKE"],
        output_path=FIGURES_DIR / f"{model_name}_confusion_matrix.png",
        title=f"{model_name} Confusion Matrix",
    )
    return metrics


def create_comparison_table(records: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    dataframe = pd.DataFrame(records).sort_values(by="f1", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return dataframe


def update_best_model_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    ensure_directories()
    best_record = max(records, key=lambda item: item["f1"])
    metadata_path = Path(best_record["metadata_path"])
    metadata = joblib.load(metadata_path)
    metadata["selected_metric"] = "f1"
    metadata["selected_score"] = best_record["f1"]
    joblib.dump(metadata, metadata_path.parent / "best_model.joblib")
    save_json(best_record, METRICS_DIR / "best_model_summary.json")
    return best_record


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate model results and select the best model.")
    parser.add_argument(
        "--ml-results",
        type=str,
        default=str(METRICS_DIR / "ml_results.joblib"),
        help="Path to saved ML results artifact.",
    )
    parser.add_argument(
        "--dl-results",
        type=str,
        default=str(METRICS_DIR / "dl_results.joblib"),
        help="Path to saved DL results artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for path_str in (args.ml_results, args.dl_results):
        path = Path(path_str)
        if path.exists():
            payload = joblib.load(path)
            if isinstance(payload, list):
                records.extend(payload)

    if not records:
        raise FileNotFoundError("No training results found. Run the training scripts first.")

    comparison = create_comparison_table(records, METRICS_DIR / "model_comparison.csv")
    best_record = update_best_model_metadata(records)
    print("Model comparison:")
    print(comparison.to_string(index=False))
    print("\nBest model:")
    print(best_record)


if __name__ == "__main__":
    main()
