from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from src.data_loading import load_dataset, split_dataset
from src.dl_models import build_lstm_model, get_callbacks, plot_training_history
from src.evaluate import evaluate_dl_model
from src.preprocessing import TextPreprocessor, fit_tokenizer, save_tokenizer, tokenize_and_pad
from src.utils import FIGURES_DIR, METRICS_DIR, MODELS_DIR, ensure_directories, timer


def parse_args():
    parser = argparse.ArgumentParser(description="Train deep learning fake news detector.")
    parser.add_argument("--dataset", default="data/fake_news_dataset.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--language", default="English")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument(
        "--basic-preprocessing",
        action="store_true",
        help="Disable lemmatization and keep only basic cleaning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    dataframe = load_dataset(
        dataset_path=args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        language=args.language,
    )
    splits = split_dataset(dataframe)
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]

    preprocessor = TextPreprocessor(language=args.language.lower(), advanced=not args.basic_preprocessing)
    x_train_processed = preprocessor.transform_series(x_train)
    x_val_processed = preprocessor.transform_series(x_val)
    x_test_processed = preprocessor.transform_series(x_test)

    tokenizer = fit_tokenizer(x_train_processed, num_words=args.vocab_size)
    x_train_seq = tokenize_and_pad(tokenizer, x_train_processed, max_length=args.max_length)
    x_val_seq = tokenize_and_pad(tokenizer, x_val_processed, max_length=args.max_length)

    model_path = MODELS_DIR / "lstm_model.keras"
    tokenizer_path = MODELS_DIR / "lstm_tokenizer.joblib"
    preprocessor_path = MODELS_DIR / "lstm_preprocessor.joblib"
    metadata_path = MODELS_DIR / "lstm_metadata.joblib"

    model = build_lstm_model(
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
    )

    with timer() as elapsed:
        history = model.fit(
            x_train_seq,
            np.array(y_train),
            validation_data=(x_val_seq, np.array(y_val)),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=get_callbacks(model_path),
            verbose=1,
        )

    plot_training_history(history, FIGURES_DIR / "lstm_training_curves.png")
    save_tokenizer(tokenizer, tokenizer_path)
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "model_type": "dl",
        "model_name": "lstm",
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "preprocessor_path": str(preprocessor_path),
        "label_mapping": {"REAL": 0, "FAKE": 1},
        "max_length": args.max_length,
        "threshold": 0.5,
    }
    joblib.dump(metadata, metadata_path)

    metrics = evaluate_dl_model(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        x_test=x_test_processed,
        y_test=y_test,
        max_length=args.max_length,
        training_time=elapsed[0],
        model_name="lstm",
    )
    metrics["artifact_path"] = str(model_path)
    metrics["metadata_path"] = str(metadata_path)
    metrics["epochs_trained"] = len(history.history["loss"])

    results = [metrics]
    joblib.dump(results, METRICS_DIR / "dl_results.joblib")
    pd.DataFrame(results).to_csv(METRICS_DIR / "dl_results.csv", index=False)

    print("\nDL training finished.")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nArtifacts saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
