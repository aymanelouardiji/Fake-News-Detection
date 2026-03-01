from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src.data_loading import load_dataset, split_dataset
from src.evaluate import evaluate_ml_model
from src.ml_models import build_search, get_ml_model_specs
from src.preprocessing import TextPreprocessor, build_tfidf_vectorizer
from src.utils import METRICS_DIR, MODELS_DIR, ensure_directories, timer


def parse_args():
    parser = argparse.ArgumentParser(description="Train classical machine learning fake news detectors.")
    parser.add_argument("--dataset", default="data/fake_news_dataset.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--language", default="English")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
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
    x_train_full, y_train_full = splits["train_full"]
    x_test, y_test = splits["test"]

    preprocessor = TextPreprocessor(language=args.language.lower(), advanced=not args.basic_preprocessing)
    x_train_processed = preprocessor.transform_series(x_train_full)
    x_test_processed = preprocessor.transform_series(x_test)

    results = []

    for spec in get_ml_model_specs():
        print(f"\nTraining {spec.name}...")
        search = build_search(
            spec.estimator,
            build_tfidf_vectorizer(),
            spec.param_grid,
            cv=args.cv,
            n_jobs=args.n_jobs,
        )
        with timer() as elapsed:
            search.fit(x_train_processed, y_train_full)
        metrics = evaluate_ml_model(
            model=search.best_estimator_,
            x_test=x_test_processed,
            y_test=y_test,
            training_time=elapsed[0],
            model_name=spec.name,
        )

        model_path = MODELS_DIR / f"{spec.name}.joblib"
        metadata_path = MODELS_DIR / f"{spec.name}_metadata.joblib"
        preprocessor_path = MODELS_DIR / f"{spec.name}_preprocessor.joblib"

        joblib.dump(search.best_estimator_, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        metadata = {
            "model_type": "ml",
            "model_name": spec.name,
            "model_path": str(model_path),
            "preprocessor_path": str(preprocessor_path),
            "label_mapping": {"REAL": 0, "FAKE": 1},
            "decision_strategy": "predict",
        }
        joblib.dump(metadata, metadata_path)

        metrics["best_params"] = search.best_params_
        metrics["metadata_path"] = str(metadata_path)
        metrics["artifact_path"] = str(model_path)
        results.append(metrics)

    results_path = METRICS_DIR / "ml_results.joblib"
    comparison_path = METRICS_DIR / "ml_results.csv"
    joblib.dump(results, results_path)
    pd.DataFrame(results).sort_values("f1", ascending=False).to_csv(comparison_path, index=False)

    print("\nML training finished.")
    print(pd.DataFrame(results).sort_values("f1", ascending=False).to_string(index=False))
    print(f"\nArtifacts saved in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
