from __future__ import annotations

from pathlib import Path

import joblib
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing import load_tokenizer, tokenize_and_pad


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=5, description="News content to classify.")


class PredictionResponse(BaseModel):
    label: str
    probability_fake: float
    probability_real: float
    model_name: str


app = FastAPI(title="Fake News Detection API", version="1.0.0")

ARTIFACT_CACHE = {"metadata": None, "model": None, "preprocessor": None, "tokenizer": None}


def load_artifacts():
    metadata_path = Path("models/best_model.joblib")
    if not metadata_path.exists():
        raise FileNotFoundError(
            "models/best_model.joblib not found. Train models and run python -m src.evaluate first."
        )

    if ARTIFACT_CACHE["metadata"] is not None:
        return ARTIFACT_CACHE

    metadata = joblib.load(metadata_path)
    preprocessor = joblib.load(metadata["preprocessor_path"])
    artifacts = {
        "metadata": metadata,
        "preprocessor": preprocessor,
        "model": None,
        "tokenizer": None,
    }

    if metadata["model_type"] == "ml":
        artifacts["model"] = joblib.load(metadata["model_path"])
    elif metadata["model_type"] == "dl":
        from tensorflow.keras.models import load_model

        artifacts["model"] = load_model(metadata["model_path"])
        artifacts["tokenizer"] = load_tokenizer(metadata["tokenizer_path"])
    else:
        raise ValueError(f"Unsupported model type: {metadata['model_type']}")

    ARTIFACT_CACHE.update(artifacts)
    return ARTIFACT_CACHE


def predict_text(text: str) -> PredictionResponse:
    artifacts = load_artifacts()
    metadata = artifacts["metadata"]
    cleaned_text = artifacts["preprocessor"].preprocess(text)

    if metadata["model_type"] == "ml":
        model = artifacts["model"]
        if hasattr(model, "predict_proba"):
            probability_fake = float(model.predict_proba([cleaned_text])[0][1])
            prediction = int(probability_fake >= 0.5)
        elif hasattr(model, "decision_function"):
            decision = float(model.decision_function([cleaned_text])[0])
            probability_fake = 1.0 / (1.0 + math.exp(-decision))
            prediction = int(probability_fake >= 0.5)
        else:
            prediction = int(model.predict([cleaned_text])[0])
            probability_fake = float(prediction)
    else:
        tokenizer = artifacts["tokenizer"]
        sequence = tokenize_and_pad(tokenizer, [cleaned_text], max_length=metadata["max_length"])
        probability_fake = float(artifacts["model"].predict(sequence, verbose=0).ravel()[0])
        prediction = int(probability_fake >= metadata.get("threshold", 0.5))

    label = "FAKE" if prediction == 1 else "REAL"
    return PredictionResponse(
        label=label,
        probability_fake=round(probability_fake, 4),
        probability_real=round(1 - probability_fake, 4),
        model_name=metadata["model_name"],
    )


@app.get("/")
def root():
    return {"message": "Fake News Detection API is running."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        return predict_text(request.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
