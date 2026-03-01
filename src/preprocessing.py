from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Iterable

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def ensure_nltk_resources() -> None:
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for resource_path, download_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


@dataclass
class TextPreprocessor:
    language: str = "english"
    advanced: bool = True

    def __post_init__(self) -> None:
        ensure_nltk_resources()
        self.stop_words = set(stopwords.words(self.language))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"<.*?>", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess(self, text: str) -> str:
        text = self.clean_text(text)
        tokens = nltk.word_tokenize(text)
        processed_tokens = []
        for token in tokens:
            if token in self.stop_words or len(token) < 2:
                continue
            if self.advanced:
                token = self.lemmatizer.lemmatize(token)
            processed_tokens.append(token)
        return " ".join(processed_tokens)

    def transform_series(self, series: Iterable[str]) -> pd.Series:
        return pd.Series([self.preprocess(text) for text in series], dtype="string")


def build_tfidf_vectorizer(max_features: int = 20000) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def fit_tokenizer(texts: Iterable[str], num_words: int = 20000):
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(list(texts))
    return tokenizer


def tokenize_and_pad(
    tokenizer: Any,
    texts: Iterable[str],
    max_length: int = 300,
):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequences = tokenizer.texts_to_sequences(list(texts))
    return pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")


def save_tokenizer(tokenizer: Tokenizer, path) -> None:
    joblib.dump(tokenizer, path)


def load_tokenizer(path):
    return joblib.load(path)
