from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict


def get_ml_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            param_grid={
                "classifier__C": [0.5, 1.0, 2.0],
                "classifier__solver": ["liblinear"],
            },
        ),
        ModelSpec(
            name="linear_svm",
            estimator=LinearSVC(class_weight="balanced"),
            param_grid={"classifier__C": [0.5, 1.0, 2.0]},
        ),
        ModelSpec(
            name="naive_bayes",
            estimator=MultinomialNB(),
            param_grid={"classifier__alpha": [0.5, 1.0, 1.5]},
        ),
    ]


def build_search(
    estimator: object,
    vectorizer: object,
    param_grid: dict,
    cv: int = 3,
    n_jobs: int = 1,
) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("classifier", estimator),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
    )
