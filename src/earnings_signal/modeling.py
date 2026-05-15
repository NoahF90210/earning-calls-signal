from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINE_FEATURES = ["prior_return_30d", "prior_volatility_30d"]
NLP_NUMERIC_FEATURES = [
    *BASELINE_FEATURES,
    "word_count",
    "positive_term_rate",
    "negative_term_rate",
    "uncertainty_term_rate",
    "net_sentiment",
    "question_count",
]
TARGET_CLASS = "target_positive_5d"
TARGET_REGRESSION = "market_adjusted_return_5d"


def time_split(events: pd.DataFrame, test_size: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = events.sort_values("call_date").reset_index(drop=True)
    split_idx = max(1, min(len(ordered) - 1, int(len(ordered) * (1 - test_size))))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def _numeric_pipeline(columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                columns,
            )
        ],
        remainder="drop",
    )


def _nlp_pipeline(columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                columns,
            ),
            (
                "tfidf",
                TfidfVectorizer(max_features=120, min_df=1, ngram_range=(1, 2)),
                "transcript",
            ),
        ],
        remainder="drop",
    )


def train_models(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = time_split(events)
    model_specs = {
        "Price-only baseline": _numeric_pipeline(BASELINE_FEATURES),
        "NLP-enhanced": _nlp_pipeline(NLP_NUMERIC_FEATURES),
    }
    result_rows: list[dict[str, object]] = []
    signal_frames: list[pd.DataFrame] = []

    for model_name, preprocessor in model_specs.items():
        classifier = Pipeline(
            [
                ("features", preprocessor),
                ("model", LogisticRegression(max_iter=1_000, class_weight="balanced")),
            ]
        )
        regressor = Pipeline([("features", preprocessor), ("model", Ridge(alpha=1.0))])

        feature_cols = (
            NLP_NUMERIC_FEATURES + ["transcript"]
            if "NLP" in model_name
            else BASELINE_FEATURES
        )
        classifier.fit(train[feature_cols], train[TARGET_CLASS])
        regressor.fit(train[feature_cols], train[TARGET_REGRESSION])

        probabilities = classifier.predict_proba(test[feature_cols])[:, 1]
        class_predictions = (probabilities >= 0.5).astype(int)
        return_predictions = regressor.predict(test[feature_cols])
        auc = (
            roc_auc_score(test[TARGET_CLASS], probabilities)
            if test[TARGET_CLASS].nunique() > 1
            else np.nan
        )
        rmse = mean_squared_error(test[TARGET_REGRESSION], return_predictions) ** 0.5

        result_rows.append(
            {
                "model": model_name,
                "train_events": len(train),
                "test_events": len(test),
                "test_start": test["call_date"].min(),
                "test_end": test["call_date"].max(),
                "accuracy": accuracy_score(test[TARGET_CLASS], class_predictions),
                "auc": auc,
                "mae": mean_absolute_error(test[TARGET_REGRESSION], return_predictions),
                "rmse": rmse,
            }
        )

        frame = test[
            [
                "event_id",
                "ticker",
                "company",
                "call_date",
                "quarter",
                "year",
                "market_adjusted_return_5d",
                "target_positive_5d",
                "net_sentiment",
                "uncertainty_term_rate",
                "transcript",
            ]
        ].copy()
        frame["model"] = model_name
        frame["signal_score"] = probabilities
        frame["predicted_return_5d"] = return_predictions
        signal_frames.append(frame)

    return pd.DataFrame(result_rows), pd.concat(signal_frames, ignore_index=True)
