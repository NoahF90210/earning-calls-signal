from __future__ import annotations

import pandas as pd

from earnings_signal.data import build_event_dataset, load_transcripts, make_demo_prices
from earnings_signal.features import add_text_features
from earnings_signal.modeling import time_split, train_models


def test_build_event_dataset_creates_primary_target() -> None:
    transcripts = load_transcripts().head(8)
    prices = make_demo_prices(transcripts)
    events = add_text_features(build_event_dataset(transcripts, prices))

    assert "market_adjusted_return_5d" in events.columns
    assert "target_positive_5d" in events.columns
    assert events["target_positive_5d"].isin([0, 1]).all()
    assert events["word_count"].min() > 0


def test_time_split_is_chronological() -> None:
    transcripts = load_transcripts()
    prices = make_demo_prices(transcripts)
    events = add_text_features(build_event_dataset(transcripts, prices))
    train, test = time_split(events)

    assert pd.to_datetime(train["call_date"]).max() <= pd.to_datetime(test["call_date"]).min()


def test_train_models_returns_baseline_and_nlp_outputs() -> None:
    transcripts = load_transcripts()
    prices = make_demo_prices(transcripts)
    events = add_text_features(build_event_dataset(transcripts, prices))
    results, signals = train_models(events)

    assert set(results["model"]) == {"Price-only baseline", "NLP-enhanced"}
    assert not signals.empty
    assert signals["signal_score"].between(0, 1).all()
