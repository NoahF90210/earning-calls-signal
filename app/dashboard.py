# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from earnings_signal.config import EVENTS_FILE, MODEL_RESULTS_FILE, SIGNALS_FILE
from earnings_signal.data import (
    build_event_dataset,
    load_table,
    load_transcripts,
    make_demo_prices,
    save_table,
)
from earnings_signal.features import add_text_features, top_language_terms
from earnings_signal.modeling import train_models

st.set_page_config(page_title="Earnings Call Signal", layout="wide")


@st.cache_data(show_spinner=False)
def load_or_build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_files_exist = (
        EVENTS_FILE.exists() and MODEL_RESULTS_FILE.exists() and SIGNALS_FILE.exists()
    )
    transcripts = load_transcripts()
    if output_files_exist:
        events = load_table(EVENTS_FILE)
    needs_rebuild = (
        (not output_files_exist)
        or "avg_sentence_length" not in events.columns
        or len(events) != len(transcripts)
    )
    if needs_rebuild:
        prices = make_demo_prices(transcripts)
        events = add_text_features(build_event_dataset(transcripts, prices))
        results, signals = train_models(events)
        save_table(events, EVENTS_FILE)
        save_table(results, MODEL_RESULTS_FILE)
        save_table(signals, SIGNALS_FILE)
    else:
        results = load_table(MODEL_RESULTS_FILE)
        signals = load_table(SIGNALS_FILE)

    events["call_date"] = pd.to_datetime(events["call_date"])
    signals["call_date"] = pd.to_datetime(signals["call_date"])
    return events, results, signals


def language_tone(row: pd.Series) -> str:
    if row["net_sentiment"] >= 0.05:
        return "Bullish language"
    if row["net_sentiment"] <= -0.02:
        return "Cautious language"
    return "Mixed language"


def term_count(row: pd.Series, rate_column: str) -> int:
    return int(round(row[rate_column] * row["word_count"]))


events, results, signals = load_or_build_outputs()
events["language_tone"] = events.apply(language_tone, axis=1)
nlp_signals = signals.query("model == 'NLP-enhanced'").copy()

st.title("Earnings Call Signal")
st.caption(
    "Testing whether management language after earnings adds signal beyond pre-call "
    "price behavior. Primary target: 5-day market-adjusted return versus SPY."
)

best_model = results.sort_values("auc", ascending=False, na_position="last").iloc[0]
nlp_accuracy = results.query("model == 'NLP-enhanced'")["accuracy"].iloc[0]
summary_col, method_col = st.columns([1.15, 1])
summary_col.markdown(
    """
    **What this answers:** can transcript language improve short-term post-earnings
    return prediction versus price behavior alone?

    **Workflow:** transcripts -> event-level returns -> text features + TF-IDF -> model
    comparison -> ranked bullish/bearish signals.
    """
)
method_col.info(
    "Demo mode uses a small deterministic sample dataset. Treat the results as a "
    "workflow demonstration until rerun on a larger public transcript corpus with live prices.",
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Events", f"{len(events):,}")
col2.metric("Tickers", events["ticker"].nunique())
col3.metric("Best AUC", f"{best_model['auc']:.2f}" if pd.notna(best_model["auc"]) else "n/a")
col4.metric("NLP Accuracy", f"{nlp_accuracy:.0%}")

with st.expander("Methodology and interpretation"):
    st.markdown(
        """
        - The target is the next 5 trading days of stock return minus benchmark return.
        - The baseline model uses prior 30-day return and volatility.
        - The NLP-enhanced model adds sentiment, uncertainty, question count,
          average sentence length, and TF-IDF transcript terms.
        - Validation is chronological, so later calls are held out from training.
        - The backtest panel is a diagnostic top-vs-bottom signal spread,
          not a production trading strategy.
        """
    )

tab_explorer, tab_models, tab_language, tab_signal = st.tabs(
    ["Event Explorer", "Model Comparison", "Language Insights", "Signal Check"]
)

with tab_explorer:
    st.subheader("Event Explorer")
    left, right = st.columns([1, 2])
    tickers = sorted(events["ticker"].unique())
    selected_ticker = left.selectbox("Ticker", tickers)
    event_options = events.query("ticker == @selected_ticker").sort_values("call_date")
    selected_event_id = left.selectbox("Call", event_options["event_id"].tolist())
    event = events.query("event_id == @selected_event_id").iloc[0]
    positive_terms = term_count(event, "positive_term_rate")
    caution_terms = term_count(event, "negative_term_rate") + term_count(
        event, "uncertainty_term_rate"
    )

    left.metric("5D Market-Adjusted Return", f"{event['market_adjusted_return_5d']:.2%}")
    left.metric("Language Tone", event["language_tone"])
    left.metric("Signal Words", f"{positive_terms} positive / {caution_terms} caution")
    left.caption("Signal words are matched terms in this transcript, not a market forecast.")

    event_signal = nlp_signals.query("event_id == @selected_event_id")
    if not event_signal.empty:
        left.metric("NLP Bullish Probability", f"{event_signal['signal_score'].iloc[0]:.0%}")

    right.write(f"**{event['company']} {int(event['year'])} Q{int(event['quarter'])}**")
    right.write(event["transcript"])

    scatter = px.scatter(
        events,
        x="net_sentiment",
        y="market_adjusted_return_5d",
        color="sector",
        size="word_count",
        size_max=18,
        hover_data=["ticker", "company", "call_date"],
        title="Sentiment vs 5-day market-adjusted return",
    )
    scatter.update_traces(marker={"line": {"width": 1, "color": "white"}})
    scatter.update_layout(yaxis_tickformat=".1%", xaxis_tickformat=".1%")
    st.plotly_chart(scatter, width="stretch")

with tab_models:
    st.subheader("Baseline vs NLP-Enhanced Model")
    classification_metrics = results.melt(
        id_vars="model",
        value_vars=["accuracy", "auc"],
        var_name="metric",
        value_name="value",
    )
    error_metrics = results.melt(
        id_vars="model",
        value_vars=["mae", "rmse"],
        var_name="metric",
        value_name="value",
    )
    st.caption("Accuracy and AUC are higher-is-better; MAE and RMSE are lower-is-better.")
    st.plotly_chart(
        px.bar(
            classification_metrics,
            x="metric",
            y="value",
            color="model",
            barmode="group",
            title="Classification metrics",
        ),
        width="stretch",
    )
    st.plotly_chart(
        px.bar(
            error_metrics,
            x="metric",
            y="value",
            color="model",
            barmode="group",
            title="Return error metrics",
        ),
        width="stretch",
    )
    result_columns = [
        "model",
        "train_events",
        "test_events",
        "test_start",
        "test_end",
        "accuracy",
        "auc",
        "mae",
        "rmse",
    ]
    st.dataframe(results[result_columns], width="stretch", hide_index=True)

    st.subheader("Signal Leaderboard")
    direction = st.radio("Rank", ["Most bullish", "Most bearish"], horizontal=True)
    ranked = nlp_signals.sort_values("signal_score", ascending=direction == "Most bearish")
    st.dataframe(
        ranked[
            [
                "ticker",
                "company",
                "call_date",
                "signal_score",
                "predicted_return_5d",
                "market_adjusted_return_5d",
                "net_sentiment",
            ]
        ].head(10),
        width="stretch",
        hide_index=True,
    )

with tab_language:
    st.subheader("Language Tone vs Return")
    tone_summary = (
        events.groupby("language_tone", as_index=False)
        .agg(
            avg_return=("market_adjusted_return_5d", "mean"),
            events=("event_id", "count"),
        )
        .sort_values("avg_return")
    )
    tone_chart = px.bar(
        tone_summary,
        x="language_tone",
        y="avg_return",
        text="events",
        title="Average 5-day market-adjusted return by transcript tone",
        labels={
            "language_tone": "Transcript tone",
            "avg_return": "Average 5-day market-adjusted return",
            "events": "Events",
        },
    )
    tone_chart.update_traces(texttemplate="%{text} events", textposition="outside")
    tone_chart.update_layout(yaxis_tickformat=".1%")
    st.plotly_chart(tone_chart, width="stretch")

    st.subheader("Words More Common In Positive Calls")
    terms = top_language_terms(events, limit=20)
    term_chart = px.bar(
        terms,
        x="lift",
        y="term",
        orientation="h",
        labels={"lift": "Positive-call lift", "term": "Term"},
        title="Terms that appear more often in positive-return events",
    )
    term_chart.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(term_chart, width="stretch")

with tab_signal:
    st.subheader("Did The Ranking Match The Outcome?")
    st.write(
        "This view compares the NLP model's bullish probability with the realized "
        "5-day market-adjusted return on the holdout events."
    )
    bucket_size = max(1, len(nlp_signals) // 3)
    top_bucket = nlp_signals.nlargest(bucket_size, "signal_score")
    bottom_bucket = nlp_signals.nsmallest(bucket_size, "signal_score")
    spread = (
        top_bucket["market_adjusted_return_5d"].mean()
        - bottom_bucket["market_adjusted_return_5d"].mean()
    )
    score_col1, score_col2, score_col3 = st.columns(3)
    score_col1.metric(
        "Top-ranked avg return", f"{top_bucket['market_adjusted_return_5d'].mean():.2%}"
    )
    score_col2.metric(
        "Bottom-ranked avg return",
        f"{bottom_bucket['market_adjusted_return_5d'].mean():.2%}",
    )
    score_col3.metric("Top-minus-bottom spread", f"{spread:.2%}")

    signal_chart = px.scatter(
        nlp_signals,
        x="signal_score",
        y="market_adjusted_return_5d",
        color="target_positive_5d",
        size="signal_score",
        size_max=20,
        text="ticker",
        hover_data=["company", "call_date"],
        labels={
            "signal_score": "Model bullish probability",
            "market_adjusted_return_5d": "Realized 5-day market-adjusted return",
            "target_positive_5d": "Positive return",
        },
        title="Predicted bullishness vs realized return",
    )
    signal_chart.add_hline(y=0, line_dash="dash", line_color="gray")
    signal_chart.add_vline(x=0.5, line_dash="dash", line_color="gray")
    signal_chart.update_traces(
        marker={"line": {"width": 1, "color": "white"}},
        textposition="top center",
    )
    signal_chart.update_layout(
        yaxis_tickformat=".1%",
        xaxis_tickformat=".0%",
        uniformtext_minsize=9,
        uniformtext_mode="show",
    )
    st.plotly_chart(signal_chart, width="stretch")

    st.dataframe(
        nlp_signals[
            [
                "ticker",
                "company",
                "call_date",
                "signal_score",
                "predicted_return_5d",
                "market_adjusted_return_5d",
            ]
        ].sort_values("signal_score", ascending=False),
        width="stretch",
        hide_index=True,
    )
