# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from earnings_signal.backtest import build_long_short_backtest
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
def load_or_build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if EVENTS_FILE.exists() and MODEL_RESULTS_FILE.exists() and SIGNALS_FILE.exists():
        events = load_table(EVENTS_FILE)
        results = load_table(MODEL_RESULTS_FILE)
        signals = load_table(SIGNALS_FILE)
    else:
        transcripts = load_transcripts()
        prices = make_demo_prices(transcripts)
        events = add_text_features(build_event_dataset(transcripts, prices))
        results, signals = train_models(events)
        save_table(events, EVENTS_FILE)
        save_table(results, MODEL_RESULTS_FILE)
        save_table(signals, SIGNALS_FILE)

    events["call_date"] = pd.to_datetime(events["call_date"])
    signals["call_date"] = pd.to_datetime(signals["call_date"])
    backtest = build_long_short_backtest(signals)
    return events, results, signals, backtest


events, results, signals, backtest = load_or_build_outputs()
nlp_signals = signals.query("model == 'NLP-enhanced'").copy()

st.title("Earnings Call Signal")
st.caption(
    "Testing whether management language after earnings adds signal beyond pre-call "
    "price behavior. Primary target: 5-day market-adjusted return versus SPY."
)

best_model = results.sort_values("auc", ascending=False, na_position="last").iloc[0]
nlp_accuracy = results.query("model == 'NLP-enhanced'")["accuracy"].iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Events", f"{len(events):,}")
col2.metric("Tickers", events["ticker"].nunique())
col3.metric("Best AUC", f"{best_model['auc']:.2f}" if pd.notna(best_model["auc"]) else "n/a")
col4.metric("NLP Accuracy", f"{nlp_accuracy:.0%}")

tab_explorer, tab_models, tab_language, tab_portfolio = st.tabs(
    ["Event Explorer", "Model Comparison", "Language Insights", "Portfolio Backtest"]
)

with tab_explorer:
    st.subheader("Event Explorer")
    left, right = st.columns([1, 2])
    tickers = sorted(events["ticker"].unique())
    selected_ticker = left.selectbox("Ticker", tickers)
    event_options = events.query("ticker == @selected_ticker").sort_values("call_date")
    selected_event_id = left.selectbox("Call", event_options["event_id"].tolist())
    event = events.query("event_id == @selected_event_id").iloc[0]

    left.metric("5D Market-Adjusted Return", f"{event['market_adjusted_return_5d']:.2%}")
    left.metric("Net Sentiment", f"{event['net_sentiment']:.2%}")
    left.metric("Uncertainty Term Rate", f"{event['uncertainty_term_rate']:.2%}")

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
        hover_data=["ticker", "company", "call_date"],
        title="Sentiment vs 5-day market-adjusted return",
    )
    st.plotly_chart(scatter, use_container_width=True)

with tab_models:
    st.subheader("Baseline vs NLP-Enhanced Model")
    metrics = results.melt(
        id_vars="model",
        value_vars=["accuracy", "auc", "mae", "rmse"],
        var_name="metric",
        value_name="value",
    )
    st.plotly_chart(
        px.bar(metrics, x="metric", y="value", color="model", barmode="group"),
        use_container_width=True,
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
    st.dataframe(results[result_columns], use_container_width=True, hide_index=True)

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
        use_container_width=True,
        hide_index=True,
    )

with tab_language:
    st.subheader("Language Associated With Positive Post-Call Moves")
    terms = top_language_terms(events, limit=20)
    st.plotly_chart(px.bar(terms, x="lift", y="term", orientation="h"), use_container_width=True)

    distribution = px.histogram(
        events,
        x="market_adjusted_return_5d",
        color="target_positive_5d",
        nbins=18,
        title="Distribution of 5-day market-adjusted returns",
    )
    st.plotly_chart(distribution, use_container_width=True)

with tab_portfolio:
    st.subheader("Top-vs-Bottom Signal Simulation")
    st.write(
        "Each quarter, buy the top-ranked NLP calls and avoid or short the bottom-ranked "
        "calls. This is a simple diagnostic, not a production trading strategy."
    )
    if backtest.empty:
        st.info("Not enough ranked events to build a backtest.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=backtest["period"],
                y=backtest["cumulative_long_short"],
                mode="lines+markers",
                name="Cumulative long-short return",
            )
        )
        fig.update_layout(yaxis_tickformat=".1%", xaxis_title="Quarter", yaxis_title="Return")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(backtest, use_container_width=True, hide_index=True)
