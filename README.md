# Earnings Call Signal

An interactive research dashboard that tests whether earnings-call language helps predict short-term stock movement after earnings.

It turns transcript text into measurable signals, compares an NLP-enhanced model against a price-only baseline, ranks bullish and bearish calls, and visualizes the results in Streamlit.

View dashboard: https://earning-calls-signal.streamlit.app

## Why It Matters

Earnings calls are high-stakes market events. Investors react not only to reported numbers, but also to how management talks about demand, margins, risk, and guidance. This project asks whether that language adds signal beyond recent price behavior.

Primary target:

`market_adjusted_return_5d = stock_return_5d - SPY_return_5d`

## What The Dashboard Shows

- Event explorer with transcript text, ticker context, sentiment, and 5-day market-adjusted return.
- Baseline vs NLP-enhanced model comparison across accuracy, AUC, MAE, and RMSE.
- Signal leaderboard ranking the most bullish and bearish earnings calls.
- Language insights showing terms associated with positive post-call moves.
- Simple top-vs-bottom signal simulation for portfolio-style diagnostics.

## Technical Build

- Python, pandas, scikit-learn, Plotly, Streamlit, and yfinance.
- Event-level dataset builder joining transcript events to stock and benchmark returns.
- Transparent text features: sentiment term rates, uncertainty terms, question count, average sentence length, and TF-IDF transcript terms.
- Chronological train/test validation to better approximate a live prediction setting.
- Reusable package code under `src/earnings_signal/` with regression tests for the core pipeline.

## Data

The repository includes a small demo transcript dataset so the full workflow can be inspected without paid APIs or large downloads. The pipeline also supports larger public transcript files and live Yahoo Finance prices.

## Limitations

The included dataset is intentionally small and deterministic, so the dashboard should be read as a demonstration of the workflow rather than proof of a durable trading edge. A full study would need a larger public transcript corpus, careful event timestamp validation, transaction cost assumptions, and robustness checks across sectors and market regimes.

This repository is for research and education only. It is not investment advice.

## Case Study

See `docs/case_study.md` for a short write-up of the problem, method, results, and limitations.
