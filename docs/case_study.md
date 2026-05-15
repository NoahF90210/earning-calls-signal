# Earnings Call Signal Case Study

## Problem

Earnings calls are high-stakes market events where investors react not only to reported numbers, but also to how management explains demand, risk, margins, and guidance. This study tests whether that language helps predict 5-day post-call stock movement beyond recent price behavior.

## Hypothesis

Companies with more confident, growth-oriented, guidance-rich language should outperform the market over the next five trading days, while calls with caution, risk, and margin pressure language should underperform.

## Data

The workflow is designed for public transcript datasets from Hugging Face or Kaggle and daily OHLCV data from Yahoo Finance through `yfinance`. The repository includes a small sample transcript dataset so the full pipeline, model, and dashboard run locally without paid APIs.

## Model

The baseline model uses only pre-call market features: prior return and volatility. The NLP-enhanced model adds transparent language features including net sentiment, positive and negative term rates, uncertainty term rate, question count, and TF-IDF transcript features. Both models are evaluated with a chronological train/test split to reduce look-ahead bias.

## Outcome

The dashboard makes the result easy to inspect: users can compare baseline vs NLP metrics, explore individual call language, rank bullish and bearish events, and review a simple event-time portfolio simulation. On a larger dataset, the same workflow can test whether language features improve AUC, reduce regression error, or produce a more informative top-vs-bottom signal spread.

## Limitations

The checked-in data is a compact demo dataset, not a full historical transcript corpus. Any production-quality conclusion would require broader coverage, careful transcript licensing, event timestamp checks, transaction cost assumptions, and robustness tests across sectors and market regimes.
