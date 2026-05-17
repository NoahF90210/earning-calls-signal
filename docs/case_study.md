# Earnings Call Signal Case Study

**Research question:** Does earnings-call language add signal beyond recent price behavior for short-term market-adjusted returns?

## Problem

Earnings calls are high-stakes market events where investors react not only to reported numbers, but also to how management explains demand, risk, margins, and guidance. This study turns transcript language into transparent model features and tests whether those features improve prediction of 5-day post-call returns after subtracting SPY's return.

## Hypothesis

Companies with more confident, growth-oriented, guidance-rich language should outperform the market over the next five trading days, while calls with caution, risk, and margin pressure language should underperform.

## Data

The workflow is designed for public transcript datasets from Hugging Face or Kaggle and daily OHLCV data from Yahoo Finance through `yfinance`. The repository includes a 48-event demo transcript dataset and deterministic demo prices so the full pipeline, model, and dashboard run locally without paid APIs.

Primary target:

`market_adjusted_return_5d = stock_return_5d - SPY_return_5d`

## Workflow

1. Join transcript events to stock and SPY prices around the call date.
2. Engineer price features: prior 30-day return and prior 30-day volatility.
3. Engineer language features: net sentiment, positive/negative term rates, uncertainty term rate, question count, average sentence length, and TF-IDF transcript terms.
4. Train a price-only baseline and an NLP-enhanced model.
5. Evaluate on a chronological holdout and rank holdout calls by bullish probability.
6. Compare predicted rankings with realized 5-day market-adjusted returns.

## Model Comparison

The baseline model uses only pre-call market features: prior return and volatility. The NLP-enhanced model adds the transparent language features above. Both models are evaluated with a chronological train/test split to reduce look-ahead bias.

## Outcome

The dashboard makes the result easy to inspect: users can compare baseline vs. NLP metrics, explore individual call language, rank bullish and bearish events, and review a simple top-vs-bottom realized return diagnostic.

Supported demo results from the checked-in dataset:

- 48 events across 48 tickers, with a 15-event chronological holdout from 2024-04-30 to 2024-06-25.
- Price-only baseline: 53% accuracy, 0.54 AUC, 0.55% MAE, and 0.67% RMSE.
- NLP-enhanced model: 80% accuracy, 1.00 AUC, 0.40% MAE, and 0.49% RMSE.
- Top-third NLP-ranked holdout events averaged 0.21% 5-day market-adjusted return; bottom-third events averaged -1.06%.

Because demo prices are deterministic and seeded from `demo_signal`, the honest conclusion is that the dashboard can recover an embedded language signal in a controlled demo. It does not yet establish that earnings-call language adds durable real-market alpha.

## Limitations

The checked-in data is a compact demo dataset, not a full historical transcript corpus. This is research and education material only, not investment advice. Any production-quality conclusion would require broader transcript coverage, careful transcript licensing, event timestamp validation, real adjusted prices, transaction cost and slippage assumptions, survivorship-bias checks, and robustness tests across sectors and market regimes.
