# Earnings Call Signal

**A Streamlit research dashboard testing whether earnings-call language adds signal beyond price behavior for 5-day market-adjusted returns.**

[Live app](https://earning-calls-signal.streamlit.app) | [Case study](docs/case_study.md)


## Portfolio Snapshot

- **Question:** Does management language on earnings calls add signal beyond recent price behavior?
- **Target:** `market_adjusted_return_5d = stock_return_5d - SPY_return_5d`.
- **Models compared:** price-only baseline vs. NLP-enhanced model with sentiment, uncertainty, Q&A, sentence-length, and TF-IDF features.
- **Workflow:** transcript events -> price/benchmark join -> feature engineering -> chronological validation -> signal rankings -> realized return check.
- **Demo data:** 48 checked-in sample events, deterministic demo prices, and a 15-event chronological holdout so the full app runs without paid transcript APIs.
- **Current answer:** the repo demonstrates the research workflow and recovers the embedded demo signal, but it does **not** prove a real-market trading edge yet.

## 30-Second Demo Path

1. Open the [live Streamlit app](https://earning-calls-signal.streamlit.app). If Streamlit is asleep, give it a minute to wake.
2. In **Event Explorer**, pick a ticker and inspect the transcript, sentiment, signal words, and realized 5-day market-adjusted return.
3. In **Model Comparison**, compare the price-only baseline with the NLP-enhanced model on the chronological holdout.
4. In **Signal Check**, review the bullish/bearish ranking and whether top-ranked calls had better realized returns than bottom-ranked calls.
5. Open **Methodology and interpretation** to see the data join, feature set, validation design, and caveats.


View dashboard: https://earning-calls-signal.streamlit.app

## Why It Matters

Earnings calls are high-stakes market events. Investors react not only to reported numbers, but also to how management talks about demand, margins, risk, and guidance. This project turns that language into measurable features and asks whether those features improve short-term, market-adjusted return prediction beyond recent price behavior.

## Key Findings

- The checked-in demo builds an end-to-end event dataset: 48 transcript events across 48 tickers from 2021-01-27 to 2024-06-25.
- On the deterministic 15-event chronological holdout, the NLP-enhanced demo model scores 80% accuracy and 1.00 AUC versus 53% accuracy and 0.54 AUC for the price-only baseline.
- Return error is lower in the demo holdout for the NLP-enhanced model: 0.40% MAE and 0.49% RMSE versus 0.55% MAE and 0.67% RMSE for the baseline.
- The top third of NLP-ranked holdout events averaged a 0.21% 5-day market-adjusted return, while the bottom third averaged -1.06%, a 1.27 percentage-point spread.
- These findings are intentionally limited: demo prices are deterministic and seeded from `demo_signal`, so the result shows that the workflow can recover an embedded language signal, not that earnings-call NLP generates investable alpha.

## Limitations

This is a research dashboard, not investment advice. The included dataset is small and demo-oriented, and the default price series is deterministic. A production-quality study would need a much larger transcript corpus, licensed transcript coverage, event timestamp validation, survivorship-bias checks, transaction cost and slippage assumptions, sector/time robustness checks, and validation on real post-call prices.

## What The Dashboard Shows

- Event explorer with transcript text, ticker context, sentiment, signal words, and realized 5-day market-adjusted return.
- Baseline vs. NLP-enhanced model comparison across accuracy, AUC, MAE, and RMSE.
- Signal leaderboard ranking the most bullish and bearish holdout calls.
- Language insights showing terms associated with positive post-call moves in the demo data.
- Top-vs-bottom signal diagnostic that compares ranked probabilities with realized returns.

## Technical Build

- Python, pandas, scikit-learn, Plotly, Streamlit, and yfinance.
- Event-level dataset builder joining transcript events to stock and benchmark returns.
- Transparent text features: sentiment term rates, uncertainty terms, question count, average sentence length, and TF-IDF transcript terms.
- Chronological train/test validation to better approximate a live prediction setting.
- Reusable package code under `src/earnings_signal/` with regression tests for the core pipeline.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 scripts/build_dataset.py
python3 scripts/train_models.py
streamlit run app/dashboard.py
```

## Case Study

See `docs/case_study.md` for a concise write-up of the problem, method, supported demo results, and limitations.
