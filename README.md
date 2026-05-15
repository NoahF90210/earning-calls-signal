# Earnings Call Signal

Earnings calls are high-stakes market events where investors react not only to reported numbers, but also to how management explains demand, margins, risk, and guidance. This project tests whether transcript language improves prediction of the next **5 trading days of market-adjusted return** after an earnings call.

## Research Question

Does management language add signal beyond recent price behavior for predicting short-term post-earnings stock movement?

The primary target is:

`market_adjusted_return_5d = stock_return_5d - SPY_return_5d`

## What Is Included

- Public-data workflow for earnings-call transcripts plus Yahoo Finance market data.
- Event-level dataset builder that joins transcript events to stock and `SPY` prices.
- Price-only baseline and NLP-enhanced model with chronological train/test validation.
- Streamlit dashboard with event explorer, model comparison, signal leaderboard, language insights, and a long/short backtest panel.
- Offline demo dataset so the pipeline runs without credentials or large downloads.

## Data

The repository includes a small checked-in demo transcript file at `data/sample/sample_transcripts.csv`. The sample text is hand-authored demo content that mirrors earnings-call language patterns and is not presented as a scraped transcript corpus.

For a larger public-data run, replace the sample file with a CSV or parquet containing:

`ticker`, `company`, `sector`, `call_date`, `quarter`, `year`, `transcript`

Practical transcript sources:

- Public Hugging Face earnings-call datasets with ticker/date/transcript fields.
- Kaggle NASDAQ or S&P 500 earnings-call transcript datasets.
- Company investor-relations transcripts when redistribution rights are clear.

Market data comes from `yfinance` daily adjusted closes. The benchmark defaults to `SPY`.

## Method

The baseline uses pre-call market features:

- Prior 30-day return.
- Prior 30-day volatility.

The NLP-enhanced model adds transparent language features:

- Positive, negative, risk, and guidance term rates.
- Net sentiment.
- Question count.
- Average sentence length.

Validation is chronological rather than random, so later calls are held out to better approximate a live signal.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/build_dataset.py
python scripts/train_models.py
streamlit run app/dashboard.py
```

To use live market data for a real transcript file:

```bash
python scripts/build_dataset.py --transcripts data/raw/my_transcripts.csv --use-yfinance
python scripts/train_models.py
```

## Dashboard

- Event explorer: transcript text, return outcome, sector, and ticker context.
- Model comparison: baseline vs NLP-enhanced accuracy, AUC, MAE, and RMSE.
- Signal leaderboard: the most bullish and bearish calls ranked by predicted probability.
- Language insights: terms associated with positive post-call moves and return distributions.
- Backtest panel: top-ranked versus bottom-ranked signal performance by event date.

## Limitations

The checked-in dataset is intentionally small and deterministic, so it should be treated as a local demo of the workflow rather than evidence of a production trading signal. A full study should rerun the pipeline on a larger public transcript corpus, inspect ticker/date coverage, and evaluate stability across sectors and market regimes.

This repository is for research and education only. It is not investment advice.

## Case Study

See `docs/case_study.md` for a concise write-up of the problem, data, method, results, and limitations.
