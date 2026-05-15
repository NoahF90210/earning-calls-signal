from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from earnings_signal.config import BENCHMARK_TICKER, PRIMARY_HORIZON_DAYS, SAMPLE_TRANSCRIPTS_FILE

REQUIRED_TRANSCRIPT_COLUMNS = {"ticker", "company", "call_date", "quarter", "year", "transcript"}


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV or parquet table."""
    data_path = Path(path)
    if data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)
    if data_path.suffix == ".csv":
        return pd.read_csv(data_path)
    raise ValueError(f"Unsupported file type: {data_path.suffix}")


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a CSV or parquet table, creating parent folders."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {output_path.suffix}")


def load_transcripts(path: str | Path = SAMPLE_TRANSCRIPTS_FILE) -> pd.DataFrame:
    """Load and validate transcript events from a local public or demo dataset."""
    transcripts = load_table(path).copy()
    missing = REQUIRED_TRANSCRIPT_COLUMNS - set(transcripts.columns)
    if missing:
        raise ValueError(f"Transcript file is missing required columns: {sorted(missing)}")

    transcripts["ticker"] = transcripts["ticker"].str.upper().str.strip()
    transcripts["call_date"] = pd.to_datetime(transcripts["call_date"])
    transcripts["event_id"] = (
        transcripts["ticker"]
        + "_"
        + transcripts["year"].astype(str)
        + "_Q"
        + transcripts["quarter"].astype(str)
    )
    return transcripts.sort_values(["call_date", "ticker"]).reset_index(drop=True)


def load_huggingface_transcripts(dataset_name: str, split: str = "train") -> pd.DataFrame:
    """Load a Hugging Face transcript dataset when the optional research dependency is installed."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install optional research dependencies with `python -m pip install -e '.[research]'`."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    return dataset.to_pandas()


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    benchmark: str = BENCHMARK_TICKER,
) -> pd.DataFrame:
    """Download adjusted daily close prices from Yahoo Finance."""
    symbols = sorted(set([*tickers, benchmark]))
    prices = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(symbols[0])
    return prices.sort_index().rename_axis("date")


def make_demo_prices(
    transcripts: pd.DataFrame,
    benchmark: str = BENCHMARK_TICKER,
    days_before: int = 45,
    days_after: int = 30,
) -> pd.DataFrame:
    """Create deterministic demo prices so the full workflow runs without network access."""
    tickers = sorted(set(transcripts["ticker"]).union({benchmark}))
    start = transcripts["call_date"].min() - pd.Timedelta(days=days_before)
    end = transcripts["call_date"].max() + pd.Timedelta(days=days_after)
    dates = pd.bdate_range(start, end)
    base = pd.DataFrame(index=dates)

    for idx, ticker in enumerate(tickers):
        daily_trend = 0.00035 + idx * 0.00003
        wave = np.sin(np.linspace(0, 8, len(dates)) + idx) * 0.006
        returns = daily_trend + wave / 20
        prices = 100 + idx * 7
        series = [prices]
        for ret in returns[1:]:
            series.append(series[-1] * (1 + ret))
        base[ticker] = series

    impact_map = transcripts.set_index("ticker")["demo_signal"].to_dict()
    for _, row in transcripts.iterrows():
        ticker = row["ticker"]
        event_idx = dates.searchsorted(row["call_date"])
        impact = float(impact_map.get(ticker, 0.0)) * 0.012
        for offset in range(1, 7):
            if event_idx + offset < len(dates):
                column_idx = base.columns.get_loc(ticker)
                base.iloc[event_idx + offset :, column_idx] *= 1 + impact / 6

    return base.rename_axis("date")


def _price_at_or_after(
    prices: pd.DataFrame,
    ticker: str,
    date: pd.Timestamp,
    offset: int = 0,
) -> float:
    index = prices.index[prices.index >= date]
    if len(index) <= offset or ticker not in prices.columns:
        return np.nan
    return float(prices.loc[index[offset], ticker])


def build_event_dataset(
    transcripts: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: str = BENCHMARK_TICKER,
    horizon_days: int = PRIMARY_HORIZON_DAYS,
) -> pd.DataFrame:
    """Join transcript events to market data and create the 5-day adjusted return target."""
    rows: list[dict[str, object]] = []
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)

    for _, event in transcripts.iterrows():
        ticker = event["ticker"]
        call_date = pd.Timestamp(event["call_date"])
        event_price = _price_at_or_after(prices, ticker, call_date)
        future_price = _price_at_or_after(prices, ticker, call_date, horizon_days)
        benchmark_event = _price_at_or_after(prices, benchmark, call_date)
        benchmark_future = _price_at_or_after(prices, benchmark, call_date, horizon_days)
        prior_start = _price_at_or_after(prices, ticker, call_date - pd.Timedelta(days=45))

        raw_return_5d = future_price / event_price - 1 if event_price else np.nan
        benchmark_return_5d = (
            benchmark_future / benchmark_event - 1 if benchmark_event else np.nan
        )
        prior_return_30d = event_price / prior_start - 1 if prior_start else np.nan
        prior_window = prices.loc[prices.index < call_date, ticker].tail(30).pct_change()
        prior_volatility_30d = float(prior_window.std()) if len(prior_window) else np.nan

        rows.append(
            {
                **event.to_dict(),
                "event_price": event_price,
                "future_price_5d": future_price,
                "raw_return_5d": raw_return_5d,
                "benchmark_return_5d": benchmark_return_5d,
                "market_adjusted_return_5d": raw_return_5d - benchmark_return_5d,
                "target_positive_5d": int(raw_return_5d - benchmark_return_5d > 0),
                "prior_return_30d": prior_return_30d,
                "prior_volatility_30d": prior_volatility_30d,
            }
        )

    return pd.DataFrame(rows).dropna(subset=["market_adjusted_return_5d"]).reset_index(drop=True)
