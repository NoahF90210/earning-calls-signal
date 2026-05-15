from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from earnings_signal.config import BENCHMARK_TICKER, EVENTS_FILE, PRIMARY_HORIZON_DAYS
from earnings_signal.data import (
    build_event_dataset,
    download_prices,
    load_transcripts,
    make_demo_prices,
    save_table,
)
from earnings_signal.features import add_text_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event-level earnings-call rows.")
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=None,
        help="CSV/parquet transcript file.",
    )
    parser.add_argument("--output", type=Path, default=EVENTS_FILE)
    parser.add_argument("--use-yfinance", action="store_true")
    parser.add_argument("--download-prices", action="store_true")
    parser.add_argument("--benchmark", default=BENCHMARK_TICKER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcripts = load_transcripts(args.transcripts) if args.transcripts else load_transcripts()

    if args.use_yfinance or args.download_prices:
        start = transcripts["call_date"].min().date().isoformat()
        end_date = transcripts["call_date"].max() + PRIMARY_HORIZON_DAYS * 3
        end = end_date.date().isoformat()
        tickers = transcripts["ticker"].unique().tolist()
        prices = download_prices(tickers, start, end, args.benchmark)
    else:
        prices = make_demo_prices(transcripts, args.benchmark)

    events = build_event_dataset(transcripts, prices, args.benchmark, PRIMARY_HORIZON_DAYS)
    events = add_text_features(events)
    save_table(events, args.output)
    print(f"Wrote {len(events):,} event rows to {args.output}")


if __name__ == "__main__":
    main()
