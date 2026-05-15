from __future__ import annotations

import argparse
from pathlib import Path

from earnings_signal.backtest import build_long_short_backtest
from earnings_signal.config import BACKTEST_FILE, EVENTS_FILE, MODEL_RESULTS_FILE, SIGNALS_FILE
from earnings_signal.data import load_table, save_table
from earnings_signal.modeling import train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline and NLP earnings-call models.")
    parser.add_argument("--events", type=Path, default=EVENTS_FILE, help="Event dataset.")
    parser.add_argument("--results", type=Path, default=MODEL_RESULTS_FILE, help="Metrics output.")
    parser.add_argument("--signals", type=Path, default=SIGNALS_FILE, help="Signals output.")
    parser.add_argument("--backtest", type=Path, default=BACKTEST_FILE, help="Backtest output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_table(args.events)
    results, signals = train_models(events)
    backtest = build_long_short_backtest(signals)
    save_table(results, args.results)
    save_table(signals, args.signals)
    save_table(backtest, args.backtest)
    print(f"Wrote model metrics to {args.results}")
    print(f"Wrote {len(signals):,} ranked signals to {args.signals}")
    print(f"Wrote backtest panel to {args.backtest}")


if __name__ == "__main__":
    main()
