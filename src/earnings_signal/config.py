from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

SAMPLE_TRANSCRIPTS_FILE = SAMPLE_DATA_DIR / "sample_transcripts.csv"

EVENTS_FILE = PROCESSED_DATA_DIR / "events.csv"
MODEL_RESULTS_FILE = PROCESSED_DATA_DIR / "model_results.csv"
SIGNALS_FILE = PROCESSED_DATA_DIR / "signals.csv"
BACKTEST_FILE = PROCESSED_DATA_DIR / "backtest.csv"

BENCHMARK_TICKER = "SPY"
PRIMARY_HORIZON_DAYS = 5
