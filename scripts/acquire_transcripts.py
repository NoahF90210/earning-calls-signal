from __future__ import annotations

import argparse
from pathlib import Path

from earnings_signal.data import load_huggingface_transcripts, save_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a public Hugging Face transcript dataset for research runs."
    )
    parser.add_argument("dataset_name", help="Example: user/dataset-name from Hugging Face")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/raw/transcripts.csv"))
    args = parser.parse_args()

    transcripts = load_huggingface_transcripts(args.dataset_name, args.split)
    save_table(transcripts, args.output)
    print(f"Wrote {len(transcripts):,} raw transcript rows to {args.output}")


if __name__ == "__main__":
    main()
