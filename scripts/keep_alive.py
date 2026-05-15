#!/usr/bin/env python3
"""Send periodic HTTP requests so a Streamlit Community Cloud app stays warm.

Free-tier apps often sleep after a period without visitors. A lightweight GET
every few minutes reduces cold starts. This does not guarantee zero downtime:
Streamlit may still restart apps for maintenance.

Usage:
  python3 scripts/keep_alive.py
  python3 scripts/keep_alive.py --url https://your-app.streamlit.app
  STREAMLIT_KEEP_ALIVE_URL=https://... python3 scripts/keep_alive.py

For hands-off scheduling, use the GitHub Actions workflow in this repo or
cron on a machine that stays online.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "https://earning-calls-signal.streamlit.app"


def main() -> int:
    env_url = os.environ.get("STREAMLIT_KEEP_ALIVE_URL", "").strip()
    resolved_default = env_url or DEFAULT_URL

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--url",
        default=resolved_default,
        help=f"App URL (default: env STREAMLIT_KEEP_ALIVE_URL or {DEFAULT_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds",
    )
    args = parser.parse_args()

    if not args.url.startswith("http"):
        print("Error: URL must start with http:// or https://", file=sys.stderr)
        return 1

    req = urllib.request.Request(
        args.url,
        headers={"User-Agent": "StreamlitKeepAlive/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            code = resp.getcode()
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {args.url}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Request failed: {e.reason}", file=sys.stderr)
        return 1

    if code != 200:
        print(f"Unexpected status {code}: {args.url}", file=sys.stderr)
        return 1

    print(f"OK {code} {args.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
