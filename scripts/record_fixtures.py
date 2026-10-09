"""Re-record AlphaVantage fixtures for the offline test suite.

Needs ALPHAVANTAGE_API_KEY. Only EARNINGS and GLOBAL_QUOTE are reachable on a
free key; the other two are recorded only if the key has premium access, and
skipped with a note otherwise.

    python scripts/record_fixtures.py NVDA AMD INTC
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx

FIXTURES = Path(__file__).resolve().parent.parent / "tests/fixtures/alphavantage"
FREE = ["EARNINGS", "GLOBAL_QUOTE"]
PREMIUM = ["COMPANY_OVERVIEW", "INCOME_STATEMENT"]

# AlphaVantage returns HTTP 200 with one of these keys instead of an error code.
ERROR_KEYS = ("Note", "Information", "Error Message")


def fetch(function: str, symbol: str, key: str) -> dict:
    r = httpx.get(
        "https://www.alphavantage.co/query",
        params={"function": function, "symbol": symbol, "apikey": key},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()


def main(symbols: list[str]) -> int:
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        print("ALPHAVANTAGE_API_KEY is not set", file=sys.stderr)
        return 1

    FIXTURES.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        for function in FREE + PREMIUM:
            payload = fetch(function, symbol, key)
            problem = next((k for k in ERROR_KEYS if k in payload), None)
            if problem:
                print(f"skip {function} {symbol}: {payload[problem][:80]}", file=sys.stderr)
                continue
            out = FIXTURES / f"{function}_{symbol}.json"
            out.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"wrote {out.name}", file=sys.stderr)
            # Free tier allows 5 requests/minute.
            time.sleep(13)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1:]))
