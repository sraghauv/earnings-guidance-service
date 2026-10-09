"""Client for the AlphaVantage REST API, with a disk cache and a fixture-backed
twin for tests.

AlphaVantage answers with HTTP 200 even when it can't serve the request --
the free-tier rate limit and a premium-only endpoint both ride along in the
JSON body instead of an HTTP error code, under one of a few different keys
depending on which kind of refusal it is. Treating that body as data would
silently look like a company with no earnings or a stock worth nothing, so
every payload is checked for those keys before anything downstream (or the
disk cache) sees it.
"""

import json
from datetime import date
from pathlib import Path

import httpx

from egs.config import get_settings
from egs.llm import cache

BASE_URL = "https://www.alphavantage.co/query"

# Checked in this order: a rate limit or premium notice still comes back as
# HTTP 200, under "Note" (free-tier rate limit) or "Information" (premium
# endpoint on a free key). "Error Message" is AlphaVantage's generic bad
# request (unknown symbol, missing param, etc.) and isn't a capacity problem.
_UNAVAILABLE_KEYS = ("Note", "Information")
_ERROR_KEY = "Error Message"


class AlphaVantageError(RuntimeError):
    """AlphaVantage rejected the request for a reason other than capacity."""


class AlphaVantageUnavailable(RuntimeError):
    """Endpoint needs a premium key, or the free-tier rate limit was hit."""


def _raise_if_unavailable(payload: dict, *, function: str, symbol: str) -> None:
    for key in _UNAVAILABLE_KEYS:
        if key in payload:
            raise AlphaVantageUnavailable(f"{function} {symbol}: {payload[key]}")
    if _ERROR_KEY in payload:
        raise AlphaVantageError(f"{function} {symbol}: {payload[_ERROR_KEY]}")


def quote_price(quote_payload: dict) -> float | None:
    """Pull the price out of a GLOBAL_QUOTE response.

    AlphaVantage uses numbered, dotted field names ("05. price") instead of
    a plain key, so this is worth a named helper rather than inlining the
    lookup everywhere a price is needed.
    """
    raw = quote_payload.get("Global Quote", {}).get("05. price")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def trailing_eps(earnings_payload: dict, as_of: date | None = None) -> float | None:
    """Sum the four most recent quarterly reportedEPS as of `as_of` (default today).

    Returns None instead of a partial sum when fewer than four qualifying
    quarters exist -- three quarters of trailing EPS understates earnings
    and would quietly inflate any P/E built on top of it. Quarters are
    sorted by fiscalDateEnding here rather than trusting response order,
    since nothing guarantees AlphaVantage always returns newest first.
    """
    as_of = as_of or date.today()
    quarters = sorted(
        earnings_payload.get("quarterlyEarnings", []),
        key=lambda q: q.get("fiscalDateEnding", ""),
        reverse=True,
    )
    total = 0.0
    count = 0
    for q in quarters:
        reported = q.get("reportedDate")
        if reported and date.fromisoformat(reported) > as_of:
            continue
        raw_eps = q.get("reportedEPS")
        if raw_eps in (None, "", "None"):
            continue
        try:
            total += float(raw_eps)
        except ValueError:
            continue
        count += 1
        if count == 4:
            break
    return total if count == 4 else None


class AlphaVantageClient:
    """Fetches AlphaVantage endpoints, caching responses to disk indefinitely.

    Fundamentals change at most quarterly, so a cached response is reused
    forever unless `force=True` -- there's no freshness window to expire on.
    GLOBAL_QUOTE is the exception in principle (it's intraday) but gets the
    same treatment: this service only uses it as a snapshot to compute a
    multiple against, and a stale snapshot is an accepted tradeoff rather
    than a correctness bug.
    """

    def __init__(
        self, *, api_key: str | None = None, cache_dir: Path | None = None
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.alphavantage_api_key
        self.cache_dir = (cache_dir or settings.cache_dir) / "alphavantage"
        self._client = httpx.Client(timeout=30.0)

    def _get(self, function: str, symbol: str, *, force: bool = False) -> dict:
        key = f"{function}_{symbol}"
        if not force:
            cached = cache.get(self.cache_dir, key)
            if cached is not None:
                return cached

        if not self.api_key:
            raise AlphaVantageError("ALPHAVANTAGE_API_KEY is not set")

        resp = self._client.get(
            BASE_URL,
            params={"function": function, "symbol": symbol, "apikey": self.api_key},
        )
        resp.raise_for_status()
        payload = resp.json()
        # Check before caching: a rate-limit or premium-notice body must
        # never reach disk, or every later call would replay it as fact.
        _raise_if_unavailable(payload, function=function, symbol=symbol)
        cache.put(self.cache_dir, key, payload)
        return payload

    def earnings(self, symbol: str, *, force: bool = False) -> dict:
        return self._get("EARNINGS", symbol, force=force)

    def global_quote(self, symbol: str, *, force: bool = False) -> dict:
        return self._get("GLOBAL_QUOTE", symbol, force=force)

    def company_overview(self, symbol: str, *, force: bool = False) -> dict:
        return self._get("COMPANY_OVERVIEW", symbol, force=force)

    def income_statement(self, symbol: str, *, force: bool = False) -> dict:
        return self._get("INCOME_STATEMENT", symbol, force=force)


class FixtureAlphaVantageClient(AlphaVantageClient):
    """Replays recorded responses from a fixtures directory -- no network, no key.

    Files are named `{function}_{symbol}.json`, same convention the disk
    cache uses, so a real cached response and a fixture are interchangeable.
    Every payload still goes through `_raise_if_unavailable`, so a fixture
    deliberately recorded as a rate-limit or premium-notice body behaves
    exactly like the live client would.
    """

    def __init__(self, fixtures_dir: Path) -> None:
        self.fixtures_dir = Path(fixtures_dir)

    def _get(self, function: str, symbol: str, *, force: bool = False) -> dict:
        path = self.fixtures_dir / f"{function}_{symbol}.json"
        if not path.exists():
            raise AlphaVantageError(
                f"no fixture recorded for {function} {symbol} ({path})"
            )
        payload = json.loads(path.read_text())
        _raise_if_unavailable(payload, function=function, symbol=symbol)
        return payload
