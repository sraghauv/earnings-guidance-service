"""Tests for the AlphaVantage client: the disk cache, and the rate-limit /
premium-notice bodies that are the whole reason this client exists instead
of a bare httpx call.

No network: the live `AlphaVantageClient` is exercised with its `httpx.Client`
swapped for a stub, and `FixtureAlphaVantageClient` reads the committed
fixtures (including a couple of hand-written ones for the error-body paths,
since AlphaVantage never recorded one of those for us on a free key).
"""

import json

import pytest

from egs.crosscheck.alphavantage import (
    AlphaVantageClient,
    AlphaVantageError,
    AlphaVantageUnavailable,
    FixtureAlphaVantageClient,
    quote_price,
    trailing_eps,
)


class _StubResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _CountingGet:
    """Stands in for httpx.Client.get, counting calls and returning `payload`."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return _StubResponse(self.payload)


# ---- disk cache ----


def test_disk_cache_avoids_a_second_fetch(tmp_path):
    client = AlphaVantageClient(api_key="fake-key", cache_dir=tmp_path)
    stub = _CountingGet({"symbol": "FOO", "annualEarnings": [], "quarterlyEarnings": []})
    client._client.get = stub

    client.earnings("FOO")
    client.earnings("FOO")

    assert stub.calls == 1
    assert (tmp_path / "alphavantage" / "EARNINGS_FOO.json").exists()


def test_disk_cache_force_bypasses_the_cache(tmp_path):
    client = AlphaVantageClient(api_key="fake-key", cache_dir=tmp_path)
    stub = _CountingGet({"symbol": "FOO", "annualEarnings": [], "quarterlyEarnings": []})
    client._client.get = stub

    client.earnings("FOO")
    client.earnings("FOO", force=True)

    assert stub.calls == 2


def test_rate_limit_body_is_never_written_to_the_cache(tmp_path):
    client = AlphaVantageClient(api_key="fake-key", cache_dir=tmp_path)
    client._client.get = _CountingGet({"Note": "rate limit hit"})

    with pytest.raises(AlphaVantageUnavailable):
        client.earnings("BAR")

    # The whole point of checking before caching: a transient rate-limit
    # body must never get persisted and replayed as fact on a later call.
    assert not (tmp_path / "alphavantage" / "EARNINGS_BAR.json").exists()


def test_error_message_body_raises_alphavantage_error_not_unavailable(tmp_path):
    client = AlphaVantageClient(api_key="fake-key", cache_dir=tmp_path)
    client._client.get = _CountingGet({"Error Message": "invalid symbol"})

    with pytest.raises(AlphaVantageError):
        client.earnings("BADSYM")


def test_missing_api_key_raises_before_any_request(tmp_path):
    client = AlphaVantageClient(api_key="placeholder", cache_dir=tmp_path)
    client.api_key = None  # force the no-key path regardless of ambient settings
    stub = _CountingGet({"symbol": "FOO"})
    client._client.get = stub

    with pytest.raises(AlphaVantageError):
        client.earnings("FOO")
    assert stub.calls == 0


# ---- FixtureAlphaVantageClient ----


def test_fixture_client_reads_real_recorded_response(fixtures_dir):
    client = FixtureAlphaVantageClient(fixtures_dir / "alphavantage")
    payload = client.earnings("NVDA")
    assert payload["symbol"] == "NVDA"
    assert len(payload["quarterlyEarnings"]) == 6


def test_fixture_client_raises_on_rate_limit_body(fixtures_dir):
    client = FixtureAlphaVantageClient(fixtures_dir / "alphavantage")
    with pytest.raises(AlphaVantageUnavailable):
        client.earnings("RATELIMIT")


def test_fixture_client_raises_on_error_message_body(fixtures_dir):
    client = FixtureAlphaVantageClient(fixtures_dir / "alphavantage")
    with pytest.raises(AlphaVantageError):
        client.earnings("BADSYMBOL")


def test_fixture_client_premium_notice_on_company_overview(fixtures_dir):
    """COMPANY_OVERVIEW is the endpoint that's actually premium-gated on a
    free key -- this is the realistic shape that call gets back."""
    client = FixtureAlphaVantageClient(fixtures_dir / "alphavantage")
    with pytest.raises(AlphaVantageUnavailable):
        client.company_overview("NVDA")


def test_fixture_client_raises_on_missing_fixture(fixtures_dir):
    client = FixtureAlphaVantageClient(fixtures_dir / "alphavantage")
    with pytest.raises(AlphaVantageError):
        client.earnings("NOSUCHTICKER")


# ---- quote_price ----


def test_quote_price_parses_numbered_key(fixtures_dir):
    payload = json.loads(
        (fixtures_dir / "alphavantage" / "GLOBAL_QUOTE_NVDA.json").read_text()
    )
    assert quote_price(payload) == 230.48


def test_quote_price_returns_none_when_missing():
    assert quote_price({}) is None
    assert quote_price({"Global Quote": {}}) is None


# ---- trailing_eps ----


def test_trailing_eps_sums_four_most_recent_quarters(fixtures_dir):
    payload = json.loads(
        (fixtures_dir / "alphavantage" / "EARNINGS_NVDA.json").read_text()
    )
    # Top four by fiscalDateEnding: 2.22 + 1.87 + 1.62 + 1.30
    assert trailing_eps(payload) == pytest.approx(7.01)


def test_trailing_eps_respects_as_of_and_excludes_future_quarters(fixtures_dir):
    from datetime import date

    payload = json.loads(
        (fixtures_dir / "alphavantage" / "EARNINGS_NVDA.json").read_text()
    )
    # As of right after the Q4 FY2025 report (2026-02-25), the trailing four
    # quarters are exactly fiscal year 2025 -- which should equal the
    # reported annual EPS for the fiscal year ending 2026-01-31.
    result = trailing_eps(payload, as_of=date(2026, 3, 1))
    assert result == pytest.approx(4.78)


def test_trailing_eps_returns_none_with_fewer_than_four_quarters():
    payload = {
        "quarterlyEarnings": [
            {"fiscalDateEnding": "2026-07-31", "reportedDate": "2026-08-26", "reportedEPS": "1.0"},
            {"fiscalDateEnding": "2026-04-30", "reportedDate": "2026-05-20", "reportedEPS": "0.9"},
            {"fiscalDateEnding": "2026-01-31", "reportedDate": "2026-02-25", "reportedEPS": "0.8"},
        ]
    }
    assert trailing_eps(payload, as_of=None) is None


def test_trailing_eps_skips_quarters_with_no_reported_figure_yet():
    payload = {
        "quarterlyEarnings": [
            {"fiscalDateEnding": "2026-07-31", "reportedDate": "2026-08-26", "reportedEPS": "None"},
            {"fiscalDateEnding": "2026-04-30", "reportedDate": "2026-05-20", "reportedEPS": "1.0"},
            {"fiscalDateEnding": "2026-01-31", "reportedDate": "2026-02-25", "reportedEPS": "0.9"},
            {"fiscalDateEnding": "2025-10-31", "reportedDate": "2025-11-19", "reportedEPS": "0.8"},
            {"fiscalDateEnding": "2025-07-31", "reportedDate": "2025-08-27", "reportedEPS": "0.7"},
        ]
    }
    # The "None" quarter is skipped, so the trailing four come from the
    # remaining rows rather than including a bogus zero.
    assert trailing_eps(payload) == pytest.approx(1.0 + 0.9 + 0.8 + 0.7)
