"""Tests for valuation/peers.py and valuation/multiples.py."""

import json
import statistics
from unittest.mock import patch

import pytest

from egs.crosscheck.alphavantage import FixtureAlphaVantageClient, trailing_eps
from egs.schemas import (
    Basis,
    FiscalPeriod,
    GuidanceItem,
    Metric,
    PeerMultiples,
    PeriodKind,
    Unit,
)
from egs.valuation import peers as peers_module
from egs.valuation.multiples import to_canonical, value_from_guidance
from egs.valuation.peers import peer_multiples, peers_for


@pytest.fixture
def av_client(fixtures_dir):
    return FixtureAlphaVantageClient(fixtures_dir / "alphavantage")


def _eps_item(
    kind=PeriodKind.QUARTER,
    fiscal_year=2026,
    fiscal_quarter=1,
    low=2.0,
    high=2.0,
    unit=Unit.USD_PER_SHARE,
):
    return GuidanceItem(
        metric=Metric.EPS,
        metric_detail=None,
        period=FiscalPeriod(kind=kind, fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter),
        low=low,
        high=high,
        unit=unit,
        basis=Basis.NON_GAAP,
        quote="we expect eps of...",
        speaker=None,
        confidence=0.9,
    )


def _revenue_item(low=30.0, high=31.0, unit=Unit.USD_BILLIONS):
    return GuidanceItem(
        metric=Metric.REVENUE,
        metric_detail=None,
        period=FiscalPeriod(kind=PeriodKind.QUARTER, fiscal_year=2026, fiscal_quarter=1),
        low=low,
        high=high,
        unit=unit,
        basis=Basis.NON_GAAP,
        quote="we expect revenue of...",
        speaker=None,
        confidence=0.9,
    )


# ---- to_canonical ----


def test_to_canonical_scales_billions_to_usd():
    assert to_canonical(1.5, Unit.USD_BILLIONS) == pytest.approx(1.5e9)


def test_to_canonical_scales_millions_to_usd():
    assert to_canonical(500.0, Unit.USD_MILLIONS) == pytest.approx(500e6)


def test_to_canonical_billions_and_millions_are_comparable():
    assert to_canonical(1.0, Unit.USD_BILLIONS) == to_canonical(1000.0, Unit.USD_MILLIONS)


def test_to_canonical_passes_through_non_dollar_units():
    assert to_canonical(75.0, Unit.PERCENT) == 75.0
    assert to_canonical(12.5, Unit.USD_PER_SHARE) == 12.5
    assert to_canonical(0.3, Unit.RATIO) == 0.3


# ---- peers.peers_for ----


def test_peers_for_unknown_ticker_returns_empty_list():
    sector, industry, peer_tickers = peers_for("ZZZZ")
    assert sector == "Unknown"
    assert industry is None
    assert peer_tickers == []


def test_peers_for_excludes_the_ticker_itself():
    _, _, peer_tickers = peers_for("amd")
    assert "AMD" not in peer_tickers
    assert "NVDA" in peer_tickers


# ---- peers.peer_multiples ----


def test_peer_multiples_uses_real_committed_peer_fixtures(av_client):
    """AMD's committed peer group is NVDA/AVGO/INTC/QCOM. QCOM has no
    recorded fixture at all, which should be skipped rather than crash."""
    pm = peer_multiples("AMD", client=av_client)
    assert pm.peer_tickers == ["NVDA", "AVGO", "INTC", "QCOM"]

    fx = av_client.fixtures_dir
    expected = statistics.median(
        [
            230.48 / trailing_eps(json.loads((fx / "EARNINGS_NVDA.json").read_text())),
            388.12 / trailing_eps(json.loads((fx / "EARNINGS_AVGO.json").read_text())),
            38.67 / trailing_eps(json.loads((fx / "EARNINGS_INTC.json").read_text())),
        ]
    )
    assert pm.median_pe == pytest.approx(expected)
    # COMPANY_OVERVIEW isn't recorded for any of these peers on a free key.
    assert pm.median_ps is None
    assert pm.median_ev_ebitda is None


def test_peer_multiples_median_skips_a_negative_eps_peer(av_client):
    """NEGPEER is a hand-written fixture with four negative quarterly EPS
    figures -- it must be excluded from the median entirely, not just
    down-weighted, or one bad peer corrupts the whole comparison."""
    with patch.object(
        peers_module, "peers_for", lambda t: ("Test", None, ["AMD", "AVGO", "NEGPEER"])
    ):
        pm = peer_multiples("TESTCO", client=av_client)

    expected = statistics.median([212.44 / 4.36, 388.12 / 6.49])
    assert pm.median_pe == pytest.approx(expected)


def test_peer_multiples_empty_peer_list_yields_no_medians(av_client):
    pm = peer_multiples("ZZZZ", client=av_client)
    assert pm.peer_tickers == []
    assert pm.median_pe is None
    assert pm.median_ps is None
    assert pm.median_ev_ebitda is None


# ---- multiples.value_from_guidance ----


def test_value_from_guidance_eps_quarter_is_annualized_before_applying_pe(av_client):
    item = _eps_item(kind=PeriodKind.QUARTER, low=2.0, high=2.0)
    valuation = value_from_guidance(item, ticker="AMD", client=av_client)

    assert valuation is not None
    assert valuation.multiple_kind == "pe"
    assert valuation.guided_midpoint == pytest.approx(2.0)
    # Quarter guidance is run-rated x4 before the peer P/E is applied.
    assert valuation.implied_price_per_share == pytest.approx(
        2.0 * 4 * valuation.multiple_used
    )
    assert valuation.current_price == pytest.approx(212.44)
    assert valuation.upside_pct == pytest.approx(
        (valuation.implied_price_per_share - valuation.current_price)
        / valuation.current_price
    )


def test_value_from_guidance_eps_fiscal_year_is_not_annualized(av_client):
    item = _eps_item(kind=PeriodKind.FISCAL_YEAR, fiscal_quarter=None, low=8.0, high=8.0)
    valuation = value_from_guidance(item, ticker="AMD", client=av_client)

    assert valuation is not None
    assert valuation.implied_price_per_share == pytest.approx(8.0 * valuation.multiple_used)


def test_value_from_guidance_returns_none_for_non_per_share_eps_unit(av_client):
    item = _eps_item(unit=Unit.USD)
    assert value_from_guidance(item, ticker="AMD", client=av_client) is None


def test_value_from_guidance_eps_returns_none_when_peers_unknown(av_client):
    item = _eps_item()
    assert value_from_guidance(item, ticker="ZZZZ", client=av_client) is None


def test_value_from_guidance_multi_year_eps_returns_none(av_client):
    item = _eps_item(kind=PeriodKind.MULTI_YEAR, fiscal_quarter=None, low=10.0, high=10.0)
    assert value_from_guidance(item, ticker="AMD", client=av_client) is None


def test_value_from_guidance_returns_none_for_unsupported_metric(av_client):
    item = GuidanceItem(
        metric=Metric.OPEX,
        metric_detail=None,
        period=FiscalPeriod(kind=PeriodKind.QUARTER, fiscal_year=2026, fiscal_quarter=1),
        low=1.0,
        high=1.0,
        unit=Unit.USD_BILLIONS,
        basis=Basis.NON_GAAP,
        quote="opex guidance",
        speaker=None,
        confidence=0.9,
    )
    assert value_from_guidance(item, ticker="AMD", client=av_client) is None


def test_value_from_guidance_revenue_returns_none_without_shares_outstanding(av_client):
    item = _revenue_item()
    fake_peers = PeerMultiples(
        sector="Technology",
        industry="Semiconductors",
        peer_tickers=["NVDA"],
        median_pe=30.0,
        median_ps=5.0,
        median_ev_ebitda=None,
    )
    with patch("egs.valuation.multiples.peer_multiples", return_value=fake_peers):
        result = value_from_guidance(
            item, ticker="AMD", client=av_client, shares_outstanding=None
        )
    assert result is None


def test_value_from_guidance_revenue_computes_when_ps_and_shares_available(av_client):
    item = _revenue_item(low=30.0, high=31.0, unit=Unit.USD_BILLIONS)
    fake_peers = PeerMultiples(
        sector="Technology",
        industry="Semiconductors",
        peer_tickers=["NVDA"],
        median_pe=30.0,
        median_ps=5.0,
        median_ev_ebitda=None,
    )
    with patch("egs.valuation.multiples.peer_multiples", return_value=fake_peers):
        valuation = value_from_guidance(
            item, ticker="AMD", client=av_client, shares_outstanding=1_000_000_000
        )

    assert valuation is not None
    assert valuation.multiple_kind == "ps"
    expected_revenue = 30.5e9  # midpoint of 30-31 usd_billions, canonicalized
    assert valuation.guided_midpoint == pytest.approx(expected_revenue)
    assert valuation.implied_equity_value == pytest.approx(expected_revenue * 5.0)
    assert valuation.implied_price_per_share == pytest.approx(
        expected_revenue * 5.0 / 1_000_000_000
    )


def test_value_from_guidance_revenue_returns_none_when_ps_unavailable(av_client):
    """On a real free key there's no COMPANY_OVERVIEW fixture at all, so
    median_ps comes back None and this must degrade rather than guess."""
    item = _revenue_item()
    result = value_from_guidance(
        item, ticker="AMD", client=av_client, shares_outstanding=1_000_000_000
    )
    assert result is None
