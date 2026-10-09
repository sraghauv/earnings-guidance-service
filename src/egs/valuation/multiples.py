"""Implied price-per-share valuation: guided midpoint x peer median multiple.

Only EPS guidance clears the whole pipeline on a free AlphaVantage key.
Revenue guidance needs a peer median P/S (from COMPANY_OVERVIEW, premium
only -- see valuation/peers.py) plus a share count to turn an implied
equity value into a price, so `value_from_guidance` returns None rather
than a number whenever either is missing, and logs why.
"""

import logging

from egs.crosscheck.alphavantage import (
    AlphaVantageClient,
    AlphaVantageError,
    AlphaVantageUnavailable,
    quote_price,
)
from egs.schemas import GuidanceItem, Metric, PeriodKind, Unit, Valuation
from egs.valuation.peers import peer_multiples

logger = logging.getLogger(__name__)

_DOLLAR_SCALE = {
    Unit.USD: 1.0,
    Unit.USD_THOUSANDS: 1e3,
    Unit.USD_MILLIONS: 1e6,
    Unit.USD_BILLIONS: 1e9,
}

# A quarter or half of guided EPS is run-rated to an annual figure before an
# annual P/E is applied -- x4 for a quarter, x2 for a half, as-is for a full
# fiscal year. multi_year isn't in here on purpose: a multi-year EPS total
# isn't a per-year figure without knowing the span, so there's nothing
# sound to annualize it by.
_ANNUALIZE_FACTOR = {
    PeriodKind.QUARTER: 4,
    PeriodKind.HALF: 2,
    PeriodKind.FISCAL_YEAR: 1,
}


def to_canonical(value: float, unit: Unit) -> float:
    """Scale a dollar-denominated value to plain USD.

    usd_per_share, percent and ratio aren't on a dollar scale and pass
    through unchanged -- there's nothing to convert them to. Mixing one of
    those with a dollar figure is a unit mismatch for reconcile.py's
    INTERNAL checks to catch, not something this function can paper over.
    """
    scale = _DOLLAR_SCALE.get(unit)
    return value * scale if scale is not None else value


def value_from_guidance(
    item: GuidanceItem,
    *,
    ticker: str,
    client: AlphaVantageClient,
    shares_outstanding: float | None = None,
) -> Valuation | None:
    if item.metric is Metric.EPS:
        return _value_from_eps(item, ticker=ticker, client=client)
    if item.metric is Metric.REVENUE:
        return _value_from_revenue(
            item, ticker=ticker, client=client, shares_outstanding=shares_outstanding
        )
    return None


def _current_price(ticker: str, client: AlphaVantageClient) -> float | None:
    try:
        return quote_price(client.global_quote(ticker))
    except (AlphaVantageError, AlphaVantageUnavailable) as exc:
        logger.info("no current price for %s: %s", ticker, exc)
        return None


def _upside(implied_price: float, current_price: float | None) -> float | None:
    if current_price is None or current_price <= 0:
        return None
    return (implied_price - current_price) / current_price


def _value_from_eps(
    item: GuidanceItem, *, ticker: str, client: AlphaVantageClient
) -> Valuation | None:
    if item.unit is not Unit.USD_PER_SHARE:
        logger.info(
            "eps guidance for %s given in %s, not usd_per_share -- skipping valuation",
            ticker,
            item.unit.value,
        )
        return None

    factor = _ANNUALIZE_FACTOR.get(item.period.kind)
    if factor is None:
        return None

    midpoint = item.midpoint()
    annualized_midpoint = midpoint * factor

    peers = peer_multiples(ticker, client=client)
    if peers.median_pe is None:
        logger.info("no peer median P/E for %s -- skipping valuation", ticker)
        return None

    implied_price = annualized_midpoint * peers.median_pe
    current_price = _current_price(ticker, client)

    return Valuation(
        ticker=ticker,
        period_label=item.period.label(),
        basis_metric=Metric.EPS,
        guided_midpoint=midpoint,
        multiple_used=peers.median_pe,
        multiple_kind="pe",
        implied_equity_value=None,
        implied_price_per_share=implied_price,
        current_price=current_price,
        upside_pct=_upside(implied_price, current_price),
        peers=peers,
    )


def _value_from_revenue(
    item: GuidanceItem,
    *,
    ticker: str,
    client: AlphaVantageClient,
    shares_outstanding: float | None,
) -> Valuation | None:
    peers = peer_multiples(ticker, client=client)
    if peers.median_ps is None:
        # P/S comes from COMPANY_OVERVIEW, premium-gated on a free key (see
        # peers.py). No fabricated multiple -- just no valuation.
        logger.info("no peer median P/S for %s -- skipping revenue valuation", ticker)
        return None
    if shares_outstanding is None:
        logger.info("no shares_outstanding for %s -- skipping revenue valuation", ticker)
        return None

    revenue = to_canonical(item.midpoint(), item.unit)
    implied_equity_value = revenue * peers.median_ps
    implied_price = implied_equity_value / shares_outstanding
    current_price = _current_price(ticker, client)

    return Valuation(
        ticker=ticker,
        period_label=item.period.label(),
        basis_metric=Metric.REVENUE,
        guided_midpoint=revenue,
        multiple_used=peers.median_ps,
        multiple_kind="ps",
        implied_equity_value=implied_equity_value,
        implied_price_per_share=implied_price,
        current_price=current_price,
        upside_pct=_upside(implied_price, current_price),
        peers=peers,
    )
