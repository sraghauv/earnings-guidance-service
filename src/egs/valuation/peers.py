"""Static sector/peer map for valuation.

Sector and industry classification lives behind AlphaVantage's
COMPANY_OVERVIEW endpoint, which is premium-gated on a free key (see
crosscheck/alphavantage.py), so peer sets can't be derived at runtime. This
is a committed lookup table instead: a handful of mega-cap US sectors, a
few names each -- enough to run a peer-multiple valuation on the tickers
this project is actually exercised against, not a general sector taxonomy.
"""

import logging
from statistics import median

from egs.crosscheck.alphavantage import (
    AlphaVantageClient,
    AlphaVantageError,
    AlphaVantageUnavailable,
    quote_price,
    trailing_eps,
)
from egs.schemas import PeerMultiples

logger = logging.getLogger(__name__)

# (sector, industry, tickers). Each ticker appears in at most one group --
# this is a flat lookup table, not a classification hierarchy.
_SECTOR_GROUPS: list[tuple[str, str | None, list[str]]] = [
    ("Technology", "Semiconductors", ["NVDA", "AMD", "AVGO", "INTC", "QCOM"]),
    ("Technology", "Software", ["MSFT", "ORCL", "CRM", "ADBE", "NOW"]),
    ("Technology", "Consumer Electronics", ["AAPL", "SONY", "DELL", "HPQ"]),
    ("Communication Services", "Internet Content", ["GOOGL", "META", "SNAP", "PINS"]),
    ("Consumer Discretionary", "Internet Retail", ["AMZN", "WMT", "COST", "TGT"]),
    ("Consumer Discretionary", "Auto Manufacturers", ["TSLA", "GM", "F", "RIVN"]),
    ("Financial Services", "Banks", ["JPM", "BAC", "WFC", "C"]),
    ("Healthcare", "Drug Manufacturers", ["LLY", "JNJ", "PFE", "MRK"]),
    ("Energy", "Oil & Gas Integrated", ["XOM", "CVX", "COP"]),
    ("Industrials", "Aerospace & Defense", ["BA", "LMT", "RTX", "GE"]),
]

_TICKER_TO_GROUP: dict[str, tuple[str, str | None, list[str]]] = {
    ticker: group for group in _SECTOR_GROUPS for ticker in group[2]
}


def peers_for(ticker: str) -> tuple[str, str | None, list[str]]:
    """Sector, industry, and peer tickers (excludes `ticker` itself).

    A ticker not in the map comes back as ("Unknown", None, []) -- callers
    have to handle an empty peer list, since there's no sector lookup to
    fall back to without a premium AlphaVantage key.
    """
    ticker = ticker.upper()
    group = _TICKER_TO_GROUP.get(ticker)
    if group is None:
        return "Unknown", None, []
    sector, industry, tickers = group
    return sector, industry, [t for t in tickers if t != ticker]


def _parse_ratio(raw: object) -> float | None:
    if raw in (None, "", "None", "-"):
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def peer_multiples(ticker: str, *, client: AlphaVantageClient) -> PeerMultiples:
    """Median trailing multiples across the static peer set.

    Median, not mean: one peer sitting at a blown-out multiple shouldn't
    swing the comparison the way it would an average, and a peer with
    missing or non-positive trailing EPS is skipped rather than distorting
    the P/E the same way.

    P/S and EV/EBITDA come straight off COMPANY_OVERVIEW's own ratio
    fields, which is premium-gated on a free key -- that failure is caught
    per peer and logged, and the corresponding median comes back None
    rather than a number computed from whatever peers happened to answer.
    """
    sector, industry, peers = peers_for(ticker)

    pe_values: list[float] = []
    ps_values: list[float] = []
    ev_ebitda_values: list[float] = []

    for peer in peers:
        try:
            eps = trailing_eps(client.earnings(peer))
            price = quote_price(client.global_quote(peer))
        except (AlphaVantageError, AlphaVantageUnavailable) as exc:
            logger.info("skipping %s for P/E: %s", peer, exc)
        else:
            if eps is None or price is None:
                logger.info("skipping %s for P/E: missing EPS or price", peer)
            elif eps <= 0:
                logger.info("skipping %s for P/E: non-positive trailing EPS %s", peer, eps)
            else:
                pe_values.append(price / eps)

        try:
            overview = client.company_overview(peer)
        except (AlphaVantageError, AlphaVantageUnavailable) as exc:
            # The expected path on a free key, not a bug -- logged so the
            # degradation is visible instead of the medians just quietly
            # excluding this peer.
            logger.info("no COMPANY_OVERVIEW for %s, skipping P/S and EV/EBITDA: %s", peer, exc)
            continue

        ps = _parse_ratio(overview.get("PriceToSalesRatioTTM"))
        if ps is not None and ps > 0:
            ps_values.append(ps)
        ev_ebitda = _parse_ratio(overview.get("EVToEBITDA"))
        if ev_ebitda is not None and ev_ebitda > 0:
            ev_ebitda_values.append(ev_ebitda)

    return PeerMultiples(
        sector=sector,
        industry=industry,
        peer_tickers=peers,
        median_pe=median(pe_values) if pe_values else None,
        median_ps=median(ps_values) if ps_values else None,
        median_ev_ebitda=median(ev_ebitda_values) if ev_ebitda_values else None,
    )
