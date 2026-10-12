"""Cross-checks guidance against what it internally implies and what the
company later reported.

The tricky part is mapping an AlphaVantage `fiscalDateEnding` (a plain
calendar date) onto the fiscal year/quarter label a company's own guidance
uses. A fiscal year is named for the calendar year it closes in, so NVDA's
fiscal 2026 ran February 2025 through January 2026 and the quarter ending
2026-07-31 already belongs to fiscal 2027. Comparing by calendar date alone
would silently line guidance up against a quarter a year away, which is worse
than not comparing at all. `fiscal_label_for_date` does that mapping and is
the one piece of this module that earns its own tests.
"""

from datetime import date
from itertools import combinations

from egs.config import get_settings
from egs.crosscheck.alphavantage import (
    AlphaVantageClient,
    AlphaVantageError,
    AlphaVantageUnavailable,
)
from egs.schemas import (
    Basis,
    FiscalPeriod,
    GuidanceItem,
    Metric,
    Mismatch,
    MismatchKind,
    PeriodKind,
    Unit,
)
from egs.valuation.multiples import to_canonical

_MARGIN_METRICS = {Metric.GROSS_MARGIN, Metric.OPERATING_MARGIN}


def fiscal_year_end_month(earnings_payload: dict) -> int:
    """Infer a company's fiscal year-end month from its most recent annual report.

    AlphaVantage doesn't expose this directly -- it has to be read off
    `annualEarnings[*]["fiscalDateEnding"]`. Needed before any
    `fiscalDateEnding` can be translated into a fiscal year/quarter label
    (see `fiscal_label_for_date`).
    """
    annual = earnings_payload.get("annualEarnings") or []
    if not annual:
        raise ValueError(
            "earnings payload has no annualEarnings to infer a fiscal year end from"
        )
    latest = max(date.fromisoformat(a["fiscalDateEnding"]) for a in annual)
    return latest.month


def fiscal_label_for_date(end_date: date, fye_month: int) -> tuple[int, int]:
    """Map a fiscalDateEnding to (fiscal_year, fiscal_quarter).

    A company's own guidance and earnings calls use its fiscal calendar, not
    the calendar date a report happens to land on.

    A fiscal year is named for the calendar year it *closes* in, so the fiscal
    year containing `end_date` is the one ending at the next occurrence of
    `fye_month` on or after that date. NVDA has a January year end, which makes
    it the useful test case: its fiscal 2026 ran Feb 2025 through Jan 2026, so
    the quarter ending 2026-01-31 is FY2026 Q4, and the quarter ending
    2026-07-31 is already FY2027 Q2. AlphaVantage agrees -- it reports
    fiscalDateEnding 2026-01-31 as the close of the fiscal year whose EPS is
    NVDA's reported FY2026 figure.

    Getting this backwards is the whole reason this helper exists: it would
    silently compare guidance against the actuals of a quarter a year away.

    December year ends need no special case -- the month is never greater than
    12, so the fiscal year is always that calendar year.

    The quarter is a count of three-month blocks starting the month after
    fye_month, which puts the block containing fye_month itself at Q4.
    """
    fiscal_year = end_date.year if end_date.month <= fye_month else end_date.year + 1

    offset = (end_date.month - fye_month - 1) % 12  # 0-11, position in the fiscal year
    fiscal_quarter = offset // 3 + 1
    return fiscal_year, fiscal_quarter


def _parse_eps(raw: object) -> float | None:
    if raw in (None, "", "None"):
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _reported_eps_for_period(
    earnings: dict, period: FiscalPeriod, fye_month: int
) -> float | None:
    if period.kind is PeriodKind.QUARTER:
        rows = earnings.get("quarterlyEarnings", [])
    elif period.kind is PeriodKind.FISCAL_YEAR:
        rows = earnings.get("annualEarnings", [])
    else:
        # HALF and MULTI_YEAR guidance don't map onto a single EARNINGS row.
        return None

    for row in rows:
        raw_date = row.get("fiscalDateEnding")
        if not raw_date:
            continue
        fy, fq = fiscal_label_for_date(date.fromisoformat(raw_date), fye_month)
        if fy != period.fiscal_year:
            continue
        if period.kind is PeriodKind.QUARTER and fq != period.fiscal_quarter:
            continue
        return _parse_eps(row.get("reportedEPS"))
    return None


def _check_internal(item: GuidanceItem, *, ticker: str) -> list[Mismatch]:
    """Flag impossibilities a schema validator alone doesn't catch.

    `GuidanceItem` already rejects low > high at construction, but the
    check stays here too as a defense-in-depth net -- cheap, and it covers
    an item that reached this function by some other path than the normal
    validated constructor.
    """
    problems: list[str] = []

    if item.low is not None and item.high is not None and item.low > item.high:
        problems.append(f"low {item.low} exceeds high {item.high}")

    if item.metric is Metric.EPS and item.unit is not Unit.USD_PER_SHARE:
        problems.append(f"eps guidance given in {item.unit.value}, expected usd_per_share")

    if item.metric is Metric.REVENUE and item.unit is Unit.USD_PER_SHARE:
        problems.append("revenue guidance given per-share, which isn't a dollar amount")

    if item.metric is Metric.SEGMENT_REVENUE and not (item.metric_detail or "").strip():
        # Without a segment name the figure can't be attributed, compared
        # against anything, or told apart from another segment's guidance.
        problems.append("segment_revenue guidance names no segment in metric_detail")

    if item.metric in _MARGIN_METRICS:
        if item.unit is not Unit.PERCENT:
            problems.append(f"{item.metric.value} given in {item.unit.value}, expected percent")
        else:
            for bound in (item.low, item.high):
                if bound is not None and not (0.0 <= bound <= 100.0):
                    problems.append(
                        f"{item.metric.value} of {bound} is outside the 0-100 percent range"
                    )

    kind = item.period.kind
    fiscal_quarter = item.period.fiscal_quarter
    if kind in (PeriodKind.QUARTER, PeriodKind.HALF) and fiscal_quarter is None:
        problems.append(f"{kind.value} period is missing fiscal_quarter")
    if kind in (PeriodKind.FISCAL_YEAR, PeriodKind.MULTI_YEAR) and fiscal_quarter is not None:
        problems.append(f"{kind.value} period has fiscal_quarter set to {fiscal_quarter}")

    return [
        Mismatch(
            kind=MismatchKind.INTERNAL,
            ticker=ticker,
            metric=item.metric,
            period_label=item.period.label(),
            guided_value=item.midpoint(),
            comparison_value=None,
            relative_error=None,
            detail=problem,
            quote=item.quote,
        )
        for problem in problems
    ]


def _check_vs_reported(
    items: list[GuidanceItem], *, client: AlphaVantageClient | None, ticker: str
) -> list[Mismatch]:
    if client is None:
        return []
    eps_items = [
        item
        for item in items
        if item.metric is Metric.EPS
        and item.unit is Unit.USD_PER_SHARE
        and item.period.kind in (PeriodKind.QUARTER, PeriodKind.FISCAL_YEAR)
    ]
    if not eps_items:
        return []

    try:
        earnings = client.earnings(ticker)
        fye_month = fiscal_year_end_month(earnings)
    except (AlphaVantageError, AlphaVantageUnavailable, ValueError):
        # No EARNINGS data to compare against -- skip rather than guess.
        return []

    tolerance = get_settings().reconcile_tolerance
    mismatches: list[Mismatch] = []
    for item in eps_items:
        actual = _reported_eps_for_period(earnings, item.period, fye_month)
        if actual is None or actual == 0:
            continue  # not reported yet, or no usable figure on file
        guided = item.midpoint()
        relative_error = (guided - actual) / actual
        if abs(relative_error) <= tolerance:
            continue
        mismatches.append(
            Mismatch(
                kind=MismatchKind.VS_REPORTED,
                ticker=ticker,
                metric=Metric.EPS,
                period_label=item.period.label(),
                guided_value=guided,
                comparison_value=actual,
                relative_error=relative_error,
                detail=(
                    f"guided EPS midpoint {guided} vs reported {actual} for "
                    f"{item.period.label()} ({relative_error:+.1%})"
                ),
                quote=item.quote,
            )
        )
    return mismatches


def _canonical_bounds(item: GuidanceItem) -> tuple[float, float]:
    lo = to_canonical(item.low, item.unit) if item.low is not None else float("-inf")
    hi = to_canonical(item.high, item.unit) if item.high is not None else float("inf")
    return lo, hi


def _ranges_overlap(a: GuidanceItem, b: GuidanceItem) -> bool:
    a_lo, a_hi = _canonical_bounds(a)
    b_lo, b_hi = _canonical_bounds(b)
    return a_lo <= b_hi and b_lo <= a_hi


def _detail_key(metric_detail: str | None) -> str:
    return (metric_detail or "").strip().casefold()


def _check_duplicates(items: list[GuidanceItem], *, ticker: str) -> list[Mismatch]:
    """Two readings of the same figure that can't both be right.

    `metric_detail` is part of the key, which matters for segments: Microsoft
    guides Productivity, Intelligent Cloud and More Personal Computing for the
    same period with ranges that don't overlap, and those are three different
    figures, not three readings of one. Keying without the detail flagged two
    false conflicts on that call.
    """
    groups: dict[tuple[Metric, str, str, Basis], list[GuidanceItem]] = {}
    for item in items:
        key = (item.metric, _detail_key(item.metric_detail), item.period.label(), item.basis)
        groups.setdefault(key, []).append(item)

    mismatches: list[Mismatch] = []
    for (metric, _detail, period_label, basis), group in groups.items():
        if len(group) < 2:
            continue
        for a, b in combinations(group, 2):
            # Values are compared in canonical units -- a and b agreeing in
            # substance but one being given in usd_millions and the other
            # in usd_billions must not read as a conflict.
            if _ranges_overlap(a, b):
                continue
            mismatches.append(
                Mismatch(
                    kind=MismatchKind.DUPLICATE_CONFLICT,
                    ticker=ticker,
                    metric=metric,
                    period_label=period_label,
                    guided_value=a.midpoint(),
                    comparison_value=b.midpoint(),
                    relative_error=None,
                    detail=(
                        f"{metric.value} {period_label} ({basis.value}): "
                        f"[{a.low}, {a.high}] {a.unit.value} vs "
                        f"[{b.low}, {b.high}] {b.unit.value} don't overlap -- "
                        f'"{a.quote}" vs "{b.quote}"'
                    ),
                    quote=None,
                )
            )
    return mismatches


_DOLLAR_UNITS = {
    Unit.USD,
    Unit.USD_THOUSANDS,
    Unit.USD_MILLIONS,
    Unit.USD_BILLIONS,
}


def _check_segment_sums(items: list[GuidanceItem], *, ticker: str) -> list[Mismatch]:
    """Segment revenue guidance that can't fit inside total revenue guidance.

    Reportable segments are disjoint parts of total revenue, so they may sum to
    less than the total -- a company needn't guide every segment, and there is
    usually an "other" line -- but never to more.

    The test deliberately uses the sum of segment *lows* against the *highest*
    total, the most favorable reading available, so it only fires when no
    combination of values inside the guided ranges could be consistent. A
    missing segment low counts as zero and a missing total high makes the total
    unbounded; both weaken the check rather than making it wrong. Periods with
    no total revenue guidance are skipped silently -- there is nothing to
    compare against, and a false mismatch is worse than a missed one.
    """
    segments: dict[str, list[GuidanceItem]] = {}
    totals: dict[str, list[GuidanceItem]] = {}
    for item in items:
        if item.unit not in _DOLLAR_UNITS:
            continue
        label = item.period.label()
        if item.metric is Metric.SEGMENT_REVENUE:
            segments.setdefault(label, []).append(item)
        elif item.metric is Metric.REVENUE:
            totals.setdefault(label, []).append(item)

    mismatches: list[Mismatch] = []
    for label, group in segments.items():
        total_items = totals.get(label)
        if not total_items:
            continue
        highs = [
            to_canonical(t.high, t.unit) for t in total_items if t.high is not None
        ]
        if len(highs) != len(total_items):
            # At least one total is open-ended ("at least $X"), so no sum of
            # segments can be shown to exceed it.
            continue
        total_high = max(highs)

        segment_low_sum = sum(
            to_canonical(item.low, item.unit) for item in group if item.low is not None
        )
        if segment_low_sum <= total_high:
            continue

        named = ", ".join(sorted((i.metric_detail or "?").strip() for i in group))
        mismatches.append(
            Mismatch(
                kind=MismatchKind.INTERNAL,
                ticker=ticker,
                metric=Metric.SEGMENT_REVENUE,
                period_label=label,
                guided_value=segment_low_sum,
                comparison_value=total_high,
                relative_error=(segment_low_sum - total_high) / total_high
                if total_high
                else None,
                detail=(
                    f"segment revenue guidance for {label} sums to at least "
                    f"{segment_low_sum:,.0f} across {len(group)} segments "
                    f"({named}), above the highest guided total revenue of "
                    f"{total_high:,.0f}"
                ),
                quote=None,
            )
        )
    return mismatches


def reconcile(
    items: list[GuidanceItem], *, client: AlphaVantageClient | None, ticker: str
) -> list[Mismatch]:
    """Run every check and return all the mismatches found.

    Each check is independent and skips what it can't verify rather than
    guessing -- a false mismatch is worse than a missed one.

    `client` may be None. Only the comparison against reported actuals needs
    AlphaVantage; the internal, duplicate and segment-sum checks are pure
    arithmetic on the items. Without this, a setup with no API key got no
    consistency checking at all rather than the three quarters of it that work
    offline.
    """
    mismatches: list[Mismatch] = []
    for item in items:
        mismatches.extend(_check_internal(item, ticker=ticker))
    mismatches.extend(_check_vs_reported(items, client=client, ticker=ticker))
    mismatches.extend(_check_duplicates(items, ticker=ticker))
    mismatches.extend(_check_segment_sums(items, ticker=ticker))
    return mismatches
