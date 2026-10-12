"""Tests for crosscheck/reconcile.py.

The fiscal-period mapping gets the most scrutiny here since it's the part
that's easy to get quietly wrong: NVDA has a January fiscal year end, so
its fiscal 2026 closed on 2026-01-31 and the quarter ending 2026-07-31 is
already fiscal 2027 Q2.
"""

import json
from datetime import date

import pytest

from egs.crosscheck.alphavantage import FixtureAlphaVantageClient
from egs.crosscheck.reconcile import (
    _check_duplicates,
    _check_segment_sums,
    fiscal_label_for_date,
    fiscal_year_end_month,
    reconcile,
)
from egs.schemas import Basis, FiscalPeriod, GuidanceItem, Metric, MismatchKind, PeriodKind, Unit


@pytest.fixture
def av_client(fixtures_dir):
    return FixtureAlphaVantageClient(fixtures_dir / "alphavantage")


@pytest.fixture
def nvda_earnings(fixtures_dir):
    return json.loads((fixtures_dir / "alphavantage" / "EARNINGS_NVDA.json").read_text())


def _item(
    metric=Metric.EPS,
    kind=PeriodKind.QUARTER,
    fiscal_year=2026,
    fiscal_quarter=1,
    low=1.0,
    high=1.0,
    unit=Unit.USD_PER_SHARE,
    basis=Basis.NON_GAAP,
    quote="guidance quote",
):
    return GuidanceItem(
        metric=metric,
        metric_detail=None,
        period=FiscalPeriod(kind=kind, fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter),
        low=low,
        high=high,
        unit=unit,
        basis=basis,
        quote=quote,
        speaker=None,
        confidence=0.9,
    )


# ---- fiscal_label_for_date: the part that gets tested hard ----


def test_fiscal_label_matches_nvda_fiscal_2027_q2():
    # A fiscal year is named for the year it closes, so a July 2026 quarter end
    # belongs to the fiscal year ending January 2027.
    assert fiscal_label_for_date(date(2026, 7, 31), fye_month=1) == (2027, 2)


def test_fiscal_label_sequence_across_all_nvda_quarters(nvda_earnings):
    fye = fiscal_year_end_month(nvda_earnings)
    labels = [
        fiscal_label_for_date(date.fromisoformat(q["fiscalDateEnding"]), fye)
        for q in nvda_earnings["quarterlyEarnings"]
    ]
    assert labels == [
        (2027, 2),
        (2027, 1),
        (2026, 4),
        (2026, 3),
        (2026, 2),
        (2026, 1),
    ]


def test_fiscal_label_annual_end_date_matches_q4_of_same_fiscal_year(nvda_earnings):
    """The annual fiscalDateEnding and the Q4 quarterly fiscalDateEnding for
    the same fiscal year are the same calendar date -- both must map to the
    same (fiscal_year, 4)."""
    fye = fiscal_year_end_month(nvda_earnings)
    annual_end = date.fromisoformat(nvda_earnings["annualEarnings"][0]["fiscalDateEnding"])
    q4_end = date.fromisoformat(nvda_earnings["quarterlyEarnings"][2]["fiscalDateEnding"])
    assert annual_end == q4_end
    assert fiscal_label_for_date(annual_end, fye) == (2026, 4)


def test_fiscal_label_calendar_year_end_has_no_year_rollover():
    """A December fiscal year end (the common case) shouldn't roll the
    label back a year the way NVDA's January year end does."""
    assert fiscal_label_for_date(date(2025, 3, 31), fye_month=12) == (2025, 1)
    assert fiscal_label_for_date(date(2025, 6, 30), fye_month=12) == (2025, 2)
    assert fiscal_label_for_date(date(2025, 12, 31), fye_month=12) == (2025, 4)


def test_fiscal_year_end_month_reads_most_recent_annual_report(nvda_earnings):
    assert fiscal_year_end_month(nvda_earnings) == 1


def test_fiscal_year_end_month_raises_without_annual_data():
    with pytest.raises(ValueError):
        fiscal_year_end_month({"quarterlyEarnings": []})


# ---- INTERNAL ----


def test_eps_guidance_in_wrong_unit_flagged_internal(av_client):
    item = _item(metric=Metric.EPS, unit=Unit.USD_BILLIONS)
    mismatches = reconcile([item], client=av_client, ticker="AMD")
    internal = [m for m in mismatches if m.kind is MismatchKind.INTERNAL]
    assert len(internal) == 1
    assert "usd_per_share" in internal[0].detail


def test_revenue_guided_per_share_flagged_internal(av_client):
    item = _item(metric=Metric.REVENUE, unit=Unit.USD_PER_SHARE, low=30.0, high=31.0)
    mismatches = reconcile([item], client=av_client, ticker="AMD")
    assert any(m.kind is MismatchKind.INTERNAL for m in mismatches)


def test_margin_outside_0_100_flagged_internal(av_client):
    item = _item(metric=Metric.GROSS_MARGIN, unit=Unit.PERCENT, low=-5.0, high=10.0)
    mismatches = reconcile([item], client=av_client, ticker="AMD")
    internal = [m for m in mismatches if m.kind is MismatchKind.INTERNAL]
    assert any("0-100" in m.detail for m in internal)


def test_margin_wrong_unit_flagged_internal(av_client):
    item = _item(metric=Metric.OPERATING_MARGIN, unit=Unit.RATIO, low=0.2, high=0.25)
    mismatches = reconcile([item], client=av_client, ticker="AMD")
    internal = [m for m in mismatches if m.kind is MismatchKind.INTERNAL]
    assert any("expected percent" in m.detail for m in internal)


def test_quarter_period_missing_fiscal_quarter_flagged_internal(av_client):
    item = _item(kind=PeriodKind.QUARTER, fiscal_quarter=None)
    mismatches = reconcile([item], client=av_client, ticker="AMD")
    internal = [m for m in mismatches if m.kind is MismatchKind.INTERNAL]
    assert any("missing fiscal_quarter" in m.detail for m in internal)


def test_fiscal_year_period_with_fiscal_quarter_set_flagged_internal(av_client):
    item = _item(metric=Metric.REVENUE, kind=PeriodKind.FISCAL_YEAR, fiscal_quarter=2, low=1.0, high=2.0, unit=Unit.USD_BILLIONS)
    mismatches = reconcile([item], client=av_client, ticker="AMD")
    internal = [m for m in mismatches if m.kind is MismatchKind.INTERNAL]
    assert any("fiscal_quarter set to 2" in m.detail for m in internal)


def test_well_formed_item_produces_no_internal_mismatch(av_client):
    item = _item(
        metric=Metric.REVENUE,
        kind=PeriodKind.QUARTER,
        fiscal_year=2027,
        fiscal_quarter=1,
        low=30.0,
        high=31.0,
        unit=Unit.USD_BILLIONS,
    )
    mismatches = reconcile([item], client=av_client, ticker="AMD")
    assert mismatches == []


# ---- VS_REPORTED ----


def test_vs_reported_fires_at_ten_percent_error(av_client):
    # NVDA fiscal 2027 Q1 (quarter ending 2026-04-30) reported EPS 1.87.
    item = _item(fiscal_year=2027, fiscal_quarter=1, low=1.87 * 1.10, high=1.87 * 1.10)
    mismatches = reconcile([item], client=av_client, ticker="NVDA")
    vs_reported = [m for m in mismatches if m.kind is MismatchKind.VS_REPORTED]
    assert len(vs_reported) == 1
    assert vs_reported[0].comparison_value == pytest.approx(1.87)
    assert vs_reported[0].relative_error == pytest.approx(0.10, abs=1e-9)


def test_vs_reported_silent_at_one_percent_error(av_client):
    item = _item(fiscal_year=2027, fiscal_quarter=1, low=1.87 * 1.01, high=1.87 * 1.01)
    mismatches = reconcile([item], client=av_client, ticker="NVDA")
    assert not any(m.kind is MismatchKind.VS_REPORTED for m in mismatches)


def test_vs_reported_skipped_when_period_not_yet_reported(av_client):
    item = _item(fiscal_year=2030, fiscal_quarter=1, low=100.0, high=100.0)
    mismatches = reconcile([item], client=av_client, ticker="NVDA")
    assert not any(m.kind is MismatchKind.VS_REPORTED for m in mismatches)


def test_vs_reported_skipped_for_fiscal_year_not_yet_closed(av_client):
    # Fiscal 2027 closes on 2027-01-31, and no annual report with that
    # fiscalDateEnding exists yet, so there is nothing to compare against.
    item = _item(kind=PeriodKind.FISCAL_YEAR, fiscal_year=2027, fiscal_quarter=None, low=100.0, high=100.0)
    mismatches = reconcile([item], client=av_client, ticker="NVDA")
    assert not any(m.kind is MismatchKind.VS_REPORTED for m in mismatches)


def test_vs_reported_matches_fiscal_year_against_annual_reported_eps(av_client):
    # Fiscal 2026 closed with annual fiscalDateEnding 2026-01-31, reportedEPS 4.78.
    item = _item(kind=PeriodKind.FISCAL_YEAR, fiscal_year=2026, fiscal_quarter=None, low=6.0, high=6.0)
    mismatches = reconcile([item], client=av_client, ticker="NVDA")
    vs_reported = [m for m in mismatches if m.kind is MismatchKind.VS_REPORTED]
    assert len(vs_reported) == 1
    assert vs_reported[0].comparison_value == pytest.approx(4.78)


def test_vs_reported_ignores_eps_given_in_wrong_unit(av_client):
    """A wildly wrong EPS number in the wrong unit is flagged INTERNAL, not
    VS_REPORTED -- comparing it to reportedEPS would be a false positive,
    not a real finding."""
    item = _item(unit=Unit.USD, fiscal_year=2026, fiscal_quarter=1, low=100.0, high=100.0)
    mismatches = reconcile([item], client=av_client, ticker="NVDA")
    assert any(m.kind is MismatchKind.INTERNAL for m in mismatches)
    assert not any(m.kind is MismatchKind.VS_REPORTED for m in mismatches)


# ---- DUPLICATE_CONFLICT ----


def test_duplicate_conflict_flagged_for_nonoverlapping_ranges(av_client):
    a = _item(metric=Metric.REVENUE, kind=PeriodKind.QUARTER, fiscal_year=2027, fiscal_quarter=1, low=30.0, high=31.0, unit=Unit.USD_BILLIONS, quote="first extraction")
    b = _item(metric=Metric.REVENUE, kind=PeriodKind.QUARTER, fiscal_year=2027, fiscal_quarter=1, low=40.0, high=41.0, unit=Unit.USD_BILLIONS, quote="second extraction")
    mismatches = reconcile([a, b], client=av_client, ticker="NVDA")
    dupes = [m for m in mismatches if m.kind is MismatchKind.DUPLICATE_CONFLICT]
    assert len(dupes) == 1


def test_duplicate_conflict_silent_when_ranges_overlap(av_client):
    a = _item(metric=Metric.REVENUE, kind=PeriodKind.QUARTER, fiscal_year=2027, fiscal_quarter=1, low=30.0, high=32.0, unit=Unit.USD_BILLIONS)
    b = _item(metric=Metric.REVENUE, kind=PeriodKind.QUARTER, fiscal_year=2027, fiscal_quarter=1, low=31.0, high=33.0, unit=Unit.USD_BILLIONS)
    mismatches = reconcile([a, b], client=av_client, ticker="NVDA")
    assert not any(m.kind is MismatchKind.DUPLICATE_CONFLICT for m in mismatches)


def test_duplicate_conflict_normalizes_units_before_comparing(av_client):
    """30-31 usd_billions and 30500-30800 usd_millions describe overlapping
    ranges (30.5-30.8B is inside 30-31B) -- comparing the raw numbers
    without normalizing would wrongly call this a conflict."""
    a = _item(metric=Metric.REVENUE, kind=PeriodKind.QUARTER, fiscal_year=2027, fiscal_quarter=1, low=30.0, high=31.0, unit=Unit.USD_BILLIONS)
    b = _item(metric=Metric.REVENUE, kind=PeriodKind.QUARTER, fiscal_year=2027, fiscal_quarter=1, low=30500.0, high=30800.0, unit=Unit.USD_MILLIONS)
    mismatches = reconcile([a, b], client=av_client, ticker="NVDA")
    assert not any(m.kind is MismatchKind.DUPLICATE_CONFLICT for m in mismatches)


def test_duplicate_conflict_normalized_units_still_catches_a_real_conflict(av_client):
    a = _item(metric=Metric.REVENUE, kind=PeriodKind.QUARTER, fiscal_year=2027, fiscal_quarter=1, low=30.0, high=31.0, unit=Unit.USD_BILLIONS)
    b = _item(metric=Metric.REVENUE, kind=PeriodKind.QUARTER, fiscal_year=2027, fiscal_quarter=1, low=40000.0, high=41000.0, unit=Unit.USD_MILLIONS)
    mismatches = reconcile([a, b], client=av_client, ticker="NVDA")
    assert any(m.kind is MismatchKind.DUPLICATE_CONFLICT for m in mismatches)


# ---- segment guidance ----


def _seg(name, low, high, *, unit=Unit.USD_BILLIONS, fiscal_year=2021, fiscal_quarter=2):
    return _item(
        metric=Metric.SEGMENT_REVENUE,
        kind=PeriodKind.QUARTER,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        low=low,
        high=high,
        unit=unit,
        quote=f"in {name}, we expect revenue between ${low} billion and ${high} billion",
    ).model_copy(update={"metric_detail": name})


def _total(low, high, *, unit=Unit.USD_BILLIONS, fiscal_year=2021, fiscal_quarter=2):
    return _item(
        metric=Metric.REVENUE,
        kind=PeriodKind.QUARTER,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        low=low,
        high=high,
        unit=unit,
        quote="we expect total revenue",
    )


def test_real_segment_guidance_is_not_a_duplicate_conflict():
    """Microsoft's three FY2021Q2 segments are three figures, not three readings.

    Keying duplicates without metric_detail flagged two false conflicts here,
    since the segment ranges legitimately don't overlap each other.
    """
    items = [
        _seg("Productivity and Business Processes", 12.75, 13.0),
        _seg("Intelligent Cloud", 13.55, 13.8),
        _seg("More Personal Computing", 13.2, 13.6),
    ]
    assert _check_duplicates(items, ticker="MSFT") == []


def test_same_segment_guided_twice_without_overlap_is_a_conflict():
    items = [
        _seg("Intelligent Cloud", 13.55, 13.8),
        _seg("Intelligent Cloud", 20.0, 21.0),
    ]
    mismatches = _check_duplicates(items, ticker="MSFT")
    assert len(mismatches) == 1
    assert mismatches[0].kind is MismatchKind.DUPLICATE_CONFLICT


def test_segment_name_differing_only_by_case_still_conflicts():
    items = [_seg("Intelligent Cloud", 13.55, 13.8), _seg("intelligent cloud", 20.0, 21.0)]
    assert len(_check_duplicates(items, ticker="MSFT")) == 1


def test_segment_revenue_without_a_segment_name_flagged_internal(av_client):
    item = _seg("   ", 13.55, 13.8)
    mismatches = reconcile([item], client=av_client, ticker="MSFT")
    internal = [m for m in mismatches if m.kind is MismatchKind.INTERNAL]
    assert any("names no segment" in m.detail for m in internal)


def test_named_segment_is_not_flagged_for_a_missing_name(av_client):
    item = _seg("Intelligent Cloud", 13.55, 13.8)
    mismatches = reconcile([item], client=av_client, ticker="MSFT")
    assert not any("names no segment" in m.detail for m in mismatches)


def test_segments_summing_above_total_revenue_flagged():
    items = [
        _seg("Productivity and Business Processes", 12.75, 13.0),
        _seg("Intelligent Cloud", 13.55, 13.8),
        _seg("More Personal Computing", 13.2, 13.6),
        _total(20.0, 21.0),
    ]
    mismatches = _check_segment_sums(items, ticker="MSFT")
    assert len(mismatches) == 1
    assert mismatches[0].kind is MismatchKind.INTERNAL
    assert mismatches[0].metric is Metric.SEGMENT_REVENUE
    # Sum of lows is 39.5bn against a 21bn total.
    assert mismatches[0].guided_value == pytest.approx(39.5e9)
    assert mismatches[0].comparison_value == pytest.approx(21e9)


def test_segments_fitting_inside_total_revenue_is_silent():
    items = [
        _seg("Productivity and Business Processes", 12.75, 13.0),
        _seg("Intelligent Cloud", 13.55, 13.8),
        _total(40.0, 41.0),
    ]
    assert _check_segment_sums(items, ticker="MSFT") == []


def test_segment_sum_skipped_when_no_total_revenue_guided():
    """Nothing to compare against, so nothing is emitted."""
    items = [
        _seg("Productivity and Business Processes", 12.75, 13.0),
        _seg("Intelligent Cloud", 13.55, 13.8),
    ]
    assert _check_segment_sums(items, ticker="MSFT") == []


def test_segment_sum_skipped_when_total_is_open_ended():
    """"At least $20 billion" can't be exceeded by any finite sum."""
    items = [_seg("Intelligent Cloud", 30.0, 31.0), _total(20.0, None)]
    assert _check_segment_sums(items, ticker="MSFT") == []


def test_segment_sum_compares_across_units():
    """A segment in millions and a total in billions must not false-positive."""
    items = [
        _seg("Intelligent Cloud", 13550.0, 13800.0, unit=Unit.USD_MILLIONS),
        _total(40.0, 41.0),
    ]
    assert _check_segment_sums(items, ticker="MSFT") == []


def test_segment_sum_only_compares_within_the_same_period():
    items = [
        _seg("Intelligent Cloud", 30.0, 31.0, fiscal_quarter=2),
        _total(20.0, 21.0, fiscal_quarter=3),
    ]
    assert _check_segment_sums(items, ticker="MSFT") == []


def test_segment_missing_a_low_bound_counts_as_zero():
    """A one-sided segment weakens the sum rather than inventing a floor."""
    items = [
        _seg("Intelligent Cloud", None, 30.0),
        _seg("Productivity and Business Processes", 5.0, 6.0),
        _total(20.0, 21.0),
    ]
    assert _check_segment_sums(items, ticker="MSFT") == []


# ---- running without AlphaVantage ----


def test_offline_checks_still_run_without_a_client():
    """No API key costs only the comparison against reported actuals.

    build_db skipped reconciliation entirely when no client could be built,
    which silently lost the three checks that are pure arithmetic.
    """
    items = [
        _seg("   ", 13.55, 13.8),
        _seg("Intelligent Cloud", 30.0, 31.0),
        _total(20.0, 21.0),
    ]
    mismatches = reconcile(items, client=None, ticker="MSFT")

    kinds = {m.kind for m in mismatches}
    assert MismatchKind.INTERNAL in kinds
    assert any("names no segment" in m.detail for m in mismatches)
    assert any("above the highest guided total" in m.detail for m in mismatches)


def test_vs_reported_is_the_only_check_that_needs_a_client():
    item = _item(fiscal_year=2027, fiscal_quarter=1, low=1.87 * 1.10, high=1.87 * 1.10)
    assert [m.kind for m in reconcile([item], client=None, ticker="NVDA")] == []
