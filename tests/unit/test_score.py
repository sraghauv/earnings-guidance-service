"""Unit tests for egs.eval.score. No network, no parquet."""

from pathlib import Path

from egs.eval.score import GoldItem, GoldSet, load_gold, match_key, score, values_agree
from egs.schemas import (
    Basis,
    FiscalPeriod,
    GuidanceExtraction,
    GuidanceItem,
    Metric,
    PeriodKind,
    Unit,
)

Q1 = FiscalPeriod(kind=PeriodKind.QUARTER, fiscal_year=2021, fiscal_quarter=1)


def _item(**overrides) -> GuidanceItem:
    defaults = dict(
        metric=Metric.REVENUE,
        metric_detail=None,
        period=Q1,
        low=2.94,
        high=3.06,
        unit=Unit.USD_BILLIONS,
        basis=Basis.UNSPECIFIED,
        quote="We expect revenue to be $3 billion plus or minus 2%.",
        speaker=None,
        confidence=0.9,
    )
    defaults.update(overrides)
    return GuidanceItem(**defaults)


def _gold_item(**overrides) -> GoldItem:
    defaults = dict(
        metric=Metric.REVENUE,
        metric_detail=None,
        basis=Basis.UNSPECIFIED,
        period_label="FY2021Q1",
        low=2.94,
        high=3.06,
        unit=Unit.USD_BILLIONS,
        source="We expect revenue to be $3 billion plus or minus 2%.",
    )
    defaults.update(overrides)
    return GoldItem(**defaults)


def _extraction(*items: GuidanceItem) -> GuidanceExtraction:
    return GuidanceExtraction(
        ticker="NVDA",
        no_guidance_given=not items,
        items=list(items),
        extraction_notes=None,
    )


def _gold(*items: GoldItem) -> GoldSet:
    return GoldSet(transcript_id="NVDA-2020Q4", ticker="NVDA", items=list(items))


def test_perfect_extraction_scores_all_ones():
    gold = _gold(_gold_item())
    extraction = _extraction(_item())

    result = score(extraction, gold)

    assert result.recall == 1.0
    assert result.precision == 1.0
    assert result.value_accuracy == 1.0
    assert len(result.matched) == 1
    assert result.missed == []
    assert result.spurious == []


def test_missed_figure_lowers_recall_not_precision():
    gold = _gold(_gold_item(), _gold_item(metric=Metric.EPS, period_label="FY2021Q1", low=1.0, high=1.0, unit=Unit.USD_PER_SHARE))
    extraction = _extraction(_item())

    result = score(extraction, gold)

    assert result.recall == 0.5
    assert result.precision == 1.0
    assert len(result.missed) == 1
    assert result.missed[0][0] == Metric.EPS


def test_spurious_figure_lowers_precision_not_recall():
    gold = _gold(_gold_item())
    extraction = _extraction(
        _item(),
        _item(metric=Metric.CAPEX, low=150.0, high=170.0, unit=Unit.USD_MILLIONS, quote="Capex guidance"),
    )

    result = score(extraction, gold)

    assert result.recall == 1.0
    assert result.precision == 0.5
    assert len(result.spurious) == 1
    assert result.spurious[0][0] == Metric.CAPEX


def test_values_agree_across_units():
    assert values_agree(1.05, 1.05, Unit.USD_BILLIONS, 1050.0, 1050.0, Unit.USD_MILLIONS)


def test_values_agree_false_for_gemma_tax_rate_miss():
    # The real gemma3:4b error against tests/fixtures/eval/nvda-2020q4.json:
    # gold tax rate is 8.0-10.0, the model returned 8.9-9.9.
    assert not values_agree(8.0, 10.0, Unit.PERCENT, 8.9, 9.9, Unit.PERCENT)


def test_metric_detail_matches_case_insensitively():
    key_upper = match_key(Metric.OTHER, "OI&E", Basis.GAAP, "FY2021Q1")
    key_lower = match_key(Metric.OTHER, "oi&e", Basis.GAAP, "FY2021Q1")
    assert key_upper == key_lower


def test_none_and_empty_string_metric_detail_are_equal():
    key_none = match_key(Metric.OTHER, None, Basis.GAAP, "FY2021Q1")
    key_empty = match_key(Metric.OTHER, "", Basis.GAAP, "FY2021Q1")
    assert key_none == key_empty


def test_one_sided_bound_does_not_agree_with_two_sided_range():
    assert not values_agree(2.94, 3.06, Unit.USD_BILLIONS, 2.94, None, Unit.USD_BILLIONS)


def test_load_gold_reads_the_nvda_fixture():
    path = Path(__file__).parent.parent / "fixtures" / "eval" / "nvda-2020q4.json"
    gold = load_gold(path)

    assert gold.transcript_id == "NVDA-2020Q4"
    assert gold.ticker == "NVDA"
    assert len(gold.items) == 10
    tax_rates = [i for i in gold.items if i.metric is Metric.TAX_RATE]
    assert len(tax_rates) == 2
    assert all(i.low == 8.0 and i.high == 10.0 for i in tax_rates)
