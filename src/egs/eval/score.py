"""Scores a `GuidanceExtraction` against a hand-labeled gold file.

Pure functions only -- no I/O beyond `load_gold`, no network. A gold file's
shape is documented in tests/fixtures/eval/nvda-2020q4.json; `labeled_from`
and `notes` keys are for humans reading the file and are ignored here.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from egs.schemas import Basis, GuidanceExtraction, Metric, Unit
from egs.valuation.multiples import to_canonical

MatchKey = tuple[Metric, str, Basis, str]


@dataclass(frozen=True)
class GoldItem:
    metric: Metric
    metric_detail: str | None
    basis: Basis
    period_label: str
    low: float | None
    high: float | None
    unit: Unit
    source: str


@dataclass(frozen=True)
class GoldSet:
    transcript_id: str
    ticker: str
    items: list[GoldItem]


def load_gold(path: Path) -> GoldSet:
    data = json.loads(Path(path).read_text())
    items = [
        GoldItem(
            metric=Metric(item["metric"]),
            metric_detail=item.get("metric_detail"),
            basis=Basis(item["basis"]),
            period_label=item["period_label"],
            low=item.get("low"),
            high=item.get("high"),
            unit=Unit(item["unit"]),
            source=item["source"],
        )
        for item in data["items"]
    ]
    return GoldSet(transcript_id=data["transcript_id"], ticker=data["ticker"], items=items)


def _detail_key(detail: str | None) -> str:
    return (detail or "").lower()


def match_key(
    metric: Metric, metric_detail: str | None, basis: Basis, period_label: str
) -> MatchKey:
    """What identifies a figure, independent of its value.

    metric_detail only counts for segment_revenue and other, where it is the
    only thing separating two figures -- "Data Center" revenue from "Gaming"
    revenue, or OI&E from any other miscellaneous line. Everywhere else it is
    an optional label, and models fill it with things like "q1 revenue
    guidance" that say nothing the metric doesn't already say. Keying on it
    there would score an otherwise exact match as a miss plus a false
    positive.

    metric_detail is lowercased and None/"" are treated as the same thing --
    the model is as likely to omit it as to echo it back verbatim.
    """
    if metric in (Metric.SEGMENT_REVENUE, Metric.OTHER):
        return (metric, _detail_key(metric_detail), basis, period_label)
    return (metric, "", basis, period_label)


def values_agree(
    gold_low: float | None,
    gold_high: float | None,
    gold_unit: Unit,
    got_low: float | None,
    got_high: float | None,
    got_unit: Unit,
    *,
    rel_tol: float = 0.01,
) -> bool:
    """Compare bounds on a common scale (see `to_canonical`).

    A bound that's set on one side and null on the other never agrees --
    that's a real disagreement about whether the figure is one-sided, not
    something unit conversion can paper over. Both sides null can't happen;
    GuidanceItem's validator already rejects that.
    """
    # The two noqa'd branches below stay nested on purpose: collapsing each
    # guard into its isclose call makes one condition too long to read.
    if (gold_low is None) != (got_low is None):
        return False
    if (gold_high is None) != (got_high is None):
        return False
    if gold_low is not None and got_low is not None:  # noqa: SIM102
        if not math.isclose(
            to_canonical(gold_low, gold_unit), to_canonical(got_low, got_unit), rel_tol=rel_tol
        ):
            return False
    if gold_high is not None and got_high is not None:  # noqa: SIM102
        if not math.isclose(
            to_canonical(gold_high, gold_unit), to_canonical(got_high, got_unit), rel_tol=rel_tol
        ):
            return False
    return True


def _format_range(low: float | None, high: float | None, unit: Unit) -> str:
    if low is not None and low == high:
        return f"{low} {unit.value}"
    lo = "None" if low is None else str(low)
    hi = "None" if high is None else str(high)
    return f"{lo}-{hi} {unit.value}"


@dataclass(frozen=True)
class ScoreRow:
    key: MatchKey
    gold_range: str
    got_range: str
    verdict: str  # one of: ok, wrong_value, missed, spurious


@dataclass(frozen=True)
class Score:
    matched: list[MatchKey]
    missed: list[MatchKey]
    spurious: list[MatchKey]
    value_correct: int
    value_wrong: int
    unit_correct: int
    rows: list[ScoreRow]

    @property
    def recall(self) -> float:
        total = len(self.matched) + len(self.missed)
        return len(self.matched) / total if total else 0.0

    @property
    def precision(self) -> float:
        total = len(self.matched) + len(self.spurious)
        return len(self.matched) / total if total else 0.0

    @property
    def value_accuracy(self) -> float:
        total = self.value_correct + self.value_wrong
        return self.value_correct / total if total else 0.0


def score(extraction: GuidanceExtraction, gold: GoldSet) -> Score:
    gold_by_key = {
        match_key(g.metric, g.metric_detail, g.basis, g.period_label): g for g in gold.items
    }
    got_by_key = {
        match_key(i.metric, i.metric_detail, i.basis, i.period.label()): i
        for i in extraction.items
    }

    matched = sorted(set(gold_by_key) & set(got_by_key))
    missed = sorted(set(gold_by_key) - set(got_by_key))
    spurious = sorted(set(got_by_key) - set(gold_by_key))

    value_correct = 0
    value_wrong = 0
    unit_correct = 0
    rows: list[ScoreRow] = []

    for key in matched:
        g, i = gold_by_key[key], got_by_key[key]
        agree = values_agree(g.low, g.high, g.unit, i.low, i.high, i.unit)
        if agree:
            value_correct += 1
        else:
            value_wrong += 1
        if i.unit == g.unit:
            unit_correct += 1
        rows.append(
            ScoreRow(
                key=key,
                gold_range=_format_range(g.low, g.high, g.unit),
                got_range=_format_range(i.low, i.high, i.unit),
                verdict="ok" if agree else "wrong_value",
            )
        )

    for key in missed:
        g = gold_by_key[key]
        rows.append(
            ScoreRow(
                key=key,
                gold_range=_format_range(g.low, g.high, g.unit),
                got_range="-",
                verdict="missed",
            )
        )

    for key in spurious:
        i = got_by_key[key]
        rows.append(
            ScoreRow(
                key=key,
                gold_range="-",
                got_range=_format_range(i.low, i.high, i.unit),
                verdict="spurious",
            )
        )

    return Score(
        matched=matched,
        missed=missed,
        spurious=spurious,
        value_correct=value_correct,
        value_wrong=value_wrong,
        unit_correct=unit_correct,
        rows=rows,
    )
