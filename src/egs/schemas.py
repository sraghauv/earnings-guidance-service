"""Typed contract for everything that crosses a boundary in this service.

`GuidanceExtraction` doubles as the JSON schema handed to the model, so every
field here has to stay expressible in strict JSON Schema: no bare `Any`, no
unions beyond `X | None`, and every property required (nullable where the model
legitimately may not know the answer).
"""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Metric(StrEnum):
    REVENUE = "revenue"
    EPS = "eps"
    GROSS_MARGIN = "gross_margin"
    OPERATING_MARGIN = "operating_margin"
    OPERATING_INCOME = "operating_income"
    EBITDA = "ebitda"
    FREE_CASH_FLOW = "free_cash_flow"
    CAPEX = "capex"
    OPEX = "opex"
    TAX_RATE = "tax_rate"
    SEGMENT_REVENUE = "segment_revenue"
    OTHER = "other"


class Unit(StrEnum):
    USD = "usd"
    USD_THOUSANDS = "usd_thousands"
    USD_MILLIONS = "usd_millions"
    USD_BILLIONS = "usd_billions"
    USD_PER_SHARE = "usd_per_share"
    PERCENT = "percent"
    RATIO = "ratio"


class Basis(StrEnum):
    GAAP = "gaap"
    NON_GAAP = "non_gaap"
    UNSPECIFIED = "unspecified"


class PeriodKind(StrEnum):
    QUARTER = "quarter"
    FISCAL_YEAR = "fiscal_year"
    HALF = "half"
    MULTI_YEAR = "multi_year"


class FiscalPeriod(BaseModel):
    """The period guidance refers to, not the period being reported on."""

    kind: PeriodKind
    fiscal_year: int
    # Set for kind=quarter (1-4) and kind=half (1-2); null otherwise.
    fiscal_quarter: int | None

    @model_validator(mode="after")
    def _check_quarter(self) -> "FiscalPeriod":
        """A sub-annual period has to say which part of the year it is.

        Not pedantry -- `label()` falls through to "FY2016" for a quarter with
        no quarter number, which is the same label a legitimate full-year
        period produces. That collision propagates into the stored
        period_label, the duplicate-conflict key, the eval match key and the
        segment-sum grouping, so a quarter silently counts as a fiscal year
        everywhere. Rejecting it here means the lenient parser drops the item
        instead of persisting a figure under the wrong period.
        """
        needs_quarter = self.kind in (PeriodKind.QUARTER, PeriodKind.HALF)
        if needs_quarter and self.fiscal_quarter is None:
            raise ValueError(f"{self.kind.value} period needs a fiscal_quarter")
        if not needs_quarter and self.fiscal_quarter is not None:
            raise ValueError(
                f"{self.kind.value} period cannot have fiscal_quarter "
                f"{self.fiscal_quarter}"
            )
        if self.kind is PeriodKind.QUARTER and not 1 <= (self.fiscal_quarter or 0) <= 4:
            raise ValueError(f"fiscal_quarter {self.fiscal_quarter} is not 1-4")
        if self.kind is PeriodKind.HALF and not 1 <= (self.fiscal_quarter or 0) <= 2:
            raise ValueError(f"half {self.fiscal_quarter} is not 1-2")
        return self

    def label(self) -> str:
        if self.kind is PeriodKind.QUARTER and self.fiscal_quarter:
            return f"FY{self.fiscal_year}Q{self.fiscal_quarter}"
        if self.kind is PeriodKind.HALF and self.fiscal_quarter:
            return f"FY{self.fiscal_year}H{self.fiscal_quarter}"
        return f"FY{self.fiscal_year}"


class GuidanceItem(BaseModel):
    """A single forward-looking figure, with the words it came from.

    `low` and `high` are equal for a point estimate. Either may be null for a
    one-sided bound ("at least $2B" sets low only).
    """

    metric: Metric
    # Free text when metric is segment_revenue or other, e.g. "Data Center".
    metric_detail: str | None
    period: FiscalPeriod
    low: float | None
    high: float | None
    unit: Unit
    basis: Basis
    # Verbatim sentence the figure came from. Required — an item with no
    # supporting quote is treated as a hallucination and dropped.
    quote: str = Field(min_length=1)
    speaker: str | None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_bounds(self) -> "GuidanceItem":
        if self.low is None and self.high is None:
            raise ValueError("guidance item needs at least one of low/high")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"low {self.low} exceeds high {self.high}")
        return self

    def midpoint(self) -> float:
        if self.low is not None and self.high is not None:
            return (self.low + self.high) / 2
        return self.low if self.low is not None else self.high  # type: ignore[return-value]


class GuidanceExtraction(BaseModel):
    """Everything the model is asked to return for one transcript."""

    ticker: str
    # True when management explicitly declined to guide or gave none at all.
    # When true, `items` must be empty.
    no_guidance_given: bool
    items: list[GuidanceItem]
    extraction_notes: str | None

    @model_validator(mode="after")
    def _check_consistency(self) -> "GuidanceExtraction":
        if self.no_guidance_given and self.items:
            raise ValueError("no_guidance_given is set but items were returned")
        return self


class MismatchKind(StrEnum):
    # Guided figure disagrees with what the company later reported.
    VS_REPORTED = "vs_reported"
    # Guided revenue disagrees with the filed figure for a period already closed.
    VS_FILED = "vs_filed"
    # Internally impossible: inverted range, implausible unit, bad period.
    INTERNAL = "internal"
    # Two extractions for the same metric and period disagree.
    DUPLICATE_CONFLICT = "duplicate_conflict"


class Mismatch(BaseModel):
    kind: MismatchKind
    ticker: str
    metric: Metric
    period_label: str
    guided_value: float | None
    comparison_value: float | None
    relative_error: float | None
    detail: str
    quote: str | None


class Transcript(BaseModel):
    """One earnings call, as stored."""

    id: str
    ticker: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: int
    call_date: date
    prepared_remarks: str
    qa: str


class PeerMultiples(BaseModel):
    """Median trailing multiples across a peer set."""

    sector: str
    industry: str | None
    peer_tickers: list[str]
    median_pe: float | None
    median_ps: float | None
    median_ev_ebitda: float | None


class Valuation(BaseModel):
    """Implied per-share value from guidance midpoint x peer median multiple."""

    ticker: str
    period_label: str
    basis_metric: Metric
    guided_midpoint: float
    multiple_used: float
    multiple_kind: str
    implied_equity_value: float | None
    implied_price_per_share: float | None
    current_price: float | None
    upside_pct: float | None
    peers: PeerMultiples
