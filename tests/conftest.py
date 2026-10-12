"""Shared fixtures for unit and integration tests."""

from datetime import date
from pathlib import Path

import pytest

from egs.db import connect
from egs.schemas import (
    Basis,
    FiscalPeriod,
    GuidanceItem,
    Metric,
    PeriodKind,
    Transcript,
    Unit,
)


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_db(tmp_path):
    """A connect()ed tmp SQLite database with the schema applied."""
    conn = connect(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def sample_transcript() -> Transcript:
    return Transcript(
        id="NVDA-2026Q2",
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        fiscal_year=2026,
        fiscal_quarter=2,
        call_date=date(2025, 8, 27),
        prepared_remarks=(
            "Thanks for joining. Q2 revenue was $30.0 billion, up sharply "
            "year over year. For Q3, we expect revenue of $32.0 billion to "
            "$33.0 billion."
        ),
        qa="We'll now begin the question-and-answer session.",
    )


@pytest.fixture
def sample_guidance_items() -> list[GuidanceItem]:
    """A handful of items covering real edge cases: a two-sided range, a
    one-sided bound, and a fiscal year that doesn't line up with the
    calendar year (NVIDIA's FY ends in January)."""
    q3 = FiscalPeriod(kind=PeriodKind.QUARTER, fiscal_year=2026, fiscal_quarter=3)
    full_year = FiscalPeriod(kind=PeriodKind.FISCAL_YEAR, fiscal_year=2026, fiscal_quarter=None)
    return [
        GuidanceItem(
            metric=Metric.REVENUE,
            metric_detail=None,
            period=q3,
            low=32.0,
            high=33.0,
            unit=Unit.USD_BILLIONS,
            basis=Basis.NON_GAAP,
            quote="For Q3, we expect revenue of $32.0 billion to $33.0 billion.",
            speaker="Colette Kress",
            confidence=0.95,
        ),
        GuidanceItem(
            metric=Metric.EPS,
            metric_detail=None,
            period=q3,
            low=1.20,
            high=None,
            unit=Unit.USD_PER_SHARE,
            basis=Basis.NON_GAAP,
            quote="We expect non-GAAP EPS of at least $1.20.",
            speaker="Colette Kress",
            confidence=0.8,
        ),
        GuidanceItem(
            metric=Metric.GROSS_MARGIN,
            metric_detail=None,
            period=full_year,
            low=75.0,
            high=75.0,
            unit=Unit.PERCENT,
            basis=Basis.NON_GAAP,
            quote="We expect full-year fiscal 2026 gross margin of approximately 75%.",
            speaker="Colette Kress",
            confidence=0.7,
        ),
    ]
