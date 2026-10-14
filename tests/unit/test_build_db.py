"""Tests for scripts/build_db.py.

It's a script rather than a package module, so it's loaded by path. Only the
two steps with no production caller before now are covered here: persisting a
valuation, and naming a ticker that isn't in the corpus.
"""

import importlib.util
from datetime import date
from pathlib import Path

import pytest

from egs.crosscheck.alphavantage import FixtureAlphaVantageClient
from egs.db import connect, save_extraction, upsert_transcript
from egs.schemas import (
    Basis,
    FiscalPeriod,
    GuidanceExtraction,
    GuidanceItem,
    Metric,
    PeriodKind,
    Transcript,
    Unit,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_db.py"


def _load_build_db():
    spec = importlib.util.spec_from_file_location("build_db", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def build_db():
    return _load_build_db()


def _eps_item(low=1.0, high=1.2):
    return GuidanceItem(
        metric=Metric.EPS,
        metric_detail=None,
        period=FiscalPeriod(kind=PeriodKind.QUARTER, fiscal_year=2021, fiscal_quarter=1),
        low=low,
        high=high,
        unit=Unit.USD_PER_SHARE,
        basis=Basis.NON_GAAP,
        quote="we expect EPS of $1.00 to $1.20",
        speaker=None,
        confidence=0.9,
    )


class _Result:
    def __init__(self, extraction):
        self.extraction = extraction


def _store(conn, items):
    transcript = Transcript(
        id="NVDA-2020Q4",
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        fiscal_year=2020,
        fiscal_quarter=4,
        call_date=date(2020, 2, 13),
        prepared_remarks="we expect EPS of $1.00 to $1.20",
        qa="",
    )
    upsert_transcript(conn, transcript)
    extraction = GuidanceExtraction(
        ticker="NVDA", no_guidance_given=False, items=items, extraction_notes=None
    )
    save_extraction(
        conn,
        transcript.id,
        extraction,
        provider="fixture",
        model="fixture",
        prompt_version="v1",
        schema_version="1",
        cache_key="key-1",
    )
    return _Result(extraction)


def test_eps_guidance_is_priced_and_persisted(build_db, tmp_path, fixtures_dir):
    """The valuations table had no writer at all before this.

    save_valuation existed, the table and its index existed, and nothing ever
    called it -- so the pipeline advertised a peer-multiple valuation and
    stored none.
    """
    conn = connect(tmp_path / "egs.db")
    result = _store(conn, [_eps_item()])
    client = FixtureAlphaVantageClient(fixtures_dir / "alphavantage")

    build_db._value_and_save([result], conn, client)

    rows = conn.execute("SELECT * FROM valuations").fetchall()
    assert len(rows) == 1
    assert rows[0]["ticker"] == "NVDA"
    assert rows[0]["multiple_kind"] == "pe"
    assert rows[0]["implied_price_per_share"] > 0


def test_nothing_is_persisted_without_an_alphavantage_client(build_db, tmp_path):
    """Pricing needs the peers' quotes, so no client means no valuation."""
    conn = connect(tmp_path / "egs.db")
    result = _store(conn, [_eps_item()])

    build_db._value_and_save([result], conn, None)

    assert conn.execute("SELECT COUNT(*) FROM valuations").fetchone()[0] == 0


def test_guidance_without_eps_persists_no_valuation(build_db, tmp_path, fixtures_dir):
    """Valuation runs off EPS, and most companies never guide it."""
    conn = connect(tmp_path / "egs.db")
    revenue = GuidanceItem(
        metric=Metric.REVENUE,
        metric_detail=None,
        period=FiscalPeriod(kind=PeriodKind.QUARTER, fiscal_year=2021, fiscal_quarter=1),
        low=2.94,
        high=3.06,
        unit=Unit.USD_BILLIONS,
        basis=Basis.UNSPECIFIED,
        quote="we expect revenue of $3 billion plus or minus 2%",
        speaker=None,
        confidence=0.9,
    )
    result = _store(conn, [revenue])
    client = FixtureAlphaVantageClient(fixtures_dir / "alphavantage")

    build_db._value_and_save([result], conn, client)

    assert conn.execute("SELECT COUNT(*) FROM valuations").fetchone()[0] == 0
