"""API tests against a tmp SQLite db with the provider and AlphaVantage
client swapped for fakes via app.dependency_overrides.

Routes that call out to a sibling module (extract/pipeline.py,
valuation/multiples.py) do a lazy import inside the handler, so tests inject a
fake through sys.modules to keep the route under test isolated from the real
extraction and valuation code.

TestClient drives the app from a background thread, so the connection handed to
the app has to work from that thread and the test's own -- which is what
db.connect()'s check_same_thread=False gives us.
"""

import sys
import types

import pytest
from fastapi.testclient import TestClient

from egs.api.deps import get_av_client, get_conn, get_provider
from egs.api.main import create_app
from egs.config import Settings
from egs.db import connect, save_extraction, save_mismatches, upsert_transcript
from egs.llm.base import Completion, LLMProvider, ProviderError
from egs.schemas import (
    Basis,
    FiscalPeriod,
    GuidanceExtraction,
    GuidanceItem,
    Metric,
    Mismatch,
    MismatchKind,
    PeriodKind,
    Unit,
)


class FakeProvider(LLMProvider):
    name = "fake"
    model = "fake-model"

    def __init__(self, *, raise_error: bool = False) -> None:
        self.raise_error = raise_error

    def complete_json(self, *, system, user, schema, repair_hint=None):
        if self.raise_error:
            raise ProviderError("fake provider: simulated backend failure")
        return Completion(text="{}")


class FakeAVClient:
    """Stand-in for crosscheck.alphavantage.AlphaVantageClient."""


@pytest.fixture
def app_client(tmp_path):
    conn = connect(tmp_path / "api.db")
    settings = Settings(db_path=tmp_path / "unused.db", llm_provider="fixture")
    app = create_app(settings=settings)
    app.dependency_overrides[get_conn] = lambda: conn
    app.dependency_overrides[get_provider] = lambda: FakeProvider()
    app.dependency_overrides[get_av_client] = lambda: FakeAVClient()
    with TestClient(app) as client:
        yield client, conn
    conn.close()


def test_healthz_reports_db_and_provider(app_client):
    client, _ = app_client
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["db_reachable"] is True
    assert body["llm_provider"] == "fixture"


def test_companies_lists_tickers_with_transcript_counts(app_client, sample_transcript):
    client, conn = app_client
    upsert_transcript(conn, sample_transcript)
    second_call = sample_transcript.model_copy(
        update={"id": "NVDA-2026Q3", "fiscal_quarter": 3}
    )
    upsert_transcript(conn, second_call)

    resp = client.get("/companies")
    assert resp.status_code == 200
    assert resp.json() == [{"ticker": "NVDA", "transcript_count": 2}]


def test_transcripts_filters_by_ticker_and_year(app_client, sample_transcript):
    client, conn = app_client
    upsert_transcript(conn, sample_transcript)
    other_ticker = sample_transcript.model_copy(
        update={"id": "AMD-2026Q2", "ticker": "AMD"}
    )
    upsert_transcript(conn, other_ticker)

    resp = client.get("/transcripts", params={"ticker": "NVDA"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["ticker"] == "NVDA"
    # no body text leaks into the listing
    assert "prepared_remarks" not in body[0]
    assert "qa" not in body[0]


def test_guidance_404_on_unknown_transcript_id(app_client):
    client, _ = app_client
    resp = client.get("/transcripts/does-not-exist/guidance")
    assert resp.status_code == 404


def test_guidance_returns_stored_items(app_client, sample_transcript, sample_guidance_items):
    client, conn = app_client
    upsert_transcript(conn, sample_transcript)
    extraction = GuidanceExtraction(
        ticker="NVDA", no_guidance_given=False, items=sample_guidance_items, extraction_notes=None
    )
    save_extraction(
        conn,
        sample_transcript.id,
        extraction,
        provider="fixture",
        model="fixture",
        prompt_version="v1",
        schema_version="1",
        cache_key="key-1",
    )

    resp = client.get(f"/transcripts/{sample_transcript.id}/guidance")
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript_id"] == sample_transcript.id
    assert len(body["items"]) == len(sample_guidance_items)


def test_extract_502_when_provider_raises(app_client, sample_transcript, monkeypatch):
    client, conn = app_client
    upsert_transcript(conn, sample_transcript)
    client.app.dependency_overrides[get_provider] = lambda: FakeProvider(raise_error=True)

    fake_pipeline = types.ModuleType("egs.extract.pipeline")

    class RepairExhaustedError(RuntimeError):
        pass

    def fake_extract_guidance(transcript, provider, *, conn, use_cache):
        provider.complete_json(system="", user="", schema={})

    fake_pipeline.extract_guidance = fake_extract_guidance
    fake_pipeline.RepairExhaustedError = RepairExhaustedError
    monkeypatch.setitem(sys.modules, "egs.extract.pipeline", fake_pipeline)

    resp = client.post("/extract", json={"transcript_id": sample_transcript.id})
    assert resp.status_code == 502
    assert "fake provider" in resp.json()["detail"]


def test_extract_404_on_unknown_transcript(app_client):
    client, _ = app_client
    resp = client.post("/extract", json={"transcript_id": "does-not-exist"})
    assert resp.status_code == 404


def test_valuation_404_when_the_ticker_is_unknown(app_client):
    """404 is reserved for a ticker the database has never seen."""
    client, _ = app_client
    resp = client.get("/valuation/NVDA")
    assert resp.status_code == 404
    assert "no guidance stored" in resp.json()["detail"]


def _store_eps_guidance(conn, transcript, items, cache_key="key-1"):
    upsert_transcript(conn, transcript)
    extraction = GuidanceExtraction(
        ticker=transcript.ticker, no_guidance_given=False, items=items, extraction_notes=None
    )
    save_extraction(
        conn,
        transcript.id,
        extraction,
        provider="fixture",
        model="fixture",
        prompt_version="v1",
        schema_version="1",
        cache_key=cache_key,
    )


def test_valuation_computes_pe_based_value_from_real_fixtures(
    app_client, sample_transcript, sample_guidance_items, fixtures_dir
):
    """NVDA's peer group (AMD/AVGO/INTC/QCOM) has real recorded fixtures for
    three of the four, which is enough for a median P/E -- exercises the
    real valuation.multiples/peers code, not a stand-in."""
    from egs.crosscheck.alphavantage import FixtureAlphaVantageClient

    client, conn = app_client
    _store_eps_guidance(conn, sample_transcript, sample_guidance_items)
    client.app.dependency_overrides[get_av_client] = lambda: FixtureAlphaVantageClient(
        fixtures_dir / "alphavantage"
    )

    resp = client.get("/valuation/NVDA")
    assert resp.status_code == 200
    body = resp.json()
    assert body["computed"] is True
    assert body["reason"] is None
    assert body["valuation"]["multiple_kind"] == "pe"
    assert body["valuation"]["implied_price_per_share"] > 0


def test_valuation_returns_reason_when_ticker_has_no_mapped_peers(
    app_client, sample_transcript, sample_guidance_items, fixtures_dir
):
    from egs.crosscheck.alphavantage import FixtureAlphaVantageClient

    client, conn = app_client
    unmapped = sample_transcript.model_copy(update={"id": "ZZZZ-2026Q2", "ticker": "ZZZZ"})
    eps_item = next(i for i in sample_guidance_items if i.metric is Metric.EPS)
    _store_eps_guidance(conn, unmapped, [eps_item])
    client.app.dependency_overrides[get_av_client] = lambda: FixtureAlphaVantageClient(
        fixtures_dir / "alphavantage"
    )

    resp = client.get("/valuation/ZZZZ")
    assert resp.status_code == 200
    body = resp.json()
    assert body["computed"] is False
    assert body["valuation"] is None
    assert "no peer set" in body["reason"]


def test_mismatches_pagination(app_client):
    client, conn = app_client
    rows = [
        Mismatch(
            kind=MismatchKind.VS_REPORTED,
            ticker="NVDA",
            metric=Metric.REVENUE,
            period_label=f"FY2026Q{i}",
            guided_value=30.0 + i,
            comparison_value=29.0 + i,
            relative_error=0.03,
            detail=f"mismatch {i}",
            quote=None,
        )
        for i in range(1, 4)
    ]
    save_mismatches(conn, rows)

    resp = client.get("/mismatches", params={"ticker": "NVDA", "limit": 2, "offset": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 0

    resp2 = client.get("/mismatches", params={"ticker": "NVDA", "limit": 2, "offset": 2})
    assert len(resp2.json()["items"]) == 1

    resp3 = client.get("/mismatches", params={"ticker": "AMD"})
    assert resp3.json()["total"] == 0


def test_valuation_explains_itself_when_eps_is_not_among_the_stored_metrics(
    app_client, sample_transcript
):
    """A company that guides revenue but not EPS isn't a 404.

    Valuation runs off EPS guidance, and plenty of companies never give it --
    NVIDIA guides revenue, margins, opex, OI&E, tax rate and capex and no EPS.
    That's the ordinary can't-compute answer, so it's reported like the other
    can't-compute cases rather than as a missing resource.
    """
    client, conn = app_client
    revenue = GuidanceItem(
        metric=Metric.REVENUE,
        metric_detail=None,
        period=FiscalPeriod(kind=PeriodKind.QUARTER, fiscal_year=2021, fiscal_quarter=1),
        low=2.94,
        high=3.06,
        unit=Unit.USD_BILLIONS,
        basis=Basis.UNSPECIFIED,
        quote="We expect revenue to be $3 billion plus or minus 2%.",
        speaker=None,
        confidence=0.9,
    )
    _store_eps_guidance(conn, sample_transcript, [revenue])

    resp = client.get("/valuation/NVDA")

    assert resp.status_code == 200
    body = resp.json()
    assert body["computed"] is False
    assert body["valuation"] is None
    assert "needs EPS guidance" in body["reason"]
    # The caller is told what it can work with instead of guessing.
    assert "revenue" in body["reason"]
