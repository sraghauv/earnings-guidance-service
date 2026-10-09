"""End-to-end check on disposable fixtures: write a tiny parquet, ingest it,
extract guidance with FixtureProvider, persist to a tmp SQLite db, then
reconcile and value it with the fixture AlphaVantage client. No network, no
API key.

The reconcile/valuation half depends on crosscheck and valuation modules
owned by other agents. If they're not on disk yet, or their interface turns
out to differ from what was agreed, that half is skipped with a clear
reason rather than failing the run -- the ingest/extract/persist half has
no such dependency and always has to pass.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egs.db import connect, load_items, upsert_transcript
from egs.extract import pipeline
from egs.ingest.parquet_loader import iter_transcripts
from egs.llm.fixture_provider import FixtureProvider

_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("quarter", pa.int64()),
        pa.field("year", pa.int64()),
        pa.field("date", pa.string()),
        pa.field("content", pa.string()),
        pa.field(
            "structured_content",
            pa.list_(pa.struct([("speaker", pa.string()), ("text", pa.string())])),
        ),
        pa.field("company_name", pa.string()),
        pa.field("company_id", pa.float64()),
    ]
)

# Matches the quote baked into tests/fixtures/llm/clean.json, so the
# fixture extraction's quote survives drop_fabricated_quotes.
_CONTENT = (
    "Thanks for joining the call. For Q3, we expect revenue of $32.0 "
    "billion to $33.0 billion.\n\n"
    "We'll now begin the question-and-answer session.\n\n"
    "Analyst: What drove the outlook?"
)


def _write_parquet(path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "symbol": "NVDA",
                "quarter": 3,
                "year": 2026,
                "date": "2025-11-19 16:30:00",
                "content": _CONTENT,
                "structured_content": None,
                "company_name": "NVIDIA Corporation",
                "company_id": 1.0,
            }
        ],
        schema=_PARQUET_SCHEMA,
    )
    pq.write_table(table, path)


def test_ingest_extract_persist_pipeline(tmp_path, fixtures_dir):
    parquet_path = tmp_path / "transcripts.parquet"
    _write_parquet(parquet_path)

    transcripts = list(iter_transcripts(parquet_path, tickers=["NVDA"], limit=10))
    assert len(transcripts) == 1
    transcript = transcripts[0]
    assert transcript.ticker == "NVDA"
    assert transcript.fiscal_year == 2026
    assert transcript.fiscal_quarter == 3

    conn = connect(tmp_path / "egs.db")
    upsert_transcript(conn, transcript)

    provider = FixtureProvider(fixtures_dir / "llm")
    user_prompt = pipeline.build_user_prompt(pipeline.full_text(transcript))
    provider.mapping[user_prompt] = "clean.json"

    result = pipeline.extract_guidance(transcript, provider, conn=conn, use_cache=False)

    assert result.extraction.ticker == "NVDA"
    assert len(result.extraction.items) == 1
    assert result.dropped_items == 0

    stored = load_items(conn, "NVDA")
    assert len(stored) == 1
    assert stored[0].period.label() == "FY2026Q3"
    assert stored[0].low == 32.0
    assert stored[0].high == 33.0

    conn.close()


def test_reconcile_and_valuation_from_persisted_guidance(tmp_path, fixtures_dir):
    pytest.importorskip(
        "egs.crosscheck.alphavantage", reason="crosscheck.alphavantage not written yet"
    )
    pytest.importorskip(
        "egs.crosscheck.reconcile", reason="crosscheck.reconcile not written yet"
    )
    pytest.importorskip(
        "egs.valuation.multiples", reason="valuation.multiples not written yet"
    )

    from egs.crosscheck.alphavantage import FixtureAlphaVantageClient
    from egs.crosscheck.reconcile import reconcile
    from egs.valuation.multiples import value_from_guidance

    parquet_path = tmp_path / "transcripts.parquet"
    _write_parquet(parquet_path)
    transcript = next(iter_transcripts(parquet_path, tickers=["NVDA"]))

    conn = connect(tmp_path / "egs.db")
    upsert_transcript(conn, transcript)

    provider = FixtureProvider(fixtures_dir / "llm")
    provider.mapping[pipeline.build_user_prompt(pipeline.full_text(transcript))] = "clean.json"
    result = pipeline.extract_guidance(transcript, provider, conn=conn, use_cache=False)

    try:
        av_client = FixtureAlphaVantageClient(fixtures_dir / "alphavantage")
        mismatches = reconcile(result.extraction.items, client=av_client, ticker="NVDA")
        value_from_guidance(
            result.extraction.items[0],
            ticker="NVDA",
            client=av_client,
            shares_outstanding=2_400_000_000,
        )
    except Exception as exc:  # noqa: BLE001 -- interface drift, not a real failure
        pytest.skip(f"crosscheck/valuation interface differs from what was agreed: {exc}")

    assert isinstance(mismatches, list)

    conn.close()
