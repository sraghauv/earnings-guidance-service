"""Extraction pipeline tests. No network, no API key -- FixtureProvider
replays canned JSON instead of calling a real model.
"""

import json
from pathlib import Path

import pytest

from egs.config import Settings
from egs.db import connect, load_items, upsert_transcript
from egs.extract import pipeline
from egs.llm.fixture_provider import FixtureProvider

FIXTURES = Path(__file__).parent.parent / "fixtures" / "llm"


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(cache_dir=tmp_path / "cache", **overrides)


def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)


def test_clean_extraction_persists_items(monkeypatch, tmp_path, tmp_db, sample_transcript):
    _patch_settings(monkeypatch, _settings(tmp_path))
    upsert_transcript(tmp_db, sample_transcript)
    provider = FixtureProvider(FIXTURES, mapping={})
    user_prompt = pipeline.build_user_prompt(pipeline.full_text(sample_transcript))
    provider.mapping[user_prompt] = "clean.json"

    result = pipeline.extract_guidance(
        sample_transcript, provider, conn=tmp_db, use_cache=False
    )

    assert result.repair_attempts == 0
    assert result.cache_hit is False
    assert result.dropped_items == 0
    assert len(result.extraction.items) == 1
    assert result.extraction.items[0].metric.value == "revenue"

    stored = load_items(tmp_db, "NVDA")
    assert len(stored) == 1
    assert stored[0].period.label() == "FY2026Q3"


def test_inverted_bounds_is_repaired(monkeypatch, tmp_path, tmp_db, sample_transcript):
    _patch_settings(monkeypatch, _settings(tmp_path, max_repair_attempts=2))
    upsert_transcript(tmp_db, sample_transcript)
    provider = FixtureProvider(FIXTURES)
    user_prompt = pipeline.build_user_prompt(pipeline.full_text(sample_transcript))
    provider.mapping[user_prompt] = ["malformed_bounds.json", "repaired.json"]

    result = pipeline.extract_guidance(
        sample_transcript, provider, conn=tmp_db, use_cache=False
    )

    assert result.repair_attempts == 1
    assert provider.calls == 2
    assert result.extraction.items[0].low == 32.0
    assert result.extraction.items[0].high == 33.0


def test_repair_exhaustion_raises(monkeypatch, tmp_path, tmp_db, sample_transcript):
    _patch_settings(monkeypatch, _settings(tmp_path, max_repair_attempts=1))
    provider = FixtureProvider(FIXTURES)
    user_prompt = pipeline.build_user_prompt(pipeline.full_text(sample_transcript))
    # Always the same bad response -- nothing ever fixes it.
    provider.mapping[user_prompt] = "malformed_bounds.json"

    with pytest.raises(pipeline.RepairExhaustedError):
        pipeline.extract_guidance(sample_transcript, provider, conn=tmp_db, use_cache=False)

    # one initial attempt, one repair attempt, then give up
    assert provider.calls == 2


def test_quoted_items_win_over_a_no_guidance_flag(
    monkeypatch, tmp_path, tmp_db, sample_transcript
):
    """A model that sets no_guidance_given and then returns items keeps them.

    The items carry verbatim quotes and get checked against the transcript;
    the flag is just a summary the model contradicted. Throwing away a quoted
    figure over a stray boolean loses real data.
    """
    _patch_settings(monkeypatch, _settings(tmp_path, max_repair_attempts=0))
    upsert_transcript(tmp_db, sample_transcript)
    provider = FixtureProvider(FIXTURES)
    user_prompt = pipeline.build_user_prompt(pipeline.full_text(sample_transcript))
    provider.mapping[user_prompt] = "malformed_no_guidance_with_items.json"

    result = pipeline.extract_guidance(
        sample_transcript, provider, conn=tmp_db, use_cache=False
    )

    assert result.extraction.no_guidance_given is False
    assert len(result.extraction.items) == 1
    assert provider.calls == 1


def test_partial_items_survive_when_one_is_unusable():
    """One item with no numeric bound must not discard the others.

    Microsoft's December-quarter call returned eleven items, four of them
    growth rates with no bound, and whole-payload validation lost all eleven.
    """
    payload = {
        "ticker": "MSFT",
        "no_guidance_given": False,
        "extraction_notes": None,
        "items": [
            {
                "metric": "segment_revenue",
                "metric_detail": "Intelligent Cloud",
                "period": {"kind": "quarter", "fiscal_year": 2021, "fiscal_quarter": 2},
                "low": 13.55,
                "high": 13.8,
                "unit": "usd_billions",
                "basis": "unspecified",
                "quote": "we expect revenue between $13.55 billion and $13.8 billion",
                "speaker": None,
                "confidence": 0.9,
            },
            {
                "metric": "segment_revenue",
                "metric_detail": "Gaming",
                "period": {"kind": "quarter", "fiscal_year": 2021, "fiscal_quarter": 2},
                "low": None,
                "high": None,
                "unit": "percent",
                "basis": "unspecified",
                "quote": "we expect revenue growth in the high 20% range",
                "speaker": None,
                "confidence": 0.8,
            },
        ],
    }
    extraction, dropped = pipeline.parse_lenient(json.dumps(payload))

    assert dropped == 1
    assert [i.metric_detail for i in extraction.items] == ["Intelligent Cloud"]


def test_nothing_usable_still_goes_to_repair():
    """A response whose every item is unusable is worth another attempt."""
    payload = {
        "ticker": "MSFT",
        "no_guidance_given": False,
        "extraction_notes": None,
        "items": [
            {
                "metric": "revenue",
                "metric_detail": None,
                "period": {"kind": "quarter", "fiscal_year": 2021, "fiscal_quarter": 2},
                "low": None,
                "high": None,
                "unit": "percent",
                "basis": "unspecified",
                "quote": "revenue should grow nicely",
                "speaker": None,
                "confidence": 0.5,
            }
        ],
    }
    extraction, dropped = pipeline.parse_lenient(json.dumps(payload))
    assert extraction.items == []
    assert dropped == 1


def test_fabricated_quote_dropped(monkeypatch, tmp_path, tmp_db, sample_transcript):
    _patch_settings(monkeypatch, _settings(tmp_path))
    upsert_transcript(tmp_db, sample_transcript)
    provider = FixtureProvider(FIXTURES)
    user_prompt = pipeline.build_user_prompt(pipeline.full_text(sample_transcript))
    provider.mapping[user_prompt] = "fabricated_quote.json"

    result = pipeline.extract_guidance(
        sample_transcript, provider, conn=tmp_db, use_cache=False
    )

    assert result.dropped_items == 1
    assert len(result.extraction.items) == 1
    assert result.extraction.items[0].metric.value == "revenue"
    # the surviving item's quote really is in the transcript
    assert result.extraction.items[0].quote in sample_transcript.prepared_remarks


def test_cache_hit_avoids_second_provider_call(monkeypatch, tmp_path, tmp_db, sample_transcript):
    _patch_settings(monkeypatch, _settings(tmp_path))
    upsert_transcript(tmp_db, sample_transcript)
    provider = FixtureProvider(FIXTURES)
    user_prompt = pipeline.build_user_prompt(pipeline.full_text(sample_transcript))
    provider.mapping[user_prompt] = "clean.json"

    first = pipeline.extract_guidance(sample_transcript, provider, conn=tmp_db, use_cache=True)
    second = pipeline.extract_guidance(sample_transcript, provider, conn=tmp_db, use_cache=True)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert provider.calls == 1
    assert second.extraction.items == first.extraction.items


def test_extract_many_collects_failures_without_aborting(
    monkeypatch, tmp_path, tmp_db, sample_transcript
):
    _patch_settings(monkeypatch, _settings(tmp_path, max_repair_attempts=0))
    upsert_transcript(tmp_db, sample_transcript)
    broken = sample_transcript.model_copy(
        update={
            "id": "NVDA-2026Q3-broken",
            "prepared_remarks": "Thanks for joining. We expect Q4 revenue of $40 billion.",
        }
    )
    upsert_transcript(tmp_db, broken)
    provider = FixtureProvider(FIXTURES)

    good_prompt = pipeline.build_user_prompt(pipeline.full_text(sample_transcript))
    bad_prompt = pipeline.build_user_prompt(pipeline.full_text(broken))
    provider.mapping[good_prompt] = "clean.json"
    provider.mapping[bad_prompt] = "malformed_bounds.json"

    results, failures = pipeline.extract_many(
        [sample_transcript, broken], provider, conn=tmp_db, use_cache=False
    )

    assert len(results) == 1
    assert len(failures) == 1
    assert failures[0].transcript_id == "NVDA-2026Q3-broken"


def test_cache_hit_still_writes_to_the_database(
    monkeypatch, tmp_path, tmp_db, sample_transcript
):
    """A cache hit skips the model, not the database.

    Rebuilding a deleted database with a warm cache wrote nothing at all and
    still exited zero, because the cache-hit path returned before persisting.
    """
    _patch_settings(monkeypatch, _settings(tmp_path))
    upsert_transcript(tmp_db, sample_transcript)
    provider = FixtureProvider(FIXTURES)
    user_prompt = pipeline.build_user_prompt(pipeline.full_text(sample_transcript))
    provider.mapping[user_prompt] = "clean.json"

    first = pipeline.extract_guidance(
        sample_transcript, provider, conn=tmp_db, use_cache=True
    )
    assert first.cache_hit is False
    stored_once = len(load_items(tmp_db, sample_transcript.ticker))
    assert stored_once > 0

    # A second database, as if the first had been deleted and rebuilt.
    rebuilt = connect(tmp_path / "rebuilt.db")
    upsert_transcript(rebuilt, sample_transcript)
    calls_before = provider.calls

    second = pipeline.extract_guidance(
        sample_transcript, provider, conn=rebuilt, use_cache=True
    )

    assert second.cache_hit is True
    assert provider.calls == calls_before, "cache hit should not call the model"
    assert len(load_items(rebuilt, sample_transcript.ticker)) == stored_once


def test_upsert_transcript_commits_on_its_own(tmp_path, sample_transcript):
    """The row has to survive the connection closing.

    It used to rely on a later save_extraction to commit, so a cached
    extraction left the transcript uncommitted and it vanished.
    """
    path = tmp_path / "commit.db"
    conn = connect(path)
    upsert_transcript(conn, sample_transcript)
    conn.close()

    reopened = connect(path)
    count = reopened.execute("select count(*) from transcripts").fetchone()[0]
    assert count == 1


def test_quarter_without_a_quarter_number_is_dropped_not_stored():
    """A quarter with no quarter number labels itself as a fiscal year.

    FiscalPeriod.label() falls through to "FY2016" when fiscal_quarter is
    null, which is exactly what a legitimate full-year period produces. Seven
    of thirty-five figures in a real extraction run arrived in that state, so
    a fifth of the database was quarters masquerading as fiscal years. The
    item is dropped now rather than persisted under the wrong period.
    """
    payload = {
        "ticker": "AAPL",
        "no_guidance_given": False,
        "extraction_notes": None,
        "items": [
            {
                "metric": "revenue",
                "metric_detail": None,
                "period": {
                    "kind": "quarter",
                    "fiscal_year": 2016,
                    "fiscal_quarter": None,
                },
                "low": 11.9,
                "high": 12.0,
                "unit": "usd_billions",
                "basis": "unspecified",
                "quote": "we expect revenue between $11.9 billion and $12 billion",
                "speaker": None,
                "confidence": 0.9,
            },
            {
                "metric": "revenue",
                "metric_detail": None,
                "period": {
                    "kind": "quarter",
                    "fiscal_year": 2016,
                    "fiscal_quarter": 2,
                },
                "low": 11.9,
                "high": 12.0,
                "unit": "usd_billions",
                "basis": "unspecified",
                "quote": "we expect revenue between $11.9 billion and $12 billion",
                "speaker": None,
                "confidence": 0.9,
            },
        ],
    }
    extraction, dropped = pipeline.parse_lenient(json.dumps(payload))

    assert dropped == 1
    assert [i.period.label() for i in extraction.items] == ["FY2016Q2"]
