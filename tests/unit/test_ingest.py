"""Tests for the parquet loader and transcript normalizer.

Builds a tiny in-memory table matching the real sp500 transcript schema
instead of touching the 1.7GB fixture — that file is never read in tests.
"""

import pyarrow as pa
import pyarrow.parquet as pq

from egs.ingest.normalize import split_call, speaker_turns, strip_boilerplate
from egs.ingest.parquet_loader import available_tickers, iter_transcripts

SCHEMA = pa.schema(
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


def _row(
    symbol="ACME",
    quarter=1,
    year=2021,
    date="2021-02-15 16:30:00",
    content="prepared remarks here.",
    structured_content=None,
    company_name="Acme Corp",
    company_id=1.0,
):
    return {
        "symbol": symbol,
        "quarter": quarter,
        "year": year,
        "date": date,
        "content": content,
        "structured_content": structured_content,
        "company_name": company_name,
        "company_id": company_id,
    }


def _write_parquet(tmp_path, rows):
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    path = tmp_path / "transcripts.parquet"
    pq.write_table(table, path)
    return path


# ---- parquet_loader ----


def test_iter_transcripts_builds_stable_id(tmp_path):
    path = _write_parquet(tmp_path, [_row(symbol="NVDA", quarter=3, year=2024)])
    transcripts = list(iter_transcripts(path))
    assert len(transcripts) == 1
    t = transcripts[0]
    assert t.id == "NVDA-2024Q3"
    assert t.ticker == "NVDA"
    assert t.fiscal_year == 2024
    assert t.fiscal_quarter == 3
    assert t.call_date.isoformat() == "2021-02-15"


def test_iter_transcripts_filters_by_ticker(tmp_path):
    path = _write_parquet(
        tmp_path,
        [
            _row(symbol="NVDA", year=2021),
            _row(symbol="AMD", year=2021),
        ],
    )
    transcripts = list(iter_transcripts(path, tickers=["AMD"]))
    assert [t.ticker for t in transcripts] == ["AMD"]


def test_iter_transcripts_filters_by_year(tmp_path):
    path = _write_parquet(
        tmp_path,
        [
            _row(symbol="NVDA", year=2020),
            _row(symbol="NVDA", year=2021),
        ],
    )
    transcripts = list(iter_transcripts(path, years=[2021]))
    assert len(transcripts) == 1
    assert transcripts[0].fiscal_year == 2021


def test_iter_transcripts_respects_limit(tmp_path):
    rows = [_row(symbol="NVDA", quarter=q) for q in (1, 2, 3, 4)]
    path = _write_parquet(tmp_path, rows)
    transcripts = list(iter_transcripts(path, limit=2))
    assert len(transcripts) == 2


def test_iter_transcripts_skips_malformed_date(tmp_path):
    rows = [
        _row(symbol="NVDA", quarter=1, date="2021-02-15 16:30:00"),
        _row(symbol="NVDA", quarter=2, date="not-a-date"),
        _row(symbol="NVDA", quarter=3, date=None),
    ]
    path = _write_parquet(tmp_path, rows)
    transcripts = list(iter_transcripts(path))
    assert len(transcripts) == 1
    assert transcripts[0].fiscal_quarter == 1


def test_available_tickers_reads_distinct_symbols(tmp_path):
    rows = [
        _row(symbol="NVDA", quarter=1),
        _row(symbol="NVDA", quarter=2),
        _row(symbol="AMD", quarter=1),
    ]
    path = _write_parquet(tmp_path, rows)
    assert available_tickers(path) == ["AMD", "NVDA"]


# ---- normalize ----


def test_split_call_finds_boundary_via_structured_content():
    structured = [
        {"speaker": "Jane CEO, Chief Executive Officer", "text": "Thanks for joining."},
        {"speaker": "Jane CEO, Chief Executive Officer", "text": "Revenue grew 10%."},
        {
            "speaker": "Operator",
            "text": "We will now begin the question-and-answer session.",
        },
        {"speaker": "Operator", "text": "Our first question comes from Analyst A."},
        {"speaker": "Analyst A - Big Bank", "text": "What drove the growth?"},
    ]
    prepared, qa = split_call(structured, content=None)
    assert "Revenue grew 10%" in prepared
    assert "question-and-answer session" in qa.lower()
    assert "What drove the growth" in qa
    assert "Revenue grew" not in qa


def test_split_call_regex_fallback_when_no_structured_content():
    content = (
        "Good morning. Thanks for joining the call. Revenue was strong this "
        "quarter.\n\nOperator: We'll now begin the question and answer session.\n\n"
        "Analyst: What drove the growth?"
    )
    prepared, qa = split_call(None, content)
    assert "Revenue was strong" in prepared
    assert "question and answer session" in qa.lower()
    assert "What drove the growth" in qa
    assert "Revenue was strong" not in qa


def test_split_call_returns_everything_as_prepared_when_no_boundary():
    structured = [
        {"speaker": "Jane CEO", "text": "Thanks for joining."},
        {"speaker": "Jane CEO", "text": "Revenue grew 10%."},
    ]
    prepared, qa = split_call(structured, content=None)
    assert qa == ""
    assert "Revenue grew 10%" in prepared


def test_split_call_regex_fallback_no_boundary_returns_everything():
    content = "Just a plain transcript with no Q&A cue phrases anywhere in it."
    prepared, qa = split_call(None, content)
    assert prepared == content
    assert qa == ""


def test_split_call_handles_empty_structured_content_list():
    prepared, qa = split_call([], "Fallback content with no cues.")
    assert prepared == "Fallback content with no cues."
    assert qa == ""


def test_speaker_turns_strips_titles_after_comma_and_dash():
    structured = [
        {"speaker": "John Smith, Chief Financial Officer", "text": "Hi."},
        {"speaker": "Jane Doe - Analyst", "text": "Hello."},
        {"speaker": "Operator", "text": "Welcome."},
    ]
    turns = speaker_turns(structured)
    assert turns == [
        ("John Smith", "Hi."),
        ("Jane Doe", "Hello."),
        ("Operator", "Welcome."),
    ]


def test_strip_boilerplate_drops_safe_harbor_paragraph():
    text = (
        "Revenue grew 10% year over year.\n\n"
        "This presentation contains forward-looking statements that are "
        "subject to risks and uncertainties under safe harbor provisions.\n\n"
        "We expect continued momentum next quarter."
    )
    cleaned = strip_boilerplate(text)
    assert "Revenue grew 10%" in cleaned
    assert "continued momentum" in cleaned
    assert "safe harbor" not in cleaned.lower()


def test_strip_boilerplate_drops_operator_housekeeping():
    text = (
        "Good day, ladies and gentlemen, and welcome to the Acme Corp "
        "earnings call. All participants are in a listen-only mode.\n\n"
        "Thanks everyone for joining, let's get started on results."
    )
    cleaned = strip_boilerplate(text)
    assert "listen-only" not in cleaned
    assert "let's get started on results" in cleaned


def test_strip_boilerplate_leaves_clean_text_untouched():
    text = "Revenue was $5B this quarter, up 12% year over year."
    assert strip_boilerplate(text) == text
