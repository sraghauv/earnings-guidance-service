"""Tests for the parquet loader and transcript normalizer.

Builds a tiny in-memory table matching the real sp500 transcript schema
instead of touching the 1.7GB fixture — that file is never read in tests.
"""

import pyarrow as pa
import pyarrow.parquet as pq

from egs.ingest.normalize import (
    MAX_BOILERPLATE_CHARS,
    speaker_turns,
    split_call,
    strip_boilerplate,
)
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


# The phrasings below are taken verbatim from real Apple and Boeing calls.
# Each one used to be split wrongly, in opposite directions: Apple's hand-off
# went undetected so the whole call became prepared remarks, and Boeing's
# opening housekeeping matched so prepared remarks came out empty.


def test_apple_hand_off_is_the_boundary_not_the_agenda_preview():
    structured = [
        {"speaker": "Operator", "text": "Good day and welcome to the Apple call."},
        {
            "speaker": "Tim Cook",
            "text": "After that, we'll open the call to questions from analysts. "
            "We expect revenue to be $90 billion.",
        },
        {"speaker": "Luca Maestri", "text": "With that let's open the call to questions."},
        {"speaker": "Analyst", "text": "Thanks for taking my question."},
    ]
    prepared, qa = split_call(structured, content=None)
    # The agenda preview in turn 1 must not end the prepared remarks.
    assert "We expect revenue to be $90 billion." in prepared
    assert qa.startswith("With that let's open the call to questions.")


def test_curly_apostrophe_cue_still_matches():
    structured = [
        {"speaker": "Operator", "text": "Welcome to the call."},
        {"speaker": "CFO", "text": "We expect revenue of $3 billion."},
        {"speaker": "Operator", "text": "We’ll now begin the question and answer session."},
        {"speaker": "Analyst", "text": "My question is about margins."},
    ]
    prepared, qa = split_call(structured, content=None)
    assert "We expect revenue of $3 billion." in prepared
    assert "question and answer session" in qa


def test_operator_describing_the_format_does_not_empty_prepared_remarks():
    """Boeing's operator mentions the Q&A while listing what's being broadcast.

    This tripped a weak cue on turn 0, which put the entire call in `qa` on
    every Boeing quarter in the corpus.
    """
    structured = [
        {
            "speaker": "Operator",
            "text": "The management discussion and slide presentation, plus the "
            "analyst question-and-answer session, are being broadcast live "
            "over the Internet.",
        },
        {"speaker": "CFO", "text": "We expect free cash flow to be positive."},
        {"speaker": "Operator", "text": "Our first question will come from the line of Myles Walton."},
        {"speaker": "Analyst", "text": "Thanks."},
    ]
    prepared, qa = split_call(structured, content=None)
    assert prepared
    assert "We expect free cash flow to be positive." in prepared
    assert qa.startswith("Our first question will come from")


def test_first_question_phrasing_variants_all_match():
    variants = [
        "Our first question will come from the line of Myles Walton with UBS.",
        "Our first question will be from Peter Arment with Baird.",
        "Our first question from Doug Harned with Bernstein, please go ahead.",
        "Operator, may we please have the first question?",
        "Operator, may have the first question, please?",
    ]
    for cue in variants:
        structured = [
            {"speaker": "Operator", "text": "Welcome to the call."},
            {"speaker": "CFO", "text": "We expect revenue of $3 billion."},
            {"speaker": "Operator", "text": cue},
        ]
        prepared, qa = split_call(structured, content=None)
        assert "We expect revenue of $3 billion." in prepared, cue
        assert qa == cue, cue


def test_regex_fallback_skips_a_cue_in_the_opening_paragraph():
    """Same guard as the structured path, for calls with no speaker turns.

    A cue inside the very first paragraph would otherwise put the boundary at
    offset 0 and leave prepared remarks empty.
    """
    content = (
        "Welcome. The analyst question-and-answer session is being broadcast live.\n\n"
        "We expect revenue to be $3 billion plus or minus 2%.\n\n"
        "Our first question will come from the line of Myles Walton."
    )
    prepared, qa = split_call(None, content)
    assert "We expect revenue to be $3 billion" in prepared
    assert qa.startswith("Our first question will come from")


def test_company_run_qa_without_an_operator_is_still_a_boundary():
    """Tesla takes written retail and institutional questions before analysts.

    There's no operator hand-off at all -- the company's own line opens the
    Q&A -- so three of ten Tesla calls found no boundary and kept the whole
    transcript as prepared remarks.
    """
    for cue in [
        "And we will now take questions from institutional and retail shareholders.",
        "And we will first take a retail questions from, say, our website.",
        "Let's now go to investor questions.",
        "And now let's go to analyst questions.",
        "Thank you and we will open it up for questions.",
    ]:
        structured = [
            {"speaker": "Operator", "text": "Welcome to the call."},
            {"speaker": "CFO", "text": "We expect revenue of $3 billion."},
            {"speaker": "IR", "text": cue},
        ]
        prepared, qa = split_call(structured, content=None)
        assert "We expect revenue of $3 billion." in prepared, cue
        assert qa == cue, cue


def test_a_presentation_containing_a_disclaimer_is_not_boilerplate():
    """Costco's CFO reads the safe harbor and the whole presentation as one turn.

    A speaker turn is one paragraph here, so matching the disclaimer and
    dropping the paragraph deleted all 19,952 characters of prepared remarks
    on three of Costco's four calls -- and prepared remarks are where guidance
    lives.
    """
    presentation = (
        "Thank you and good afternoon. I will start by stating that these "
        "discussions will include forward-looking statements within the meaning "
        "of the Private Securities Litigation Reform Act, which involve risks "
        "and uncertainties that could cause results to differ materially. "
        + "Comparable sales rose and we expect capital expenditure of $900 million. " * 60
    )
    assert len(presentation) > MAX_BOILERPLATE_CHARS
    kept = strip_boilerplate(presentation)
    assert "capital expenditure of $900 million" in kept
    # Nothing removed at all -- only surrounding whitespace.
    assert kept == presentation.strip()


def test_short_disclaimer_paragraphs_are_still_dropped():
    text = (
        "Ladies and gentlemen, thank you for standing by and welcome to the call.\n\n"
        "We expect revenue of $3 billion in the fourth quarter.\n\n"
        "This call contains forward-looking statements involving risks and "
        "uncertainties that could cause actual results to differ materially."
    )
    kept = strip_boilerplate(text)
    assert kept == "We expect revenue of $3 billion in the fourth quarter."


def test_operator_opening_plus_one_long_turn_keeps_the_long_turn():
    """The real COST-2020Q3 shape: 421 chars of housekeeping, then 20k of remarks."""
    text = (
        "Ladies and gentlemen, thank you for standing by and welcome to the Q3 "
        "earnings call. All participants are in a listen-only mode.\n\n"
        "Thank you. These discussions include forward-looking statements with "
        "risks and uncertainties. "
        + "We expect membership fee income to grow. " * 80
    )
    kept = strip_boilerplate(text)
    assert "Ladies and gentlemen" not in kept
    assert "membership fee income" in kept


def test_host_intro_previewing_the_qa_is_not_the_boundary():
    """Walmart's IR host previews the agenda before the executives speak.

    "That will be followed by our question-and-answer session" is a forward
    reference, but the guard allowed only "followed by a", so this landed as
    the boundary and put all 18k characters of executive remarks in the Q&A.
    """
    structured = [
        {"speaker": "Operator", "text": "Welcome to the call. " * 10},
        {
            "speaker": "Dan Binder",
            "text": "Good morning. In a few moments, Doug and Brett will provide an "
            "update on the business. That will be followed by our "
            "question-and-answer session. Before I turn the call over to Doug, "
            "let me remind you of our safe harbor.",
        },
        {"speaker": "Doug McMillon", "text": "We expect comp sales to grow. " * 300},
        {"speaker": "Operator", "text": "Our first question comes from Simeon Gutman."},
        {"speaker": "Analyst", "text": "Thanks."},
    ]
    prepared, qa = split_call(structured, content=None)
    assert "We expect comp sales to grow." in prepared
    assert qa.startswith("Our first question comes from")


def test_weak_cue_is_not_trusted_before_any_real_remarks():
    """A weak cue that early is mis-splitting the call, whatever it says.

    The structural net: an operator line plus a short host intro is not a
    prepared-remarks section, so a weak cue sitting there can't be the
    hand-off no matter how it's phrased.
    """
    structured = [
        {"speaker": "Operator", "text": "Welcome."},
        {"speaker": "IR", "text": "We will have a question and answer session shortly."},
        {"speaker": "CEO", "text": "We expect revenue of $3 billion. " * 200},
        {"speaker": "Operator", "text": "We'll now begin the question and answer session."},
    ]
    prepared, qa = split_call(structured, content=None)
    assert "We expect revenue of $3 billion." in prepared
    assert qa.startswith("We'll now begin")


def test_strong_cue_is_trusted_even_with_short_prepared_remarks():
    """Only weak cues carry the length condition; strong cues are unambiguous."""
    structured = [
        {"speaker": "Operator", "text": "Welcome."},
        {"speaker": "CFO", "text": "We expect revenue of $3 billion."},
        {"speaker": "Operator", "text": "We'll now begin the question and answer session."},
    ]
    prepared, qa = split_call(structured, content=None)
    assert prepared.endswith("We expect revenue of $3 billion.")
    assert qa.startswith("We'll now begin")
