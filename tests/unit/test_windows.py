"""Tests for extract/windows.py.

The windowing exists because a small model handed a whole call reports no
guidance at all, so the cases that matter here are the ones where a window
would otherwise come back empty or swallow the entire transcript.
"""

from datetime import date

from egs.extract.windows import (
    MAX_WINDOW_CHARS,
    MAX_WINDOWS,
    guidance_windows,
)
from egs.schemas import Transcript

OUTLOOK = (
    "Let me turn to the outlook for the first quarter of fiscal 2021. "
    "We expect revenue to be $3 billion plus or minus 2%. "
    "Capital expenditures are expected to be approximately $150 million."
)
FILLER = "Thank you all for joining us on the call this afternoon. " * 200


def _transcript(prepared: str, qa: str = "") -> Transcript:
    return Transcript(
        id="TEST-2025Q1",
        ticker="TEST",
        company_name="Test Co",
        fiscal_year=2025,
        fiscal_quarter=1,
        call_date=date(2025, 2, 13),
        prepared_remarks=prepared,
        qa=qa,
    )


def test_short_call_is_one_window():
    t = _transcript("We expect revenue of $3 billion.", "Any questions?")
    windows = guidance_windows(t)
    assert len(windows) == 1
    assert "3 billion" in windows[0]
    assert "Any questions?" in windows[0]


def test_cue_window_excludes_unrelated_filler():
    t = _transcript(FILLER + OUTLOOK + FILLER)
    windows = guidance_windows(t)
    assert len(windows) == 1
    assert "plus or minus 2%" in windows[0]
    # The whole point: the window is a small slice of a long call.
    assert len(windows[0]) <= MAX_WINDOW_CHARS
    assert len(windows[0]) < len(t.prepared_remarks) / 4


def test_overlapping_cues_do_not_produce_duplicate_windows():
    # Two cues close together fall inside one window span and must merge.
    t = _transcript(FILLER + OUTLOOK + " We expect revenue to be flat. " + FILLER)
    assert len(guidance_windows(t)) == 1


def test_no_cue_falls_back_to_chunks_not_whole_call():
    body = "We expect to grow and anticipate 5 percent improvement. " * 400
    t = _transcript(body)
    windows = guidance_windows(t)
    assert windows
    assert all(len(w) <= MAX_WINDOW_CHARS for w in windows)


def test_chunk_fallback_skips_passages_with_no_figures():
    # Forward-looking language but no numbers anywhere: nothing to extract.
    t = _transcript("We expect good things and anticipate growth. " * 300)
    assert guidance_windows(t) == []


def test_window_count_is_capped():
    body = (OUTLOOK + FILLER[:200]) * 40
    assert len(guidance_windows(body and _transcript(body))) <= MAX_WINDOWS


def test_everything_in_qa_still_yields_windows():
    """A Q&A boundary at the very start leaves prepared_remarks empty.

    Real calls do this -- COST-2020Q2 produced zero windows before the
    combined-text fallback existed, despite being 38k characters long.
    """
    body = "We expect revenue of $3 billion and anticipate 5% growth. " * 200
    t = _transcript("", body)
    windows = guidance_windows(t)
    assert windows
    assert all(len(w) <= MAX_WINDOW_CHARS for w in windows)


def test_qa_guidance_is_picked_up_by_cue():
    t = _transcript(FILLER, FILLER + "Our guidance for the full year is unchanged." + FILLER)
    windows = guidance_windows(t)
    assert any("guidance for the full year" in w for w in windows)


def test_guidance_free_long_call_yields_nothing():
    t = _transcript("Thanks for joining. It was a solid quarter. " * 300)
    assert guidance_windows(t) == []
