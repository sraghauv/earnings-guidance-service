"""Tests for extract/batch_runner.py against a stubbed Anthropic client.

The batch path can't be exercised for real here (no API key, and a batch takes
up to an hour), so these pin down the parts that are easy to get wrong without
noticing: one request per window rather than per transcript, results matched by
custom_id rather than position, and per-window failures not taking the rest of
the transcript down with them.
"""

import json
from datetime import date
from types import SimpleNamespace

import pytest

from egs.extract.batch_runner import collect_results, submit_batch
from egs.schemas import Transcript

GUIDANCE = (
    "Let me turn to the outlook for the first quarter of fiscal 2021. "
    "We expect revenue to be $3 billion plus or minus 2%."
)
FILLER = "Thanks for joining us on the call today. " * 150


def _transcript(tid: str, ticker: str, prepared: str) -> Transcript:
    return Transcript(
        id=tid,
        ticker=ticker,
        company_name=f"{ticker} Inc",
        fiscal_year=2020,
        fiscal_quarter=4,
        call_date=date(2020, 2, 13),
        prepared_remarks=prepared,
        qa="",
    )


def _extraction_json(ticker: str, quote: str, low: float, high: float) -> str:
    """One revenue item, as the model would return it."""
    return json.dumps(
        {
            "ticker": ticker,
            "no_guidance_given": False,
            "extraction_notes": None,
            "items": [
                {
                    "metric": "revenue",
                    "metric_detail": None,
                    "period": {
                        "kind": "quarter",
                        "fiscal_year": 2021,
                        "fiscal_quarter": 1,
                    },
                    "low": low,
                    "high": high,
                    "unit": "usd_billions",
                    "basis": "unspecified",
                    "quote": quote,
                    "speaker": None,
                    "confidence": 0.9,
                }
            ],
        }
    )


class StubBatches:
    def __init__(self, results=None):
        self.created_requests = None
        self._results = results or []

    def create(self, *, requests):
        self.created_requests = requests
        return SimpleNamespace(id="batch_test")

    def results(self, batch_id):
        return iter(self._results)


class StubClient:
    def __init__(self, results=None):
        self.messages = SimpleNamespace(batches=StubBatches(results))


def _result(custom_id, *, text=None, type_="succeeded", stop_reason="end_turn"):
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    message = SimpleNamespace(
        content=content, model="claude-opus-5", stop_reason=stop_reason
    )
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type=type_, message=message),
    )


def test_submit_creates_one_request_per_window():
    t = _transcript("NVDA-2020Q4", "NVDA", FILLER + GUIDANCE + FILLER)
    client = StubClient()
    submit_batch([t], client=client, model="claude-opus-5")

    requests = client.messages.batches.created_requests
    assert len(requests) >= 1
    # custom_id has to carry both the transcript and which window it was.
    assert all(r["custom_id"].startswith("NVDA-2020Q4#") for r in requests)
    # The request body is a window, not the whole call.
    body = requests[0]["params"]["messages"][0]["content"]
    assert len(body) < len(t.prepared_remarks)


def test_submit_raises_when_nothing_has_a_window():
    t = _transcript("DULL-2020Q4", "DULL", "It was a fine quarter. " * 300)
    with pytest.raises(ValueError, match="nothing to submit"):
        submit_batch([t], client=StubClient(), model="claude-opus-5")


def test_windows_merge_into_one_extraction_per_transcript():
    t = _transcript("NVDA-2020Q4", "NVDA", FILLER + GUIDANCE + FILLER)
    quote = "We expect revenue to be $3 billion plus or minus 2%."
    client = StubClient(
        [
            _result("NVDA-2020Q4#0", text=_extraction_json("NVDA", quote, 2.94, 3.06)),
            _result("NVDA-2020Q4#1", text=_extraction_json("NVDA", quote, 2.94, 3.06)),
        ]
    )
    results, failures = collect_results("batch_test", [t], client=client)

    assert failures == []
    assert len(results) == 1
    # Same figure from two overlapping windows is one figure, not two.
    assert len(results[0].extraction.items) == 1


def test_one_bad_window_keeps_the_rest_of_the_transcript():
    t = _transcript("NVDA-2020Q4", "NVDA", FILLER + GUIDANCE + FILLER)
    quote = "We expect revenue to be $3 billion plus or minus 2%."
    client = StubClient(
        [
            _result("NVDA-2020Q4#0", text="{not json at all"),
            _result("NVDA-2020Q4#1", text=_extraction_json("NVDA", quote, 2.94, 3.06)),
        ]
    )
    results, failures = collect_results("batch_test", [t], client=client)

    assert len(results) == 1
    assert len(results[0].extraction.items) == 1
    assert len(failures) == 1
    assert "window 0" in failures[0].error


def test_refusal_is_reported_not_parsed():
    t = _transcript("NVDA-2020Q4", "NVDA", FILLER + GUIDANCE + FILLER)
    client = StubClient([_result("NVDA-2020Q4#0", text=None, stop_reason="refusal")])
    results, failures = collect_results("batch_test", [t], client=client)

    assert results == []
    assert "declined" in failures[0].error


def test_unknown_custom_id_becomes_a_failure():
    t = _transcript("NVDA-2020Q4", "NVDA", FILLER + GUIDANCE + FILLER)
    client = StubClient([_result("WHO-2020Q1#0", text="{}")])
    results, failures = collect_results("batch_test", [t], client=client)

    assert results == []
    assert "no transcript matches" in failures[0].error


def test_fabricated_quote_is_dropped_after_merge():
    t = _transcript("NVDA-2020Q4", "NVDA", FILLER + GUIDANCE + FILLER)
    client = StubClient(
        [
            _result(
                "NVDA-2020Q4#0",
                text=_extraction_json("NVDA", "We expect revenue of $99 trillion.", 99, 99),
            )
        ]
    )
    results, failures = collect_results("batch_test", [t], client=client)

    assert failures == []
    assert len(results) == 1
    assert results[0].dropped_items == 1
    assert results[0].extraction.items == []
    assert results[0].extraction.no_guidance_given is False
