"""Tests for llm/anthropic_provider.py against a stubbed client.

There's no API key on this machine, so none of this reaches Anthropic. What
these pin down is the request shape and the degradation path: that `fallbacks`
goes out by default, that a key without the beta still gets its extraction,
and that an unrelated 400 isn't retried into a second wasted request.
"""

import httpx
import pytest

import anthropic
from egs.llm.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from egs.llm.base import ProviderError

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


class FakeMessage:
    def __init__(self, text="{}", stop_reason="end_turn", stop_details=None):
        self.content = (
            [type("Block", (), {"type": "text", "text": text})()] if text is not None else []
        )
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.model = "claude-opus-5"
        self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()


def _bad_request(message: str) -> anthropic.BadRequestError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request, json={"error": {"message": message}})
    return anthropic.BadRequestError(message, response=response, body=None)


class StubMessages:
    """Records each call's kwargs and replays a queued outcome per call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else FakeMessage()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class StubClient:
    def __init__(self, outcomes=()):
        self.messages = StubMessages(outcomes)
        # The provider must use the beta namespace -- `fallbacks` is not a
        # parameter on plain client.messages.create.
        self.beta = type("Beta", (), {"messages": self.messages})()


def _provider(client):
    return AnthropicProvider(model="claude-opus-5", client=client)


def test_fallbacks_are_sent_by_default():
    client = StubClient([FakeMessage('{"ok": true}')])
    result = _provider(client).complete_json(system="s", user="u", schema=SCHEMA)

    assert result.text == '{"ok": true}'
    call = client.messages.calls[0]
    assert call["fallbacks"] == "default"
    assert call["betas"] == [FALLBACK_BETA]


def test_request_carries_the_schema_and_adaptive_thinking():
    client = StubClient()
    _provider(client).complete_json(system="s", user="u", schema=SCHEMA)

    call = client.messages.calls[0]
    assert call["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert call["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in str(call["thinking"])


def test_a_key_without_the_beta_still_gets_its_extraction():
    """The fallback is a safety net, not a requirement to run."""
    client = StubClient(
        [_bad_request("fallbacks is not supported"), FakeMessage('{"ok": true}')]
    )
    result = _provider(client).complete_json(system="s", user="u", schema=SCHEMA)

    assert result.text == '{"ok": true}'
    assert len(client.messages.calls) == 2
    assert "fallbacks" not in client.messages.calls[1]
    assert "betas" not in client.messages.calls[1]


def test_rejection_latches_so_later_calls_skip_the_doomed_attempt():
    client = StubClient(
        [
            _bad_request("fallbacks is not supported"),
            FakeMessage('{"first": true}'),
            FakeMessage('{"second": true}'),
        ]
    )
    provider = _provider(client)
    provider.complete_json(system="s", user="u", schema=SCHEMA)
    provider.complete_json(system="s", user="u2", schema=SCHEMA)

    # Three calls total, not four -- the second extraction doesn't re-try the
    # beta it already learned is unavailable.
    assert len(client.messages.calls) == 3
    assert "fallbacks" not in client.messages.calls[2]


def test_unrelated_bad_request_is_not_retried():
    """A malformed request retried unchanged just buries the real error."""
    client = StubClient([_bad_request("messages: at least one message is required")])
    with pytest.raises(ProviderError, match="rejected the request"):
        _provider(client).complete_json(system="s", user="u", schema=SCHEMA)

    assert len(client.messages.calls) == 1


def test_refusal_raises_rather_than_reading_empty_content():
    details = type("Details", (), {"category": "cyber"})()
    client = StubClient([FakeMessage(text=None, stop_reason="refusal", stop_details=details)])
    with pytest.raises(ProviderError, match="declined"):
        _provider(client).complete_json(system="s", user="u", schema=SCHEMA)


def test_truncated_response_raises_instead_of_returning_broken_json():
    client = StubClient([FakeMessage('{"items": [', stop_reason="max_tokens")])
    with pytest.raises(ProviderError, match="token cap"):
        _provider(client).complete_json(system="s", user="u", schema=SCHEMA)


def test_repair_replays_the_prior_response_as_history_not_a_prefill():
    """Prefill 400s on claude-opus-5, so the retry has to be a full turn."""
    client = StubClient([FakeMessage('{"bad": true}'), FakeMessage('{"good": true}')])
    provider = _provider(client)
    provider.complete_json(system="s", user="u", schema=SCHEMA)
    provider.complete_json(system="s", user="u", schema=SCHEMA, repair_hint="low > high")

    messages = client.messages.calls[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == '{"bad": true}'
    assert "low > high" in messages[2]["content"]
