"""Anthropic Message Batches job for extracting guidance from many
transcripts at once, at 50% of the per-request price.

One request per guidance window, not per transcript, matching what
`extract.pipeline` does interactively -- a whole call is too much context and
comes back empty. `custom_id` is "<transcript id>#<window index>" and the
per-window results are merged back per transcript on collection.

No refusal fallbacks here either, unlike the interactive provider -- the
`fallbacks` parameter is rejected on the Batches API. A window the classifiers
decline comes back in `failures` and has to be re-run through
`extract_guidance` to get the fallback.

There's no repair loop inside a batch: a batch request can't react to its own
output. A window whose result fails schema validation is dropped and noted in
`failures`, and the transcript still keeps whatever its other windows
returned. Re-run a transcript through `extract_guidance` if you want repair.
"""

import sqlite3
import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import ValidationError

from egs import db
from egs.config import get_settings
from egs.extract.pipeline import (
    BatchFailure,
    ExtractionResult,
    drop_fabricated_quotes,
    full_text,
    merge_windows,
    parse_lenient,
)
from egs.extract.windows import guidance_windows
from egs.llm.anthropic_provider import MAX_TOKENS
from egs.llm.base import SCHEMA_VERSION
from egs.llm.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from egs.schemas import GuidanceExtraction, Transcript

POLL_INTERVAL_SECONDS = 30.0


def submit_batch(
    transcripts: list[Transcript],
    *,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
) -> str:
    """Create the batch and return its id. Does not wait for completion."""
    settings = get_settings()
    client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
    model = model or settings.anthropic_model
    schema = GuidanceExtraction.model_json_schema()

    requests = [
        Request(
            custom_id=f"{t.id}#{i}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": build_user_prompt(window)}],
            ),
        )
        for t in transcripts
        for i, window in enumerate(guidance_windows(t))
    ]
    if not requests:
        raise ValueError(
            f"none of the {len(transcripts)} transcripts produced a guidance "
            "window, so there is nothing to submit"
        )
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def wait_for_batch(
    batch_id: str,
    *,
    client: anthropic.Anthropic | None = None,
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> None:
    """Block until the batch has finished processing (not necessarily
    succeeded -- `processing_status == "ended"` covers mixed success too)."""
    settings = get_settings()
    client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return
        time.sleep(poll_interval)


def collect_results(
    batch_id: str,
    transcripts: list[Transcript],
    *,
    client: anthropic.Anthropic | None = None,
    conn: sqlite3.Connection | None = None,
) -> tuple[list[ExtractionResult], list[BatchFailure]]:
    """Validate per-window results, merge them per transcript, and persist.

    Batch results arrive in no particular order, so everything is keyed on
    `custom_id` and grouped afterward -- never matched by position.
    """
    settings = get_settings()
    client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
    by_id = {t.id: t for t in transcripts}

    per_transcript: dict[str, list[GuidanceExtraction]] = {}
    failures: list[BatchFailure] = []
    model_used: dict[str, str] = {}
    dropped_on_parse: dict[str, int] = {}

    for result in client.messages.batches.results(batch_id):
        transcript_id, _, window_index = result.custom_id.partition("#")
        transcript = by_id.get(transcript_id)
        if transcript is None:
            failures.append(
                BatchFailure(
                    transcript_id=result.custom_id,
                    error="no transcript matches this custom_id",
                )
            )
            continue

        where = f"window {window_index}" if window_index else "whole transcript"

        if result.result.type != "succeeded":
            failures.append(
                BatchFailure(
                    transcript_id=transcript.id,
                    error=f"{where}: batch result was {result.result.type}",
                )
            )
            continue

        message = result.result.message
        if message.stop_reason == "refusal":
            failures.append(
                BatchFailure(
                    transcript_id=transcript.id, error=f"{where}: model declined"
                )
            )
            continue

        text = next((b.text for b in message.content if b.type == "text"), None)
        if text is None:
            failures.append(
                BatchFailure(
                    transcript_id=transcript.id,
                    error=f"{where}: no text block in response",
                )
            )
            continue

        try:
            # Same leniency as the interactive path: one unusable item drops
            # itself, not the whole window. There's no repair loop here to
            # recover from an all-or-nothing rejection, so whole-payload
            # validation cost strictly more in a batch than interactively.
            extraction, item_drops = parse_lenient(text)
        except ValidationError as e:
            failures.append(
                BatchFailure(
                    transcript_id=transcript.id,
                    error=f"{where}: schema validation failed: {e}",
                )
            )
            continue

        if item_drops:
            failures.append(
                BatchFailure(
                    transcript_id=transcript.id,
                    error=f"{where}: dropped {item_drops} unusable item(s)",
                )
            )
        dropped_on_parse[transcript.id] = (
            dropped_on_parse.get(transcript.id, 0) + item_drops
        )
        per_transcript.setdefault(transcript.id, []).append(extraction)
        model_used[transcript.id] = message.model

    results: list[ExtractionResult] = []
    for transcript_id, extractions in per_transcript.items():
        transcript = by_id[transcript_id]
        extraction = merge_windows(extractions, ticker=transcript.ticker)
        extraction, quote_drops = drop_fabricated_quotes(
            extraction, full_text(transcript)
        )
        dropped = quote_drops + dropped_on_parse.get(transcript_id, 0)

        if conn is not None:
            db.save_extraction(
                conn,
                transcript.id,
                extraction,
                provider="anthropic",
                model=model_used.get(transcript.id, "unknown"),
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
                cache_key=f"batch:{batch_id}:{transcript.id}",
                repair_attempts=0,
            )

        results.append(
            ExtractionResult(
                extraction=extraction,
                repair_attempts=0,
                cache_hit=False,
                dropped_items=dropped,
            )
        )

    return results, failures
