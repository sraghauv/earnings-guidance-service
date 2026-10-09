"""Anthropic Message Batches job for extracting guidance from many
transcripts at once, at 50% of the per-request price.

There's no repair loop inside a batch -- the model gets one shot per
transcript, since a batch request can't react to its own output. A
transcript whose result fails schema validation comes back in `failures`
instead of being retried; run it through `extract.pipeline.extract_guidance`
afterward if you want the repair loop.
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
)
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
            custom_id=t.id,
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[
                    {"role": "user", "content": build_user_prompt(full_text(t))}
                ],
            ),
        )
        for t in transcripts
    ]
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
    """Validate and persist every result. Batch results arrive in no
    particular order, so match on `custom_id`, never on position."""
    settings = get_settings()
    client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
    by_id = {t.id: t for t in transcripts}

    results: list[ExtractionResult] = []
    failures: list[BatchFailure] = []

    for result in client.messages.batches.results(batch_id):
        transcript = by_id.get(result.custom_id)
        if transcript is None:
            failures.append(
                BatchFailure(
                    transcript_id=result.custom_id,
                    error="no transcript matches this custom_id",
                )
            )
            continue

        if result.result.type != "succeeded":
            failures.append(
                BatchFailure(
                    transcript_id=transcript.id,
                    error=f"batch result was {result.result.type}",
                )
            )
            continue

        message = result.result.message
        text = next((b.text for b in message.content if b.type == "text"), None)
        if text is None:
            failures.append(
                BatchFailure(
                    transcript_id=transcript.id, error="no text block in response"
                )
            )
            continue

        try:
            extraction = GuidanceExtraction.model_validate_json(text)
        except ValidationError as e:
            failures.append(
                BatchFailure(
                    transcript_id=transcript.id,
                    error=f"schema validation failed: {e}",
                )
            )
            continue

        extraction, dropped = drop_fabricated_quotes(extraction, full_text(transcript))

        if conn is not None:
            db.save_extraction(
                conn,
                transcript.id,
                extraction,
                provider="anthropic",
                model=message.model,
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
