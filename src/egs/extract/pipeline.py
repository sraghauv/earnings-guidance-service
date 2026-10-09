"""Validation, repair, caching and persistence around an `LLMProvider`.

A provider's only job is to hand back raw JSON text (see `llm.base`).
Everything that decides whether that text is usable lives here: schema
validation, retrying with a repair hint, dropping hallucinated quotes,
writing to disk, and caching.
"""

import sqlite3
from dataclasses import dataclass

from pydantic import ValidationError

from egs import db
from egs.config import get_settings
from egs.llm import cache
from egs.llm.base import SCHEMA_VERSION, LLMProvider, ProviderError
from egs.llm.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from egs.schemas import GuidanceExtraction, Transcript


class RepairExhaustedError(RuntimeError):
    """Provider never returned schema-valid JSON within the repair budget."""


@dataclass(frozen=True)
class ExtractionResult:
    extraction: GuidanceExtraction
    repair_attempts: int
    cache_hit: bool
    dropped_items: int


@dataclass(frozen=True)
class BatchFailure:
    transcript_id: str
    error: str


def full_text(transcript: Transcript) -> str:
    """The prepared remarks and Q&A concatenated -- what the model sees."""
    return f"{transcript.prepared_remarks}\n\n{transcript.qa}"


def _normalize(text: str) -> str:
    return " ".join(text.split())


def drop_fabricated_quotes(
    extraction: GuidanceExtraction, transcript_text: str
) -> tuple[GuidanceExtraction, int]:
    """Strip items whose quote doesn't actually occur in the transcript.

    Whitespace is normalized on both sides first -- the model routinely
    collapses line breaks or re-spaces a quote without changing a word of
    it, and that shouldn't count as a hallucination.
    """
    haystack = _normalize(transcript_text)
    kept = [item for item in extraction.items if _normalize(item.quote) in haystack]
    dropped = len(extraction.items) - len(kept)
    if dropped:
        extraction = extraction.model_copy(update={"items": kept})
    return extraction, dropped


def extract_guidance(
    transcript: Transcript,
    provider: LLMProvider,
    *,
    conn: sqlite3.Connection | None = None,
    use_cache: bool = True,
) -> ExtractionResult:
    settings = get_settings()
    transcript_text = full_text(transcript)
    user_prompt = build_user_prompt(transcript_text)
    key = provider.cache_key(transcript_text=user_prompt, prompt_version=PROMPT_VERSION)

    if use_cache:
        cached = cache.get(settings.cache_dir, key)
        if cached is not None:
            return ExtractionResult(
                extraction=GuidanceExtraction.model_validate(cached),
                repair_attempts=0,
                cache_hit=True,
                dropped_items=0,
            )

    schema = GuidanceExtraction.model_json_schema()
    repair_hint: str | None = None
    extraction: GuidanceExtraction | None = None
    attempts = 0

    while extraction is None:
        completion = provider.complete_json(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=schema,
            repair_hint=repair_hint,
        )
        try:
            extraction = GuidanceExtraction.model_validate_json(completion.text)
        except ValidationError as e:
            if attempts >= settings.max_repair_attempts:
                raise RepairExhaustedError(
                    f"transcript {transcript.id}: no valid extraction after "
                    f"{attempts} repair attempts, last error: {e}"
                ) from e
            attempts += 1
            repair_hint = str(e)

    extraction, dropped = drop_fabricated_quotes(extraction, transcript_text)

    if conn is not None:
        db.save_extraction(
            conn,
            transcript.id,
            extraction,
            provider=provider.name,
            model=provider.model,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            cache_key=key,
            repair_attempts=attempts,
        )

    if use_cache:
        cache.put(settings.cache_dir, key, extraction.model_dump(mode="json"))

    return ExtractionResult(
        extraction=extraction,
        repair_attempts=attempts,
        cache_hit=False,
        dropped_items=dropped,
    )


def extract_many(
    transcripts: list[Transcript],
    provider: LLMProvider,
    *,
    conn: sqlite3.Connection | None = None,
    use_cache: bool = True,
) -> tuple[list[ExtractionResult], list[BatchFailure]]:
    """Like `extract_guidance`, but one bad transcript doesn't sink the run."""
    results = []
    failures = []
    for transcript in transcripts:
        try:
            results.append(
                extract_guidance(transcript, provider, conn=conn, use_cache=use_cache)
            )
        except (RepairExhaustedError, ProviderError) as e:
            failures.append(BatchFailure(transcript_id=transcript.id, error=str(e)))
    return results, failures
