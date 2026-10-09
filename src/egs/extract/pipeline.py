"""Validation, repair, caching and persistence around an `LLMProvider`.

A provider's only job is to hand back raw JSON text (see `llm.base`).
Everything that decides whether that text is usable lives here: schema
validation, retrying with a repair hint, dropping hallucinated quotes,
writing to disk, and caching.
"""

import json
import sqlite3
from dataclasses import dataclass

from pydantic import ValidationError

from egs import db
from egs.config import get_settings
from egs.llm import cache
from egs.llm.base import SCHEMA_VERSION, LLMProvider, ProviderError
from egs.extract.windows import guidance_windows
from egs.llm.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from egs.schemas import GuidanceExtraction, GuidanceItem, Transcript


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


def parse_lenient(text: str) -> tuple[GuidanceExtraction, int]:
    """Validate a response, dropping individual items that don't hold up.

    Validating the whole payload at once means one bad item discards every
    good one with it. Microsoft's December-quarter call is the case that forced
    this: the model returned eleven items, four of them growth rates with no
    numeric bound, and all eleven were lost. Items are independent figures, so
    they are validated independently -- the same way a fabricated quote drops
    one item rather than failing the call.

    Returns (extraction, dropped). Raises ValidationError when the payload
    isn't usable at all, which is what the repair loop is for.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValidationError.from_exception_data("GuidanceExtraction", []) from e
    if not isinstance(payload, dict):
        raise ValidationError.from_exception_data("GuidanceExtraction", [])

    kept: list[GuidanceItem] = []
    dropped = 0
    for raw in payload.get("items") or []:
        try:
            kept.append(GuidanceItem.model_validate(raw))
        except ValidationError:
            dropped += 1

    extraction = GuidanceExtraction(
        ticker=str(payload.get("ticker") or ""),
        # The model sometimes sets this alongside items; the items win.
        no_guidance_given=bool(payload.get("no_guidance_given")) and not kept,
        items=kept,
        extraction_notes=payload.get("extraction_notes"),
    )
    return extraction, dropped


def _extract_window(
    window: str,
    provider: LLMProvider,
    schema: dict,
    *,
    max_repair: int,
) -> tuple[GuidanceExtraction, int]:
    """Extract one window, repairing schema failures up to `max_repair` times.

    A response with some usable items is accepted even if others were dropped.
    Repair is for a response with nothing usable in it.
    """
    repair_hint: str | None = None
    attempts = 0
    while True:
        completion = provider.complete_json(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(window),
            schema=schema,
            repair_hint=repair_hint,
        )
        try:
            extraction, dropped = parse_lenient(completion.text)
        except ValidationError as e:
            failure: Exception = e
        else:
            if extraction.items or not dropped:
                return extraction, attempts
            failure = ValueError(
                f"all {dropped} items failed validation, nothing usable"
            )

        if attempts >= max_repair:
            raise RepairExhaustedError(
                f"no valid extraction after {attempts} repair attempts, "
                f"last error: {failure}"
            ) from failure
        attempts += 1
        repair_hint = str(failure)


def merge_windows(
    extractions: list[GuidanceExtraction], *, ticker: str
) -> GuidanceExtraction:
    """Collapse per-window extractions into one.

    Windows overlap, so the same figure legitimately arrives more than once.
    Items are keyed on what identifies a figure rather than on its value --
    metric, detail, period and basis -- and the highest-confidence parse wins,
    breaking ties toward the longer quote as the better-supported one. Keeping
    both parses of one figure would otherwise read downstream as a genuine
    disagreement in the company's guidance, which it isn't.
    """
    best: dict[tuple, GuidanceItem] = {}
    notes: list[str] = []
    for extraction in extractions:
        if extraction.extraction_notes:
            notes.append(extraction.extraction_notes)
        for item in extraction.items:
            key = (item.metric, item.metric_detail, item.period.label(), item.basis)
            current = best.get(key)
            if current is None or (item.confidence, len(item.quote)) > (
                current.confidence,
                len(current.quote),
            ):
                best[key] = item

    items = list(best.values())
    return GuidanceExtraction(
        ticker=ticker,
        no_guidance_given=not items,
        items=items,
        extraction_notes="; ".join(notes) or None,
    )


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
    windows = guidance_windows(transcript)
    per_window: list[GuidanceExtraction] = []
    attempts = 0
    last_error: Exception | None = None

    for window in windows:
        try:
            got, used = _extract_window(
                window, provider, schema, max_repair=settings.max_repair_attempts
            )
        except (RepairExhaustedError, ProviderError) as e:
            # One unreadable passage shouldn't discard the rest of the call.
            last_error = e
            continue
        attempts += used
        per_window.append(got)

    if windows and not per_window:
        raise RepairExhaustedError(
            f"transcript {transcript.id}: every one of {len(windows)} windows "
            f"failed, last error: {last_error}"
        )

    extraction = merge_windows(per_window, ticker=transcript.ticker)
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
