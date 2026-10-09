"""Run guidance extraction against hand-labeled gold files and score it.

For each gold file, pulls the matching transcript out of the parquet, runs
`extract_guidance`, and compares the result against the gold items. Prints a
per-figure table and aggregate recall/precision/value-accuracy, plus the
window count and character total `guidance_windows` produced -- windows are
what drive the per-transcript cost.

    python scripts/eval_extraction.py
    python scripts/eval_extraction.py --provider anthropic --no-cache
    python scripts/eval_extraction.py --json
"""

import argparse
import json
import sys
from pathlib import Path

from egs.api.deps import build_provider
from egs.config import get_settings
from egs.eval.score import GoldSet, Score, load_gold, score
from egs.extract.pipeline import RepairExhaustedError, extract_guidance, full_text
from egs.extract.windows import guidance_windows
from egs.ingest.parquet_loader import iter_transcripts
from egs.llm.base import ProviderError
from egs.schemas import Transcript

DEFAULT_GOLD_DIR = Path("tests/fixtures/eval")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--provider", choices=["ollama", "anthropic", "fixture"], default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _find_transcript(gold: GoldSet, parquet_path: Path) -> Transcript:
    for transcript in iter_transcripts(parquet_path, tickers=[gold.ticker]):
        if transcript.id == gold.transcript_id:
            return transcript
    raise RuntimeError(
        f"transcript {gold.transcript_id!r} not found for ticker {gold.ticker!r} "
        f"in {parquet_path}"
    )


def _print_table(gold_path: Path, result_score: Score) -> None:
    print(f"\n=== {gold_path.name} ===")
    print(f"{'verdict':<10} {'key':<55} {'gold':<28} {'got'}")
    for row in result_score.rows:
        key_str = f"{row.key[0]}/{row.key[1] or '-'}/{row.key[2]}/{row.key[3]}"
        print(f"{row.verdict:<10} {key_str:<55} {row.gold_range:<28} {row.got_range}")
    print(
        f"recall={result_score.recall:.2f} precision={result_score.precision:.2f} "
        f"value_accuracy={result_score.value_accuracy:.2f} "
        f"(matched={len(result_score.matched)} missed={len(result_score.missed)} "
        f"spurious={len(result_score.spurious)})"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    settings = get_settings()
    if args.provider is not None:
        settings = settings.model_copy(update={"llm_provider": args.provider})

    if not settings.transcripts_parquet:
        print("EGS_TRANSCRIPTS_PARQUET is not set", file=sys.stderr)
        return 1

    gold_paths = sorted(args.gold.glob("*.json"))
    if not gold_paths:
        print(f"no gold files found in {args.gold}", file=sys.stderr)
        return 1

    try:
        provider = build_provider(settings)
    except Exception as exc:
        print(f"could not build provider: {exc}", file=sys.stderr)
        return 1

    aggregate_matched = aggregate_missed = aggregate_spurious = 0
    aggregate_value_correct = aggregate_value_wrong = 0
    per_file: dict[str, dict] = {}

    for gold_path in gold_paths:
        gold = load_gold(gold_path)
        try:
            transcript = _find_transcript(gold, settings.transcripts_parquet)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        windows = guidance_windows(transcript)
        window_chars = sum(len(w) for w in windows)
        total_chars = len(full_text(transcript))

        try:
            result = extract_guidance(transcript, provider, use_cache=not args.no_cache)
        except (RepairExhaustedError, ProviderError) as exc:
            print(f"extraction failed for {transcript.id}: {exc}", file=sys.stderr)
            return 1

        file_score = score(result.extraction, gold)
        aggregate_matched += len(file_score.matched)
        aggregate_missed += len(file_score.missed)
        aggregate_spurious += len(file_score.spurious)
        aggregate_value_correct += file_score.value_correct
        aggregate_value_wrong += file_score.value_wrong

        per_file[gold_path.name] = {
            "transcript_id": transcript.id,
            "windows": len(windows),
            "window_chars": window_chars,
            "total_chars": total_chars,
            "recall": file_score.recall,
            "precision": file_score.precision,
            "value_accuracy": file_score.value_accuracy,
        }

        if not args.json:
            print(
                f"{gold_path.name}: {len(windows)} windows, "
                f"{window_chars}/{total_chars} chars ({window_chars / total_chars:.0%})"
            )
            _print_table(gold_path, file_score)

    matched_plus_missed = aggregate_matched + aggregate_missed
    matched_plus_spurious = aggregate_matched + aggregate_spurious
    value_total = aggregate_value_correct + aggregate_value_wrong
    aggregate = {
        "recall": aggregate_matched / matched_plus_missed if matched_plus_missed else 0.0,
        "precision": aggregate_matched / matched_plus_spurious if matched_plus_spurious else 0.0,
        "value_accuracy": aggregate_value_correct / value_total if value_total else 0.0,
        "matched": aggregate_matched,
        "missed": aggregate_missed,
        "spurious": aggregate_spurious,
        "files": per_file,
    }

    if args.json:
        print(json.dumps(aggregate, indent=2))
    else:
        print("\n=== aggregate ===")
        print(
            f"recall={aggregate['recall']:.2f} precision={aggregate['precision']:.2f} "
            f"value_accuracy={aggregate['value_accuracy']:.2f} "
            f"(matched={aggregate_matched} missed={aggregate_missed} "
            f"spurious={aggregate_spurious})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
