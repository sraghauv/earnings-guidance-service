"""Build (or add to) the EGS database from the transcript parquet.

Streams transcripts for the given tickers, runs guidance extraction on
each, reconciles the results against AlphaVantage where a key is
configured, and persists everything to EGS_DB_PATH.

    python scripts/build_db.py --tickers NVDA,AMD --limit 50
    python scripts/build_db.py --tickers NVDA --batch   # anthropic batch API
"""

import argparse
import sys

from egs.api.deps import build_av_client, build_provider
from egs.config import get_settings
from egs.db import load_items, save_mismatches, save_valuation, upsert_transcript
from egs.extract import pipeline
from egs.llm.base import ProviderError
from egs.schemas import Metric


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers", required=True, help="comma-separated tickers, e.g. NVDA,AMD"
    )
    parser.add_argument(
        "--years", default=None, help="comma-separated fiscal years to include"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--batch",
        action="store_true",
        help="submit extraction as an anthropic message batch instead of "
        "one request per transcript (see extract/batch_runner.py)",
    )
    return parser.parse_args(argv)


def _extract_one_by_one(transcripts, provider, conn) -> list:
    """Run extract_guidance per transcript; one bad transcript doesn't stop
    the rest. Doesn't use extract_many directly because its results drop
    which transcript each one came from -- we only need that for logging
    here, since reconciliation keys off GuidanceExtraction.ticker instead."""
    results = []
    for transcript in transcripts:
        try:
            result = pipeline.extract_guidance(
                transcript, provider, conn=conn, use_cache=True
            )
        except (pipeline.RepairExhaustedError, ProviderError) as exc:
            print(f"extraction failed for {transcript.id}: {exc}", file=sys.stderr)
            continue
        results.append(result)
    return results


def _extract_via_batch(transcripts, conn) -> list:
    from egs.extract.batch_runner import collect_results, submit_batch, wait_for_batch

    print(f"submitting batch of {len(transcripts)} transcripts...", file=sys.stderr)
    batch_id = submit_batch(transcripts)
    print(f"batch {batch_id} submitted, waiting for it to finish...", file=sys.stderr)
    wait_for_batch(batch_id)
    results, failures = collect_results(batch_id, transcripts, conn=conn)
    for failure in failures:
        print(f"extraction failed for {failure.transcript_id}: {failure.error}", file=sys.stderr)
    return results


def _reconcile_and_save(results, conn, av_client) -> None:
    """Reconcile every extraction, with or without AlphaVantage.

    A missing client only costs the comparison against reported actuals. The
    internal, duplicate and segment-sum checks still run.
    """
    from egs.crosscheck.reconcile import reconcile

    for result in results:
        if not result.extraction.items:
            continue
        ticker = result.extraction.ticker
        try:
            mismatches = reconcile(result.extraction.items, client=av_client, ticker=ticker)
        except Exception as exc:
            print(f"reconcile failed for {ticker}: {exc}", file=sys.stderr)
            continue
        if mismatches:
            n = save_mismatches(conn, mismatches)
            print(f"{ticker}: {n} mismatches saved", file=sys.stderr)


def _value_and_save(results, conn, av_client) -> None:
    """Price the stored EPS guidance and keep each valuation.

    Nothing wrote the valuations table before this -- `save_valuation` had no
    caller at all, so the table and its index were permanently empty while the
    pipeline advertised a peer-multiple valuation. Needs AlphaVantage for the
    peers' prices and earnings, so it is skipped without a client.

    A stored row is a snapshot: the implied price depends on the quote at the
    time, which is why the table records `created_at` and why this appends
    rather than replacing.
    """
    if av_client is None:
        return
    from egs.valuation.multiples import value_from_guidance

    for ticker in sorted({r.extraction.ticker for r in results}):
        eps_items = [i for i in load_items(conn, ticker) if i.metric is Metric.EPS]
        if not eps_items:
            continue
        saved = 0
        for item in eps_items:
            try:
                valuation = value_from_guidance(item, ticker=ticker, client=av_client)
            except Exception as exc:
                print(f"valuation failed for {ticker}: {exc}", file=sys.stderr)
                continue
            if valuation is not None:
                save_valuation(conn, valuation)
                saved += 1
        if saved:
            print(f"{ticker}: {saved} valuations saved", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    settings = get_settings()

    if not settings.transcripts_parquet:
        print("EGS_TRANSCRIPTS_PARQUET is not set", file=sys.stderr)
        return 1

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("--tickers produced an empty list", file=sys.stderr)
        return 1
    years = [int(y) for y in args.years.split(",")] if args.years else None

    from egs.db import connect
    from egs.ingest.parquet_loader import available_tickers, iter_transcripts

    conn = connect(settings.db_path)
    provider = build_provider(settings)

    try:
        av_client = build_av_client(settings)
    except Exception as exc:
        print(
            f"no AlphaVantage client ({exc}) -- running the offline checks only, "
            "without the comparison against reported actuals",
            file=sys.stderr,
        )
        av_client = None

    transcripts = list(
        iter_transcripts(settings.transcripts_parquet, tickers=tickers, years=years, limit=args.limit)
    )
    if not transcripts:
        # Name the ones that don't exist. A typo otherwise reads as "the
        # corpus has nothing for you" after a slow full scan.
        known = set(available_tickers(settings.transcripts_parquet))
        unknown = [t for t in tickers if t not in known]
        if unknown:
            print(
                f"no such ticker in the corpus: {', '.join(unknown)}", file=sys.stderr
            )
        else:
            print(
                f"no transcripts matched {tickers} with the given years/limit",
                file=sys.stderr,
            )
        return 1

    print(f"ingesting {len(transcripts)} transcripts", file=sys.stderr)
    for transcript in transcripts:
        upsert_transcript(conn, transcript)

    if args.batch:
        results = _extract_via_batch(transcripts, conn)
    else:
        results = _extract_one_by_one(transcripts, provider, conn)

    print(f"extracted {len(results)}/{len(transcripts)} transcripts", file=sys.stderr)
    _reconcile_and_save(results, conn, av_client)
    _value_and_save(results, conn, av_client)

    conn.close()
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
