"""Streams Transcript rows out of the sp500 earnings-call parquet.

The source file is 1.7GB; nothing here loads it whole. `iter_transcripts`
scans the parquet with pyarrow's dataset API, which prunes row groups against
`tickers`/`years` using parquet statistics and decodes only the columns we
need, batch by batch.
"""

import logging
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

from egs.ingest.normalize import split_call, strip_boilerplate
from egs.schemas import Transcript

logger = logging.getLogger(__name__)

_COLUMNS = ["symbol", "quarter", "year", "date", "content", "structured_content", "company_name"]
_BATCH_SIZE = 256  # transcripts carry full call text; keep batches small


def available_tickers(path: Path | str) -> list[str]:
    """Distinct symbols, reading only the symbol column."""
    table = ds.dataset(path, format="parquet").to_table(columns=["symbol"])
    return sorted(table.column("symbol").unique().to_pylist())


def iter_transcripts(
    path: Path | str,
    *,
    tickers: list[str] | None = None,
    years: list[int] | None = None,
    limit: int | None = None,
) -> Iterator[Transcript]:
    """Stream Transcript rows out of the parquet.

    Rows with a missing or malformed `date` are skipped (and logged) rather
    than raised on, since a handful of bad rows shouldn't kill the whole scan.
    """
    filter_expr = None
    if tickers:
        filter_expr = pc.field("symbol").isin(list(tickers))
    if years:
        year_expr = pc.field("year").isin(list(years))
        filter_expr = year_expr if filter_expr is None else filter_expr & year_expr

    scanner = ds.dataset(path, format="parquet").scanner(
        columns=_COLUMNS, filter=filter_expr, batch_size=_BATCH_SIZE
    )

    yielded = 0
    skipped_dates = 0
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            if limit is not None and yielded >= limit:
                logger.info("iter_transcripts: hit limit=%d, skipped %d bad dates", limit, skipped_dates)
                return

            call_date = _parse_date(row["date"])
            if call_date is None:
                skipped_dates += 1
                logger.warning(
                    "skipping %s-%sQ%s: unparseable date %r",
                    row["symbol"], row["year"], row["quarter"], row["date"],
                )
                continue

            prepared, qa = split_call(row["structured_content"], row["content"])
            yield Transcript(
                id=f"{row['symbol']}-{row['year']}Q{row['quarter']}",
                ticker=row["symbol"],
                company_name=row["company_name"],
                fiscal_year=row["year"],
                fiscal_quarter=row["quarter"],
                call_date=call_date,
                prepared_remarks=strip_boilerplate(prepared),
                qa=strip_boilerplate(qa),
            )
            yielded += 1

    if skipped_dates:
        logger.info("iter_transcripts: skipped %d rows with bad dates", skipped_dates)


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None
