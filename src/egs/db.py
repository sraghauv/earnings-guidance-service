"""SQLite storage. Schema is applied idempotently on connect."""

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from egs.schemas import (
    GuidanceExtraction,
    GuidanceItem,
    Mismatch,
    Transcript,
    Valuation,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id              TEXT PRIMARY KEY,
    ticker          TEXT NOT NULL,
    company_name    TEXT NOT NULL,
    fiscal_year     INTEGER NOT NULL,
    fiscal_quarter  INTEGER NOT NULL,
    call_date       TEXT NOT NULL,
    prepared_remarks TEXT NOT NULL,
    qa              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_transcripts_ticker ON transcripts(ticker, call_date);

CREATE TABLE IF NOT EXISTS extractions (
    transcript_id   TEXT NOT NULL REFERENCES transcripts(id),
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    schema_version  TEXT NOT NULL,
    cache_key       TEXT NOT NULL,
    no_guidance     INTEGER NOT NULL,
    notes           TEXT,
    repair_attempts INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (transcript_id, cache_key)
);

CREATE TABLE IF NOT EXISTS guidance_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id   TEXT NOT NULL REFERENCES transcripts(id),
    cache_key       TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    metric          TEXT NOT NULL,
    metric_detail   TEXT,
    period_kind     TEXT NOT NULL,
    fiscal_year     INTEGER NOT NULL,
    fiscal_quarter  INTEGER,
    period_label    TEXT NOT NULL,
    low             REAL,
    high            REAL,
    unit            TEXT NOT NULL,
    basis           TEXT NOT NULL,
    quote           TEXT NOT NULL,
    speaker         TEXT,
    confidence      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_items_ticker_period
    ON guidance_items(ticker, period_label, metric);

CREATE TABLE IF NOT EXISTS mismatches (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    kind               TEXT NOT NULL,
    ticker             TEXT NOT NULL,
    metric             TEXT NOT NULL,
    period_label       TEXT NOT NULL,
    guided_value       REAL,
    comparison_value   REAL,
    relative_error     REAL,
    detail             TEXT NOT NULL,
    quote              TEXT,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_mismatches_ticker ON mismatches(ticker, kind);

CREATE TABLE IF NOT EXISTS valuations (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                 TEXT NOT NULL,
    period_label           TEXT NOT NULL,
    basis_metric           TEXT NOT NULL,
    guided_midpoint        REAL NOT NULL,
    multiple_used          REAL NOT NULL,
    multiple_kind          TEXT NOT NULL,
    implied_equity_value   REAL,
    implied_price_per_share REAL,
    current_price          REAL,
    upside_pct             REAL,
    peers_json             TEXT NOT NULL,
    created_at             TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_valuations_ticker ON valuations(ticker, period_label);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with the schema applied and foreign keys on.

    `check_same_thread=False` because FastAPI runs sync route handlers in a
    threadpool, so the connection opened during startup gets used from worker
    threads. Writes still go through one connection, and WAL plus SQLite's own
    serialized threading mode keep that safe.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


def upsert_transcript(conn: sqlite3.Connection, t: Transcript) -> None:
    conn.execute(
        """INSERT INTO transcripts
               (id, ticker, company_name, fiscal_year, fiscal_quarter,
                call_date, prepared_remarks, qa)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               prepared_remarks = excluded.prepared_remarks,
               qa = excluded.qa""",
        (
            t.id,
            t.ticker,
            t.company_name,
            t.fiscal_year,
            t.fiscal_quarter,
            t.call_date.isoformat(),
            t.prepared_remarks,
            t.qa,
        ),
    )


def save_extraction(
    conn: sqlite3.Connection,
    transcript_id: str,
    extraction: GuidanceExtraction,
    *,
    provider: str,
    model: str,
    prompt_version: str,
    schema_version: str,
    cache_key: str,
    repair_attempts: int = 0,
) -> None:
    """Persist one extraction and its items. Re-running the same cache_key
    replaces the prior items rather than duplicating them."""
    conn.execute(
        """INSERT INTO extractions
               (transcript_id, provider, model, prompt_version, schema_version,
                cache_key, no_guidance, notes, repair_attempts)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(transcript_id, cache_key) DO UPDATE SET
               no_guidance = excluded.no_guidance,
               notes = excluded.notes,
               repair_attempts = excluded.repair_attempts""",
        (
            transcript_id,
            provider,
            model,
            prompt_version,
            schema_version,
            cache_key,
            int(extraction.no_guidance_given),
            extraction.extraction_notes,
            repair_attempts,
        ),
    )
    conn.execute(
        "DELETE FROM guidance_items WHERE transcript_id = ? AND cache_key = ?",
        (transcript_id, cache_key),
    )
    conn.executemany(
        """INSERT INTO guidance_items
               (transcript_id, cache_key, ticker, metric, metric_detail,
                period_kind, fiscal_year, fiscal_quarter, period_label,
                low, high, unit, basis, quote, speaker, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                transcript_id,
                cache_key,
                extraction.ticker,
                item.metric.value,
                item.metric_detail,
                item.period.kind.value,
                item.period.fiscal_year,
                item.period.fiscal_quarter,
                item.period.label(),
                item.low,
                item.high,
                item.unit.value,
                item.basis.value,
                item.quote,
                item.speaker,
                item.confidence,
            )
            for item in extraction.items
        ],
    )
    conn.commit()


def save_mismatches(conn: sqlite3.Connection, rows: Iterable[Mismatch]) -> int:
    rows = list(rows)
    conn.executemany(
        """INSERT INTO mismatches
               (kind, ticker, metric, period_label, guided_value,
                comparison_value, relative_error, detail, quote)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                m.kind.value,
                m.ticker,
                m.metric.value,
                m.period_label,
                m.guided_value,
                m.comparison_value,
                m.relative_error,
                m.detail,
                m.quote,
            )
            for m in rows
        ],
    )
    conn.commit()
    return len(rows)


def save_valuation(conn: sqlite3.Connection, v: Valuation) -> None:
    conn.execute(
        """INSERT INTO valuations
               (ticker, period_label, basis_metric, guided_midpoint,
                multiple_used, multiple_kind, implied_equity_value,
                implied_price_per_share, current_price, upside_pct, peers_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            v.ticker,
            v.period_label,
            v.basis_metric.value,
            v.guided_midpoint,
            v.multiple_used,
            v.multiple_kind,
            v.implied_equity_value,
            v.implied_price_per_share,
            v.current_price,
            v.upside_pct,
            json.dumps(v.peers.model_dump()),
        ),
    )
    conn.commit()


def load_items(
    conn: sqlite3.Connection, ticker: str, period_label: str | None = None
) -> list[GuidanceItem]:
    """Read back guidance items as validated models."""
    sql = "SELECT * FROM guidance_items WHERE ticker = ?"
    params: list[object] = [ticker]
    if period_label:
        sql += " AND period_label = ?"
        params.append(period_label)
    rows = conn.execute(sql, params).fetchall()
    return [
        GuidanceItem(
            metric=r["metric"],
            metric_detail=r["metric_detail"],
            period={
                "kind": r["period_kind"],
                "fiscal_year": r["fiscal_year"],
                "fiscal_quarter": r["fiscal_quarter"],
            },
            low=r["low"],
            high=r["high"],
            unit=r["unit"],
            basis=r["basis"],
            quote=r["quote"],
            speaker=r["speaker"],
            confidence=r["confidence"],
        )
        for r in rows
    ]
