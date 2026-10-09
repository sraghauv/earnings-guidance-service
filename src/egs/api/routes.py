"""Route handlers and their response models.

Response shapes live here rather than in schemas.py: schemas.py is the
provider/storage contract, and most of what the API returns is either a
thin read of stored rows (no body text, paginated, etc.) or a "could not
compute, here's why" wrapper that the domain layer has no reason to know
about. Where an existing schemas.py model already is the right shape
(GuidanceItem, GuidanceExtraction, Mismatch, Valuation) we return it as-is
instead of re-declaring it.
"""

import sqlite3
from datetime import date
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from egs.api.deps import get_app_settings, get_av_client, get_conn, get_provider
from egs.config import Settings
from egs.llm.base import LLMProvider, ProviderError, SchemaViolation
from egs.schemas import (
    GuidanceExtraction,
    GuidanceItem,
    Metric,
    Mismatch,
    MismatchKind,
    Transcript,
    Valuation,
)

if TYPE_CHECKING:
    from egs.crosscheck.alphavantage import AlphaVantageClient

router = APIRouter()


# ---- healthz ----


class HealthResponse(BaseModel):
    status: str
    db_reachable: bool
    llm_provider: str


@router.get("/healthz", response_model=HealthResponse)
def healthz(
    conn: sqlite3.Connection = Depends(get_conn),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    try:
        conn.execute("SELECT 1")
        db_reachable = True
    except sqlite3.Error:
        db_reachable = False
    return HealthResponse(
        status="ok" if db_reachable else "degraded",
        db_reachable=db_reachable,
        llm_provider=settings.llm_provider,
    )


# ---- companies ----


class CompanyOut(BaseModel):
    ticker: str
    transcript_count: int


@router.get("/companies", response_model=list[CompanyOut])
def list_companies(conn: sqlite3.Connection = Depends(get_conn)) -> list[CompanyOut]:
    rows = conn.execute(
        "SELECT ticker, COUNT(*) AS n FROM transcripts GROUP BY ticker ORDER BY ticker"
    ).fetchall()
    return [CompanyOut(ticker=r["ticker"], transcript_count=r["n"]) for r in rows]


# ---- transcripts ----


class TranscriptOut(BaseModel):
    """Transcript metadata only — prepared_remarks/qa are deliberately omitted."""

    id: str
    ticker: str
    company_name: str
    fiscal_year: int
    fiscal_quarter: int
    call_date: date


def _row_to_transcript_out(row: sqlite3.Row) -> TranscriptOut:
    return TranscriptOut(
        id=row["id"],
        ticker=row["ticker"],
        company_name=row["company_name"],
        fiscal_year=row["fiscal_year"],
        fiscal_quarter=row["fiscal_quarter"],
        call_date=date.fromisoformat(row["call_date"]),
    )


@router.get("/transcripts", response_model=list[TranscriptOut])
def list_transcripts(
    ticker: str | None = None,
    year: int | None = None,
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[TranscriptOut]:
    sql = "SELECT id, ticker, company_name, fiscal_year, fiscal_quarter, call_date FROM transcripts"
    clauses = []
    params: list[object] = []
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    if year is not None:
        clauses.append("fiscal_year = ?")
        params.append(year)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY call_date"
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_transcript_out(r) for r in rows]


def _get_transcript_row(conn: sqlite3.Connection, transcript_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM transcripts WHERE id = ?", (transcript_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"no transcript stored with id {transcript_id!r}"
        )
    return row


def _row_to_guidance_item(row: sqlite3.Row) -> GuidanceItem:
    return GuidanceItem(
        metric=row["metric"],
        metric_detail=row["metric_detail"],
        period={
            "kind": row["period_kind"],
            "fiscal_year": row["fiscal_year"],
            "fiscal_quarter": row["fiscal_quarter"],
        },
        low=row["low"],
        high=row["high"],
        unit=row["unit"],
        basis=row["basis"],
        quote=row["quote"],
        speaker=row["speaker"],
        confidence=row["confidence"],
    )


class GuidanceResponse(BaseModel):
    transcript_id: str
    ticker: str
    items: list[GuidanceItem]


@router.get(
    "/transcripts/{transcript_id}/guidance", response_model=GuidanceResponse
)
def transcript_guidance(
    transcript_id: str, conn: sqlite3.Connection = Depends(get_conn)
) -> GuidanceResponse:
    transcript_row = _get_transcript_row(conn, transcript_id)
    rows = conn.execute(
        "SELECT * FROM guidance_items WHERE transcript_id = ?", (transcript_id,)
    ).fetchall()
    return GuidanceResponse(
        transcript_id=transcript_id,
        ticker=transcript_row["ticker"],
        items=[_row_to_guidance_item(r) for r in rows],
    )


# ---- extract ----


class ExtractRequest(BaseModel):
    transcript_id: str


@router.post("/extract", response_model=GuidanceExtraction)
def run_extract(
    body: ExtractRequest,
    conn: sqlite3.Connection = Depends(get_conn),
    provider: LLMProvider = Depends(get_provider),
) -> GuidanceExtraction:
    row = _get_transcript_row(conn, body.transcript_id)
    transcript = Transcript(
        id=row["id"],
        ticker=row["ticker"],
        company_name=row["company_name"],
        fiscal_year=row["fiscal_year"],
        fiscal_quarter=row["fiscal_quarter"],
        call_date=date.fromisoformat(row["call_date"]),
        prepared_remarks=row["prepared_remarks"],
        qa=row["qa"],
    )

    try:
        from egs.extract.pipeline import RepairExhaustedError, extract_guidance
    except ImportError as exc:
        raise HTTPException(
            status_code=502, detail=f"extraction pipeline unavailable: {exc}"
        ) from exc

    try:
        result = extract_guidance(transcript, provider, conn=conn, use_cache=True)
    except (ProviderError, SchemaViolation, RepairExhaustedError) as exc:
        raise HTTPException(
            status_code=502, detail=f"extraction failed for {body.transcript_id}: {exc}"
        ) from exc
    return result.extraction


# ---- valuation ----


class ValuationResult(BaseModel):
    ticker: str
    computed: bool
    valuation: Valuation | None = None
    reason: str | None = None


@router.get("/valuation/{ticker}", response_model=ValuationResult)
def get_valuation(
    ticker: str,
    period: str | None = None,
    shares_outstanding: float | None = Query(default=None, gt=0),
    conn: sqlite3.Connection = Depends(get_conn),
    av_client: "AlphaVantageClient" = Depends(get_av_client),
) -> ValuationResult:
    ticker = ticker.upper()
    sql = "SELECT * FROM guidance_items WHERE ticker = ? AND metric = ?"
    params: list[object] = [ticker, Metric.EPS.value]
    if period:
        sql += " AND period_label = ?"
        params.append(period)
    sql += " ORDER BY id DESC LIMIT 1"
    row = conn.execute(sql, params).fetchone()
    if row is None:
        detail = f"no EPS guidance stored for {ticker}"
        if period:
            detail += f" in period {period}"
        raise HTTPException(status_code=404, detail=detail)
    item = _row_to_guidance_item(row)

    try:
        from egs.valuation.multiples import value_from_guidance
        from egs.valuation.peers import peers_for
    except ImportError as exc:
        return ValuationResult(
            ticker=ticker, computed=False, reason=f"valuation backend unavailable: {exc}"
        )

    try:
        valuation = value_from_guidance(
            item, ticker=ticker, client=av_client, shares_outstanding=shares_outstanding
        )
    except Exception as exc:  # AlphaVantage/peer lookups are meant to degrade
        return ValuationResult(ticker=ticker, computed=False, reason=str(exc))

    if valuation is not None:
        return ValuationResult(ticker=ticker, computed=True, valuation=valuation)

    # value_from_guidance only ever logs why it gave up a None, so recover a
    # caller-facing reason ourselves from the one piece of that it's cheap
    # (static, no network) to re-derive: whether this ticker has any peers
    # mapped at all.
    sector, _industry, peer_tickers = peers_for(ticker)
    if not peer_tickers:
        reason = f"no peer set mapped for {ticker} (sector classified as {sector!r})"
    else:
        reason = (
            f"could not price {ticker} off its peers' trailing P/E -- AlphaVantage "
            "had no usable earnings/price data for them, or no current quote for "
            f"{ticker} itself"
        )
    return ValuationResult(ticker=ticker, computed=False, reason=reason)


# ---- mismatches ----


class MismatchesResponse(BaseModel):
    items: list[Mismatch]
    total: int
    limit: int
    offset: int


@router.get("/mismatches", response_model=MismatchesResponse)
def list_mismatches(
    ticker: str | None = None,
    kind: MismatchKind | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    conn: sqlite3.Connection = Depends(get_conn),
) -> MismatchesResponse:
    clauses = []
    params: list[object] = []
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    if kind:
        clauses.append("kind = ?")
        params.append(kind.value)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    total = conn.execute(
        f"SELECT COUNT(*) AS n FROM mismatches{where}", params
    ).fetchone()["n"]
    rows = conn.execute(
        f"SELECT * FROM mismatches{where} ORDER BY id LIMIT ? OFFSET ?",
        [*params, limit, offset],
    ).fetchall()

    items = [
        Mismatch(
            kind=r["kind"],
            ticker=r["ticker"],
            metric=r["metric"],
            period_label=r["period_label"],
            guided_value=r["guided_value"],
            comparison_value=r["comparison_value"],
            relative_error=r["relative_error"],
            detail=r["detail"],
            quote=r["quote"],
        )
        for r in rows
    ]
    return MismatchesResponse(items=items, total=total, limit=limit, offset=offset)
