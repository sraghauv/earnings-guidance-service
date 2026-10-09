# earnings-guidance-service

Pulls forward guidance out of earnings call transcripts with an LLM, checks the
extracted numbers against what the company actually reported later, and runs a
peer-multiple valuation off the guidance midpoint.

The corpus is [kurry/sp500_earnings_transcripts](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts)
— 33,362 calls across 685 companies, 2005 to 2025.

## Why the schema does the work

The naive version of this is "ask the model for guidance, parse the JSON, hope."
That fails in boring ways: the model returns the quarter being *reported* instead
of the quarter being *guided*, mixes millions and billions in the same response,
or invents a number that was never said on the call.

So the schema is the contract, not the prompt. `GuidanceExtraction` in
`schemas.py` is a Pydantic model handed to the API as a JSON schema, and three
things get enforced on the way back:

1. **Structured output.** The response is constrained to the schema, so a
   malformed shape can't reach the database. When validation does fail, the
   pydantic error goes back to the model as a repair hint — up to
   `max_repair_attempts` times, then the extraction is recorded as failed
   rather than silently dropped.
2. **Every figure carries its quote.** An item whose `quote` doesn't appear
   verbatim in the transcript is discarded before it's persisted. This is the
   hallucination guard, and it catches more than the schema does.
3. **Ranges have to make sense.** `low > high` is rejected at the model layer,
   and a one-sided bound ("at least $2 billion") is representable instead of
   being forced into a fake range.

## Cross-checking

Guidance is a claim about the future, which means it eventually becomes
checkable. `crosscheck/reconcile.py` compares guided EPS against the
`reportedEPS` the company later filed and flags anything outside a 2% tolerance,
alongside internal checks (impossible units, margins outside 0-100, a quarter
number attached to a full-year period) and conflicts between two extractions
for the same metric and period.

The fiscal-period mapping is the part that actually takes care. A fiscal year is
named for the calendar year it closes in, so NVDA's fiscal 2026 ran February
2025 through January 2026 and the quarter ending 2026-07-31 is already fiscal
2027 Q2. Compare guidance to reported results by calendar date instead and you
quietly line up quarters a year apart. The mapping works off fiscal year-end
dates and is tested against January, June and December year-ends.

Where a comparison can't be made — period not yet reported, no EPS on file —
nothing is emitted. A false mismatch is worse than a missed one.

## What the free API tier actually allows

Worth stating plainly, because it shaped the design. On a free AlphaVantage key
only `EARNINGS` and `GLOBAL_QUOTE` respond; `COMPANY_OVERVIEW` and
`INCOME_STATEMENT` return a premium notice with an HTTP 200, which is its own
trap — a rate-limit body parsed as data looks like a company with no revenue.
The client raises on those bodies instead.

Consequences:

- **P/E is the only multiple computable end to end on a free key**, from
  `GLOBAL_QUOTE` price over trailing four-quarter reported EPS.
- P/S and EV/EBITDA are implemented but return `None` without a premium key,
  and say why rather than guessing.
- Sector data is premium-gated too, so peer sets come from a static map in
  `valuation/peers.py` rather than being derived.

Peer multiples use the median, not the mean — one peer at a 300x P/E shouldn't
move the comparison.

## Providers

Extraction runs against either backend behind one interface:

- `ollama` (default) — `gemma3:4b` locally, so the whole pipeline runs for free
- `anthropic` — `claude-opus-5` with structured outputs; bulk runs go through
  the Batches API at half cost
- `fixture` — replays recorded responses, which is what lets CI run with no
  keys and no network

Extractions are cached by `sha256(provider + model + prompt_version +
schema_version + transcript)`, so reruns are free but editing the prompt
correctly invalidates everything it touched.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env          # point EGS_TRANSCRIPTS_PARQUET at the parquet
ollama pull gemma3:4b

python scripts/build_db.py --tickers NVDA,AMD --limit 20
.venv/bin/uvicorn egs.api.main:app --reload
```

```
GET  /companies
GET  /transcripts?ticker=NVDA
GET  /transcripts/{id}/guidance
POST /extract                     {"transcript_id": "NVDA-2025Q2"}
GET  /valuation/NVDA?period=FY2026Q3
GET  /mismatches?ticker=NVDA&kind=vs_reported
```

```bash
.venv/bin/pytest          # no keys, no network
```

## Still to do

- Hand-label a sample of calls and score extraction precision/recall against
  it. Right now correctness rests on the quote check and the reconciliation
  pass, which is evidence but not a measurement.
- Segment-level revenue guidance is extracted but not yet reconciled, since
  segment actuals aren't in the free endpoints.
