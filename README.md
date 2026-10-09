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

## Windowing

Feeding a whole call to the model does not work, and the failure is quiet: it
reports no guidance at all rather than erroring. gemma3:4b returns
`no_guidance_given` on a full NVDA call and then pulls all eight figures out of
that same call's outlook paragraph. A transcript is about 12k tokens and the
guidance occupies a few hundred of them.

So `extract/windows.py` narrows the call first, from outlook cues where they
exist and from ranked overlapping chunks where they don't, and the pipeline
merges per-window results keyed on metric, detail, period and basis. Across 40
real calls that is 4.2 windows each and 30% of the original text. Overlap means
the same figure arrives more than once, and keying the merge on identity rather
than value keeps that from reading downstream as the company contradicting
itself.

## How good is the extraction

`scripts/eval_extraction.py` scores an extraction against hand-labeled calls in
`tests/fixtures/eval/`. On NVDA-2020Q4 with gemma3:4b:

```
recall=0.20  precision=0.40  value_accuracy=0.50   (matched=2 missed=8 spurious=3)
```

That is a usable floor, not a good result, and worth being plain about: a 4B
model gets the easy figures right — revenue $3B ±2% resolves correctly to
2.94-3.06, capex to $150-170M — and then mislabels basis, assigns GAAP's gross
margin range to non-GAAP, and misses OI&E entirely. The structural guarantees
hold regardless of model: the schema rejects malformed output, unquoted figures
get dropped, and the reconciliation pass still catches what survives.

One labeled call is a very small sample, and runs vary a little even at
temperature 0. The `claude-opus-5` provider is wired up but has not been
measured, since that needs an API key.

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
.venv/bin/pytest                             # no keys, no network
python scripts/eval_extraction.py            # score against the labeled calls
```

## Still to do

- Label more than one call. The harness exists and the numbers above are real,
  but a single transcript is not a benchmark.
- Measure `claude-opus-5` against the same gold set, which needs an API key.
- Segment-level revenue guidance is extracted but not yet reconciled, since
  segment actuals aren't in the free endpoints.
