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
for the same figure and period.

One error is rejected outright rather than flagged. A quarter with no quarter
number labels itself "FY2016" — the same label a legitimate full-year period
produces — and that label is what the stored row, the duplicate key, the eval
match key and the segment grouping all key on, so the quarter silently counts
as a fiscal year everywhere downstream. Seven of thirty-five figures in a real
run arrived that way. `FiscalPeriod` refuses to build one now, so the item is
dropped instead of persisted under the wrong period, and the reconcile check
for it stays as a net for anything reaching that code by another path.

Two unit errors get caught the same way, both found by extracting real calls
rather than reasoned about in advance. An amount guided as a percentage is the
model capturing a growth rate instead of a level ("services revenue up 26%"),
and a dollar figure whose canonical value passes ten trillion is a scale
mistake rather than a number — Apple's iPhone revenue came back as $75
trillion once, a figure already in dollars that was labeled millions. The cap
sits far enough out that it cannot fire on real guidance; across 35 extracted
figures it flagged four, all of them genuinely wrong, and none of NVIDIA's
sixteen plausible ones.

Segment guidance gets its own arithmetic check. Reportable segments are
disjoint parts of total revenue, so they can sum to less than the guided total
but never to more. The comparison uses the sum of segment *lows* against the
*highest* guided total — the most favorable reading available — so it only
fires when no values inside the guided ranges could be consistent. Identity
matters here too: Microsoft guides three segments for one period with ranges
that don't overlap each other, and treating those as readings of one figure
produced two false conflicts until the segment name became part of the key.

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

Getting the prepared remarks *to* the windower turned out to be the harder
half. A speaker turn is one paragraph, and some CFOs read the safe-harbor
sentence and the entire presentation as a single turn — Costco's does, in a
19,952-character paragraph. Boilerplate stripping matched the disclaimer and
dropped the paragraph, so three of Costco's four calls arrived with zero
prepared remarks. Only short paragraphs are eligible to be stripped now.
Tesla needed the opposite fix: it takes written retail and institutional
questions before analysts, so its Q&A opens on the company's own line rather
than an operator's and three of ten calls found no boundary at all.

Walmart showed the third variant. Its IR host previews the agenda — "that will
be followed by our question-and-answer session" — and the guard for exactly
that kind of forward reference allowed only "followed by a", so the host's
intro became the boundary and the executives' remarks all landed in the Q&A.
Beyond widening the guard, a weak cue is now only believed once the turns
before it hold a plausible amount of prepared remarks; an operator line plus a
short host intro isn't one, whatever the phrasing says. Strong cues are
unambiguous hand-offs and stay trusted wherever they appear.

Across 48 calls that leaves no transcript with empty prepared remarks, down
from four.

So `extract/windows.py` narrows the call first, from outlook cues where they
exist and from ranked overlapping chunks where they don't, and the pipeline
merges per-window results keyed on metric, detail, period and basis. Across 48
real calls that is 3.9 windows each and 26% of the original text. Overlap means
the same figure arrives more than once, and keying the merge on identity rather
than value keeps that from reading downstream as the company contradicting
itself.

## How good is the extraction

`scripts/eval_extraction.py` scores extractions against hand-labeled calls in
`tests/fixtures/eval/`. Three calls are labeled, chosen to stress different
things: NVIDIA's January fiscal year end and GAAP/non-GAAP pairs, Apple
declining to guide revenue and margin while guiding three other line items
anyway, and Microsoft's segment-level revenue ranges under a June fiscal year
end. With gemma3:4b:

```
nvda-2020q4   recall=0.20  precision=0.33  value_accuracy=0.50
aapl-2020q3   recall=0.00  precision=0.00
msft-2021q1   recall=0.00  precision=0.00
aggregate     recall=0.11  precision=0.29  value_accuracy=0.50
```

The first labeled call flattered the model. One transcript gave recall 0.20;
three give 0.11, which is what adding calls to a benchmark is supposed to do.

Precision was 0.25 before the fiscal-period validator landed. The item it
gained was an Apple revenue figure labeled FY2016 — a quarter that never said
which quarter — which is now dropped at parse time instead of counted as a
wrong answer. The other four false positives are identical across runs.

Numbers move a little between runs even at temperature 0: the same Microsoft
figure came back as `None-20.0 percent` on one run and `13.2-13.6
usd_billions` on the next, keeping its key and changing its values. Three
labeled calls is not enough to read small differences into.

The failures are the model, not the retrieval. All five of Microsoft's guided
figures sit inside the 6,859 characters the windower hands over, and the model
returns one item with a null bound and a fiscal year four years off. Apple is
the same: OpEx, tax rate and OI&E are all in the window and none come back.
NVIDIA's outlook is a tight labeled paragraph and the model does roughly half
of it correctly, which is the pattern -- it reads a dense list of figures and
loses track of prose with numbers scattered through it.

What holds regardless of model: the schema rejects unusable items, unquoted
figures get dropped, and reconciliation catches what survives. A 4B model is
the floor here, not the ceiling. The `claude-opus-5` provider is wired up and
window-aware but unmeasured, since that needs an API key.

## Providers

Extraction runs against either backend behind one interface:

- `ollama` (default) — `gemma3:4b` locally, so the whole pipeline runs for free
- `anthropic` — `claude-opus-5` with structured outputs, and `fallbacks:
  "default"` so a request the safety classifiers decline is re-run on another
  model server-side instead of coming back as a refusal. A key without that
  beta still works: the provider notices the rejection, drops the parameter
  and carries on without the safety net. Bulk runs go through the Batches API
  at half cost, which doesn't accept `fallbacks` — a declined window there has
  to be re-run interactively.
- `fixture` — replays recorded responses, which is what lets CI run with no
  keys and no network

Extractions are cached by `sha256(provider + model + prompt_version +
schema_version + transcript)`, so reruns are free but editing the prompt
correctly invalidates everything it touched.

## What the valuation actually covers

Worth being blunt about, because running it is how I found out. The valuation
path is guided EPS times a peer median P/E, and plenty of companies never
guide EPS. NVIDIA's Q4 FY2020 outlook guides revenue, gross margin, operating
expenses, OI&E, tax rate and capex — and no EPS — so all sixteen extracted
figures correctly value to nothing.

A revenue-based path would need P/S, which needs a share count, which is on
the premium `COMPANY_OVERVIEW` endpoint. Deriving EPS from guided revenue and
a margin assumption would mean inventing the number the model is supposed to
be reading, so the code returns nothing instead.

Peer multiples use the median rather than the mean — one peer at a 300x P/E
shouldn't move the comparison.

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

- Measure `claude-opus-5` against the same gold set, which needs an API key.
  Three labeled calls is a start, not a benchmark; more would sharpen it.
- Segment guidance is checked for internal consistency — an unnamed segment,
  segments summing past the guided total, the same segment guided twice with
  ranges that can't both hold. Comparing a segment against what it actually
  reported still needs segment actuals, which the free endpoints don't carry.
