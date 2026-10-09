# AlphaVantage fixtures

`*_NVDA.json` are real responses, recorded 2026-10-08. Peer fixtures are
hand-written against the same shapes so tests stay deterministic.

Free-tier reality check: only `EARNINGS` and `GLOBAL_QUOTE` are reachable
without a premium key. `COMPANY_OVERVIEW` and `INCOME_STATEMENT` return a
rate_limit/premium error, which is why P/E is the only multiple the service
computes end to end on a free key — see `crosscheck/alphavantage.py`.

Re-record with `python scripts/record_fixtures.py NVDA AMD INTC`.
