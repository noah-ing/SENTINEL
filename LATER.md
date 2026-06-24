# LATER — the board

Deferred on purpose. Nothing here gets pulled forward until the current week's two
tasks are done. Shiny ideas land here and wait.

## This week (the only two things)
- [x] **Task 1 — Arm the default policy.** Default `SecurityPolicy()` now ships
  `strict_policy`'s blocked-pattern regex + approval routing. Verified: the default
  gate blocks path traversal and routes `send_email`/`run_shell` to human approval
  (previously waved 2 of 3 attacks through).
- [x] **Task 2 — Benchmark the gate.** `sentinel gate-benchmark` drives
  `check_tool_call` directly over 24 labeled tool calls (12 attack, 12 benign on the
  same high-risk tools). Real result on the default policy:
  - Attacks prevented (blocked or held for approval): **91.7% (11/12)**.
  - One miss: **SQL injection** in `database_query` auto-executes (no regex/heuristic
    for it).
  - Benign: 41.7% clean auto-allow, 50% held for approval (friction), **8.3% hard-block
    false positive** (a security doc that mentions "password").
  Code: `src/sentinel/evaluation/gate_benchmark.py`, data:
  `src/sentinel/evaluation/gate_cases.json`.

## Decisions parked (do NOT solve now)
- **PyPI name.** `sentinel-ai` is taken on PyPI by a different project (Lennard Gross,
  data-poisoning toolkit). We need a new handle eventually, but we don't publish this
  week, so we don't need the name this week. Decide when v1 is ready to ship.

## Surfaced by the gate benchmark (next candidates, after this week)
- **SQL injection is a real gap.** `database_query` with `DROP TABLE`/`'; --` is neither
  in `blocked_patterns` nor caught by the heuristic, so it auto-executes. Add SQL-injection
  patterns (or argument-aware checks for query tools).
- **`password` blocked-pattern over-triggers.** It hard-blocks a benign doc that merely
  mentions the word. Consider scoping credential patterns to `key=value` / assignment
  shapes instead of bare keywords.
- **Approval short-circuits argument inspection.** For tools in `require_approval`
  (send_email, run_shell, http_request, ...), the gate returns "approval" at Check 3,
  before the regex/heuristic arg checks ever run. Fine if a human always reviews, but it
  means true argument-level detection only fires on non-approval tools. Decide if that's
  the intended contract.

## Found tonight, fix when relevant (not this week's scope)
- **2 pre-existing test failures** in `tests/test_detection.py::TestHeuristicDetector`
  (`test_detects_authority_claims`: 0.25 vs >0.3 threshold;
  `test_multiple_matches_increase_confidence`: both cap at 1.0). Heuristic-tuning, not
  policy. Unrelated to Task 1.
- **Benchmark honesty gaps** (surface before citing the 89.5% publicly):
  - Runs with `use_mock_models=True` — the ML classifier and LLM judge are
    keyword stand-ins, not the advertised real layers.
  - No benign control set → false-positive rate is structurally uncomputable. It
    measures recall only. (Task 2 fixes this for the gate specifically.)
  - The tool-hijacking "80%" is text-scanning payloads, and 1 of its 4 hits is
    propped by the mock classifier. The gate itself has never been benchmarked.
