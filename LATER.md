# LATER — the board

Deferred on purpose. Nothing here gets pulled forward until the current week's two
tasks are done. Shiny ideas land here and wait.

## This week (the only two things)
- [x] **Task 1 — Arm the default policy.** Default `SecurityPolicy()` now ships
  `strict_policy`'s blocked-pattern regex + approval routing. Verified: the default
  gate blocks path traversal and routes `send_email`/`run_shell` to human approval
  (previously waved 2 of 3 attacks through).
- [ ] **Task 2 — Benchmark the gate.** Write a benchmark that calls
  `SentinelMiddleware.check_tool_call` directly (NOT text-scanning payloads), with
  both benign tool calls and attacks in the set. Produces real precision + recall for
  the actual wedge, replacing the borrowed 80%.

## Decisions parked (do NOT solve now)
- **PyPI name.** `sentinel-ai` is taken on PyPI by a different project (Lennard Gross,
  data-poisoning toolkit). We need a new handle eventually, but we don't publish this
  week, so we don't need the name this week. Decide when v1 is ready to ship.

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
