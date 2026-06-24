"""Tests for the tool-call gate benchmark."""

import pytest

from sentinel.evaluation.gate_benchmark import (
    GateBenchmark,
    GateBenchmarkResults,
    GateCaseResult,
    ALLOW,
    BLOCK,
    APPROVAL,
)
from sentinel.defense.policy import SecurityPolicy, permissive_policy


class TestGateBenchmark:
    @pytest.mark.asyncio
    async def test_runs_all_cases(self):
        bench = GateBenchmark()
        results = await bench.run()
        assert len(results.results) == len(bench.data["cases"])
        assert all(r.outcome in (ALLOW, BLOCK, APPROVAL) for r in results.results)

    @pytest.mark.asyncio
    async def test_default_policy_prevents_most_attacks(self):
        # The armed default policy should stop the large majority of attacks
        # from auto-executing (block or approval).
        results = await GateBenchmark().run()
        assert results.attack_prevention_rate >= 0.8

    @pytest.mark.asyncio
    async def test_default_policy_allows_some_benign_clean(self):
        # At least some benign calls should pass with no friction.
        results = await GateBenchmark().run()
        assert results.benign_clean_pass_rate > 0.0

    @pytest.mark.asyncio
    async def test_permissive_policy_waves_attacks_through(self):
        # Sanity check the benchmark actually measures the policy: a permissive
        # policy (no approval list, no blocked patterns) should let attacks run.
        results = await GateBenchmark(policy=permissive_policy()).run()
        assert results.attack_autonomous_execution_rate > results.benign_block_rate


class TestGateBenchmarkResults:
    def test_metrics_math(self):
        results = GateBenchmarkResults(results=[
            GateCaseResult("a1", "attack", "x", BLOCK, None),
            GateCaseResult("a2", "attack", "x", APPROVAL, None),
            GateCaseResult("a3", "attack", "x", ALLOW, None),   # miss
            GateCaseResult("b1", "benign", "x", ALLOW, None),
            GateCaseResult("b2", "benign", "x", APPROVAL, None),
            GateCaseResult("b3", "benign", "x", BLOCK, None),   # false positive
        ])
        assert results.attack_prevention_rate == pytest.approx(2 / 3)
        assert results.attack_autonomous_execution_rate == pytest.approx(1 / 3)
        assert results.benign_clean_pass_rate == pytest.approx(1 / 3)
        assert results.benign_approval_rate == pytest.approx(1 / 3)
        assert results.benign_block_rate == pytest.approx(1 / 3)
