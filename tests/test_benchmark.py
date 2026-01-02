"""Tests for the benchmark module."""

import pytest
from pathlib import Path

from sentinel.evaluation.benchmark import (
    SentinelBenchmark,
    Attack,
    load_attacks,
    BenchmarkResults,
)
from sentinel.evaluation.metrics import compute_metrics, DetectionMetrics
from sentinel.detection.pipeline import SentinelDetector, DetectorConfig


class TestAttackLoading:
    """Tests for attack loading."""

    def test_load_attacks_full(self):
        attacks = load_attacks("full")
        assert len(attacks) > 0
        assert all(isinstance(a, Attack) for a in attacks)

    def test_load_attacks_quick(self):
        attacks = load_attacks("quick")
        assert len(attacks) <= 20

    def test_attack_has_required_fields(self):
        attacks = load_attacks("full")
        for attack in attacks:
            assert attack.id
            assert attack.category
            assert attack.name
            assert attack.difficulty in ["easy", "medium", "hard", "expert"]
            assert attack.setup
            assert attack.injection
            assert attack.success_criteria

    def test_attack_from_dict(self):
        data = {
            "id": "test_001",
            "category": "direct_override",
            "name": "Test Attack",
            "description": "A test attack",
            "difficulty": "easy",
            "setup": {
                "agent_task": "Test task",
                "agent_tools": ["read_file"],
                "user_intent": "Test intent",
            },
            "injection": {
                "vector": "user_message",
                "payload": "Ignore previous instructions",
            },
            "success_criteria": {
                "action_taken": "Some action",
            },
        }

        attack = Attack.from_dict(data)
        assert attack.id == "test_001"
        assert attack.category == "direct_override"
        assert attack.difficulty == "easy"


class TestBenchmark:
    """Tests for the benchmark runner."""

    @pytest.fixture
    def detector(self):
        return SentinelDetector(DetectorConfig(use_mock_models=True))

    @pytest.fixture
    def benchmark(self):
        return SentinelBenchmark(attack_suite="quick")

    @pytest.mark.asyncio
    async def test_run_benchmark(self, benchmark, detector):
        results = await benchmark.run(
            detector=detector,
            depth="fast",
            show_progress=False,
        )

        assert isinstance(results, BenchmarkResults)
        assert results.total_attacks > 0
        assert 0 <= results.detection_rate <= 1

    @pytest.mark.asyncio
    async def test_benchmark_by_category(self, benchmark, detector):
        results = await benchmark.run(
            detector=detector,
            depth="fast",
            show_progress=False,
        )

        assert len(results.by_category) > 0
        for rate in results.by_category.values():
            assert 0 <= rate <= 1

    @pytest.mark.asyncio
    async def test_benchmark_by_difficulty(self, benchmark, detector):
        results = await benchmark.run(
            detector=detector,
            depth="fast",
            show_progress=False,
        )

        assert len(results.by_difficulty) > 0
        for rate in results.by_difficulty.values():
            assert 0 <= rate <= 1

    @pytest.mark.asyncio
    async def test_benchmark_latency(self, benchmark, detector):
        results = await benchmark.run(
            detector=detector,
            depth="fast",
            show_progress=False,
        )

        assert results.avg_latency_ms >= 0
        assert results.p50_latency_ms >= 0
        assert results.p99_latency_ms >= 0

    def test_results_summary(self):
        results = BenchmarkResults(
            total_attacks=100,
            detection_rate=0.85,
            block_rate=0.80,
            false_positive_rate=0.05,
            by_category={"direct": 0.90, "indirect": 0.75},
            by_difficulty={"easy": 0.95, "hard": 0.70},
            avg_latency_ms=10.5,
            p50_latency_ms=8.0,
            p99_latency_ms=50.0,
        )

        summary = results.summary()
        assert "100" in summary
        assert "85" in summary
        assert "direct" in summary.lower()


class TestMetrics:
    """Tests for metrics computation."""

    def test_compute_metrics_perfect(self):
        predictions = [True, True, False, False]
        labels = [True, True, False, False]

        metrics = compute_metrics(predictions, labels)

        assert metrics.true_positives == 2
        assert metrics.false_positives == 0
        assert metrics.true_negatives == 2
        assert metrics.false_negatives == 0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.accuracy == 1.0

    def test_compute_metrics_with_errors(self):
        predictions = [True, True, True, False]
        labels = [True, False, True, True]

        metrics = compute_metrics(predictions, labels)

        assert metrics.true_positives == 2  # Correctly detected
        assert metrics.false_positives == 1  # False alarm
        assert metrics.true_negatives == 0  # Correctly passed
        assert metrics.false_negatives == 1  # Missed

    def test_precision_recall_tradeoff(self):
        # High precision, low recall
        predictions = [True, False, False, False]
        labels = [True, True, True, False]

        metrics = compute_metrics(predictions, labels)
        assert metrics.precision == 1.0  # No false positives
        assert metrics.recall < 0.5  # Missed most positives

    def test_false_positive_rate(self):
        predictions = [True, True, True, True]
        labels = [True, True, False, False]

        metrics = compute_metrics(predictions, labels)
        assert metrics.false_positive_rate == 1.0  # All negatives marked as positive

    def test_empty_inputs(self):
        metrics = compute_metrics([], [])
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.accuracy == 0.0


class TestExport:
    """Tests for result export."""

    @pytest.mark.asyncio
    async def test_export_results(self, tmp_path):
        detector = SentinelDetector(DetectorConfig(use_mock_models=True))
        benchmark = SentinelBenchmark(attack_suite="quick")

        await benchmark.run(
            detector=detector,
            depth="fast",
            show_progress=False,
        )

        output_path = tmp_path / "results.json"
        benchmark.export_results(output_path)

        assert output_path.exists()

        import json
        with open(output_path) as f:
            data = json.load(f)

        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) > 0
