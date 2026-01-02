"""Benchmark harness for evaluating detection and defense effectiveness."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import asyncio
from collections import defaultdict
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

from sentinel.detection.pipeline import SentinelDetector
from sentinel.defense.middleware import SentinelMiddleware, AgentContext


@dataclass
class Attack:
    """Represents a single attack from the benchmark."""

    id: str
    category: str
    name: str
    description: str
    difficulty: str
    target_capability: str
    setup: dict
    injection: dict
    success_criteria: dict
    detection_hints: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    source: str = "original"

    @classmethod
    def from_dict(cls, data: dict) -> "Attack":
        return cls(
            id=data["id"],
            category=data["category"],
            name=data["name"],
            description=data["description"],
            difficulty=data["difficulty"],
            target_capability=data.get("target_capability", "general"),
            setup=data["setup"],
            injection=data["injection"],
            success_criteria=data["success_criteria"],
            detection_hints=data.get("detection_hints", {}),
            tags=data.get("tags", []),
            source=data.get("source", "original"),
        )


@dataclass
class AttackResult:
    """Result of running a single attack."""

    attack_id: str
    attack_category: str
    attack_difficulty: str
    detected: bool
    detection_confidence: float
    detection_layer: str | None
    detection_latency_ms: float | None
    blocked: bool | None = None
    details: Any = None


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    total_attacks: int
    detection_rate: float
    block_rate: float | None
    false_positive_rate: float | None
    by_category: dict[str, float]
    by_difficulty: dict[str, float]
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    cost_estimate: float | None = None

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 50,
            "SENTINEL Benchmark Results",
            "=" * 50,
            f"Total Attacks: {self.total_attacks}",
            f"Detection Rate: {self.detection_rate:.1%}",
        ]

        if self.block_rate is not None:
            lines.append(f"Block Rate: {self.block_rate:.1%}")

        if self.false_positive_rate is not None:
            lines.append(f"False Positive Rate: {self.false_positive_rate:.1%}")

        lines.extend([
            "",
            "By Category:",
        ])
        for cat, rate in sorted(self.by_category.items()):
            lines.append(f"  {cat}: {rate:.1%}")

        lines.extend([
            "",
            "By Difficulty:",
        ])
        for diff, rate in sorted(self.by_difficulty.items()):
            lines.append(f"  {diff}: {rate:.1%}")

        lines.extend([
            "",
            "Latency:",
            f"  Average: {self.avg_latency_ms:.1f}ms",
            f"  p50: {self.p50_latency_ms:.1f}ms",
            f"  p99: {self.p99_latency_ms:.1f}ms",
            "=" * 50,
        ])

        return "\n".join(lines)


def load_attacks(suite: str = "full", attacks_dir: Path | None = None) -> list[Attack]:
    """
    Load attacks from the benchmark suite.

    Args:
        suite: Attack suite to load ("full", "quick", or category name)
        attacks_dir: Directory containing attack JSON files

    Returns:
        List of Attack objects
    """
    if attacks_dir is None:
        # Default to package attacks directory
        attacks_dir = Path(__file__).parent.parent.parent.parent / "attacks"

    attacks = []

    # Find all JSON files in attack directories
    for category_dir in attacks_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue

        # Filter by suite
        if suite not in ("full", "quick") and suite not in category_dir.name:
            continue

        for json_file in category_dir.glob("*.json"):
            if json_file.name == "schema.json":
                continue

            try:
                with open(json_file) as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        attacks.append(Attack.from_dict(item))
                else:
                    attacks.append(Attack.from_dict(data))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load {json_file}: {e}")

    # For quick suite, sample a subset
    if suite == "quick":
        import random
        attacks = random.sample(attacks, min(20, len(attacks)))

    return attacks


class SentinelBenchmark:
    """
    Evaluate detection and defense effectiveness.

    Usage:
        benchmark = SentinelBenchmark(attack_suite="full")
        results = await benchmark.run(detector=my_detector)
        print(results.summary())
    """

    def __init__(
        self,
        attack_suite: str = "full",
        attacks_dir: Path | None = None,
    ):
        self.attacks = load_attacks(attack_suite, attacks_dir)
        self.results: list[AttackResult] = []

    async def run(
        self,
        detector: SentinelDetector,
        defense: SentinelMiddleware | None = None,
        depth: str = "adaptive",
        show_progress: bool = True,
    ) -> BenchmarkResults:
        """
        Run the benchmark.

        Args:
            detector: SENTINEL detector to evaluate
            defense: Optional defense middleware to test
            depth: Detection depth to use
            show_progress: Show progress bar

        Returns:
            BenchmarkResults with aggregated metrics
        """
        self.results = []

        iterator = tqdm(self.attacks, desc="Running benchmark") if show_progress else self.attacks

        for attack in iterator:
            result = await self._run_single(attack, detector, defense, depth)
            self.results.append(result)

        return self._compute_metrics()

    async def _run_single(
        self,
        attack: Attack,
        detector: SentinelDetector,
        defense: SentinelMiddleware | None,
        depth: str,
    ) -> AttackResult:
        """Run a single attack."""
        # Build context
        context = {
            "task": attack.setup["agent_task"],
            "tools": attack.setup["agent_tools"],
        }

        # Get the injection payload
        payload = attack.injection["payload"]

        # Run detection
        detection = await detector.scan(payload, context, depth=depth)

        # Run defense if provided
        blocked = None
        if defense:
            agent_context = AgentContext(
                current_task=attack.setup["agent_task"],
                available_tools=attack.setup["agent_tools"],
            )
            # Test with a generic tool call
            tool_result = await defense.check_tool_call(
                tool_name="process_content",
                tool_args={"content": payload},
                context=agent_context,
            )
            blocked = tool_result.blocked

        return AttackResult(
            attack_id=attack.id,
            attack_category=attack.category,
            attack_difficulty=attack.difficulty,
            detected=detection.is_injection,
            detection_confidence=detection.confidence,
            detection_layer=detection.layer,
            detection_latency_ms=detection.latency_ms,
            blocked=blocked,
            details=detection.details,
        )

    def _compute_metrics(self) -> BenchmarkResults:
        """Compute aggregated metrics from results."""
        total = len(self.results)
        if total == 0:
            return BenchmarkResults(
                total_attacks=0,
                detection_rate=0,
                block_rate=None,
                false_positive_rate=None,
                by_category={},
                by_difficulty={},
                avg_latency_ms=0,
                p50_latency_ms=0,
                p99_latency_ms=0,
            )

        detected = sum(1 for r in self.results if r.detected)
        blocked = sum(1 for r in self.results if r.blocked)
        has_block = any(r.blocked is not None for r in self.results)

        # By category
        by_category = defaultdict(list)
        for r in self.results:
            by_category[r.attack_category].append(r.detected)

        category_rates = {
            cat: sum(detections) / len(detections)
            for cat, detections in by_category.items()
        }

        # By difficulty
        by_difficulty = defaultdict(list)
        for r in self.results:
            by_difficulty[r.attack_difficulty].append(r.detected)

        difficulty_rates = {
            diff: sum(detections) / len(detections)
            for diff, detections in by_difficulty.items()
        }

        # Latency
        latencies = [r.detection_latency_ms for r in self.results if r.detection_latency_ms]
        if latencies:
            latencies.sort()
            avg_latency = sum(latencies) / len(latencies)
            p50_latency = latencies[len(latencies) // 2]
            p99_latency = latencies[int(len(latencies) * 0.99)]
        else:
            avg_latency = p50_latency = p99_latency = 0

        return BenchmarkResults(
            total_attacks=total,
            detection_rate=detected / total,
            block_rate=blocked / total if has_block else None,
            false_positive_rate=None,  # Would need benign examples to compute
            by_category=category_rates,
            by_difficulty=difficulty_rates,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p99_latency_ms=p99_latency,
        )

    def export_results(self, path: Path) -> None:
        """Export detailed results to JSON."""
        data = {
            "results": [
                {
                    "attack_id": r.attack_id,
                    "category": r.attack_category,
                    "difficulty": r.attack_difficulty,
                    "detected": r.detected,
                    "confidence": r.detection_confidence,
                    "layer": r.detection_layer,
                    "latency_ms": r.detection_latency_ms,
                    "blocked": r.blocked,
                }
                for r in self.results
            ],
            "summary": {
                "total": len(self.results),
                "detected": sum(1 for r in self.results if r.detected),
                "detection_rate": sum(1 for r in self.results if r.detected) / len(self.results)
                if self.results else 0,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
