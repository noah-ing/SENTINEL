"""
Red Team Loop: Adversarial attack generation and detector hardening.

The core research loop:
1. Generate attacks designed to bypass detection
2. Test attacks against the detector
3. Identify successful bypasses
4. Use bypasses to improve detection
5. Repeat
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import asyncio
from datetime import datetime
from typing import Callable

from sentinel.detection.pipeline import SentinelDetector, DetectorConfig
from sentinel.evaluation.benchmark import Attack, SentinelBenchmark
from sentinel.redteam.generator import AttackGenerator, GeneratorConfig, MockAttackGenerator
from sentinel.redteam.mutations import AttackMutator, MutationResult


@dataclass
class BypassResult:
    """A successful bypass of the detector."""

    attack: Attack
    detection_confidence: float
    detection_layer: str | None
    bypass_type: str  # "generated" or "mutated"
    source_attack_id: str | None = None  # For mutations
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "attack_id": self.attack.id,
            "attack_name": self.attack.name,
            "category": self.attack.category,
            "payload": self.attack.injection.get("payload", ""),
            "detection_confidence": self.detection_confidence,
            "detection_layer": self.detection_layer,
            "bypass_type": self.bypass_type,
            "source_attack_id": self.source_attack_id,
            "timestamp": self.timestamp,
        }


@dataclass
class RedTeamResult:
    """Results from a red team session."""

    total_attacks_tested: int
    bypasses_found: int
    bypass_rate: float
    bypasses: list[BypassResult]
    by_category: dict[str, dict]
    by_bypass_type: dict[str, int]
    duration_seconds: float

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "RED TEAM SESSION RESULTS",
            "=" * 60,
            f"Total Attacks Tested: {self.total_attacks_tested}",
            f"Bypasses Found: {self.bypasses_found}",
            f"Bypass Rate: {self.bypass_rate:.1%}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
            "Bypasses by Category:",
        ]

        for cat, stats in sorted(self.by_category.items()):
            lines.append(f"  {cat}: {stats['bypasses']}/{stats['tested']} ({stats['rate']:.1%})")

        lines.extend([
            "",
            "Bypasses by Type:",
            f"  Generated: {self.by_bypass_type.get('generated', 0)}",
            f"  Mutated: {self.by_bypass_type.get('mutated', 0)}",
            "=" * 60,
        ])

        return "\n".join(lines)


class RedTeamLoop:
    """
    Automated red team loop for adversarial testing.

    Generates attacks, tests them against the detector, and collects
    successful bypasses for analysis and detector improvement.

    Usage:
        loop = RedTeamLoop(detector)
        result = await loop.run(
            num_generated=50,
            num_mutations=100,
        )
        print(result.summary())
        loop.export_bypasses("bypasses.json")
    """

    def __init__(
        self,
        detector: SentinelDetector | None = None,
        generator: AttackGenerator | None = None,
        bypass_threshold: float = 0.5,  # Below this = bypass
    ):
        self.detector = detector or SentinelDetector(
            DetectorConfig(use_mock_models=True)
        )
        self.generator = generator or MockAttackGenerator()
        self.mutator = AttackMutator()
        self.bypass_threshold = bypass_threshold

        self.bypasses: list[BypassResult] = []
        self.tested_attacks: list[tuple[Attack, float, bool]] = []

    async def run(
        self,
        num_generated: int = 20,
        num_mutations: int = 50,
        existing_attacks: list[Attack] | None = None,
        categories: list[str] | None = None,
        on_bypass: Callable[[BypassResult], None] | None = None,
    ) -> RedTeamResult:
        """
        Run the red team loop.

        Args:
            num_generated: Number of new attacks to generate
            num_mutations: Number of mutation variants to create
            existing_attacks: Attacks to use for mutation source
            categories: Categories to focus on
            on_bypass: Callback for each bypass found

        Returns:
            RedTeamResult with session statistics
        """
        import time
        start_time = time.time()

        self.bypasses = []
        self.tested_attacks = []

        # Load existing attacks if needed
        if existing_attacks is None:
            from sentinel.evaluation.benchmark import load_attacks
            existing_attacks = load_attacks("full")

        # Load examples for generator
        self.generator.load_examples(existing_attacks)

        # Phase 1: Generate novel attacks
        print(f"Generating {num_generated} novel attacks...")
        generated = await self._generate_attacks(num_generated, categories)
        await self._test_attacks(generated, "generated", on_bypass)

        # Phase 2: Mutate existing attacks
        print(f"Creating {num_mutations} mutation variants...")
        mutations = self._mutate_attacks(existing_attacks, num_mutations)
        mutated_attacks = [m.mutated for m in mutations]
        await self._test_attacks(mutated_attacks, "mutated", on_bypass)

        # Compute results
        duration = time.time() - start_time
        result = self._compute_results(duration)

        return result

    async def _generate_attacks(
        self,
        count: int,
        categories: list[str] | None,
    ) -> list[Attack]:
        """Generate novel attacks."""
        attacks = []

        for i in range(count):
            try:
                category = None
                if categories:
                    category = categories[i % len(categories)]

                attack = await self.generator.generate(category=category)
                attacks.append(attack)

                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{count} attacks")

            except Exception as e:
                print(f"  Generation failed: {e}")

        return attacks

    def _mutate_attacks(
        self,
        attacks: list[Attack],
        count: int,
    ) -> list[MutationResult]:
        """Create mutation variants."""
        # Select attacks to mutate
        selected = []
        while len(selected) < count:
            selected.extend(attacks)
        selected = selected[:count]

        return self.mutator.mutate_batch(selected, mutations_per_attack=1)

    async def _test_attacks(
        self,
        attacks: list[Attack],
        bypass_type: str,
        on_bypass: Callable | None,
    ) -> None:
        """Test attacks against detector."""
        for attack in attacks:
            payload = attack.injection.get("payload", "")
            if not payload:
                continue

            context = {
                "task": attack.setup.get("agent_task", "unknown"),
                "tools": attack.setup.get("agent_tools", []),
            }

            result = await self.detector.scan(payload, context, depth="adaptive")

            is_bypass = not result.is_injection or result.confidence < self.bypass_threshold
            self.tested_attacks.append((attack, result.confidence, is_bypass))

            if is_bypass:
                bypass = BypassResult(
                    attack=attack,
                    detection_confidence=result.confidence,
                    detection_layer=result.layer,
                    bypass_type=bypass_type,
                    source_attack_id=attack.source if "mutation:" in str(attack.source) else None,
                )
                self.bypasses.append(bypass)

                if on_bypass:
                    on_bypass(bypass)

    def _compute_results(self, duration: float) -> RedTeamResult:
        """Compute session results."""
        total = len(self.tested_attacks)
        num_bypasses = len(self.bypasses)

        # By category
        by_category = {}
        for attack, confidence, is_bypass in self.tested_attacks:
            cat = attack.category
            if cat not in by_category:
                by_category[cat] = {"tested": 0, "bypasses": 0}
            by_category[cat]["tested"] += 1
            if is_bypass:
                by_category[cat]["bypasses"] += 1

        for cat in by_category:
            stats = by_category[cat]
            stats["rate"] = stats["bypasses"] / stats["tested"] if stats["tested"] > 0 else 0

        # By type
        by_type = {}
        for bypass in self.bypasses:
            t = bypass.bypass_type
            by_type[t] = by_type.get(t, 0) + 1

        return RedTeamResult(
            total_attacks_tested=total,
            bypasses_found=num_bypasses,
            bypass_rate=num_bypasses / total if total > 0 else 0,
            bypasses=self.bypasses,
            by_category=by_category,
            by_bypass_type=by_type,
            duration_seconds=duration,
        )

    def export_bypasses(self, path: Path | str) -> None:
        """Export bypasses to JSON file."""
        path = Path(path)

        data = {
            "timestamp": datetime.now().isoformat(),
            "total_bypasses": len(self.bypasses),
            "bypasses": [b.to_dict() for b in self.bypasses],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(self.bypasses)} bypasses to {path}")

    def get_bypass_payloads(self) -> list[str]:
        """Get all bypass payloads for analysis."""
        return [b.attack.injection.get("payload", "") for b in self.bypasses]

    def get_weakest_categories(self, top_n: int = 3) -> list[tuple[str, float]]:
        """Get categories with highest bypass rates."""
        by_category = {}
        for attack, confidence, is_bypass in self.tested_attacks:
            cat = attack.category
            if cat not in by_category:
                by_category[cat] = {"tested": 0, "bypasses": 0}
            by_category[cat]["tested"] += 1
            if is_bypass:
                by_category[cat]["bypasses"] += 1

        rates = [
            (cat, stats["bypasses"] / stats["tested"])
            for cat, stats in by_category.items()
            if stats["tested"] >= 3  # Minimum sample size
        ]

        rates.sort(key=lambda x: x[1], reverse=True)
        return rates[:top_n]


async def run_red_team_session(
    num_generated: int = 20,
    num_mutations: int = 50,
    output_path: str | None = None,
) -> RedTeamResult:
    """
    Convenience function to run a red team session.

    Usage:
        result = await run_red_team_session(
            num_generated=50,
            num_mutations=100,
            output_path="bypasses.json"
        )
    """
    detector = SentinelDetector(DetectorConfig(use_mock_models=True))
    loop = RedTeamLoop(detector)

    def on_bypass(bypass: BypassResult):
        print(f"  [BYPASS] {bypass.attack.name} (confidence: {bypass.detection_confidence:.2f})")

    result = await loop.run(
        num_generated=num_generated,
        num_mutations=num_mutations,
        on_bypass=on_bypass,
    )

    print()
    print(result.summary())

    if output_path:
        loop.export_bypasses(output_path)

    return result
