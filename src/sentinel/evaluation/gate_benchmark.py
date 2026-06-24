"""Benchmark for the tool-call gate (SentinelMiddleware.check_tool_call).

Unlike SentinelBenchmark, which text-scans attack payloads through the detector,
this benchmark drives the *defense gate* directly: it calls check_tool_call with
structured (tool_name, tool_args) pairs and measures whether the gate makes the
right allow / block / approval decision.

The dataset mixes attacks and benign calls on the *same* high-risk tools, so the
numbers reflect whether the gate distinguishes malicious arguments from
legitimate ones - not just whether it flags risky tool names.

Note on scope: each case is judged in isolation (a fresh middleware per call), so
the behavioral monitor's multi-call signals (exfiltration sequences, rate limits)
are not exercised here. This measures the per-call gate decision only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import asyncio

from sentinel.detection.pipeline import SentinelDetector, DetectorConfig
from sentinel.defense.middleware import SentinelMiddleware, AgentContext
from sentinel.defense.policy import SecurityPolicy


# A gate decision collapses to one of these outcomes.
ALLOW = "allow"        # call permitted to execute autonomously
BLOCK = "block"        # call hard-blocked
APPROVAL = "approval"  # call held for human approval


@dataclass
class GateCaseResult:
    case_id: str
    label: str          # "attack" or "benign"
    tool_name: str
    outcome: str        # ALLOW / BLOCK / APPROVAL
    reason: str | None


@dataclass
class GateBenchmarkResults:
    results: list[GateCaseResult]

    # --- helpers -------------------------------------------------------
    def _count(self, label: str, outcome: str) -> int:
        return sum(1 for r in self.results if r.label == label and r.outcome == outcome)

    def _total(self, label: str) -> int:
        return sum(1 for r in self.results if r.label == label)

    # --- headline metrics ----------------------------------------------
    @property
    def attack_prevention_rate(self) -> float:
        """Fraction of attacks NOT auto-allowed (blocked or held for approval).

        This is the security-relevant recall: an attack routed to a human is
        prevented from executing autonomously.
        """
        total = self._total("attack")
        if total == 0:
            return 0.0
        prevented = total - self._count("attack", ALLOW)
        return prevented / total

    @property
    def attack_autonomous_execution_rate(self) -> float:
        """Fraction of attacks the gate let through to autonomous execution.

        The dangerous failure mode - this is what you do not want.
        """
        total = self._total("attack")
        if total == 0:
            return 0.0
        return self._count("attack", ALLOW) / total

    @property
    def benign_clean_pass_rate(self) -> float:
        """Fraction of benign calls auto-allowed with no friction."""
        total = self._total("benign")
        if total == 0:
            return 0.0
        return self._count("benign", ALLOW) / total

    @property
    def benign_approval_rate(self) -> float:
        """Fraction of benign calls sent for human approval (soft friction)."""
        total = self._total("benign")
        if total == 0:
            return 0.0
        return self._count("benign", APPROVAL) / total

    @property
    def benign_block_rate(self) -> float:
        """Fraction of benign calls hard-blocked (the costly false positive)."""
        total = self._total("benign")
        if total == 0:
            return 0.0
        return self._count("benign", BLOCK) / total

    def summary(self) -> str:
        n_atk = self._total("attack")
        n_ben = self._total("benign")
        lines = [
            "=" * 60,
            "SENTINEL Tool-Call Gate Benchmark",
            "(drives check_tool_call directly; per-call decisions)",
            "=" * 60,
            f"Cases: {len(self.results)}  ({n_atk} attacks, {n_ben} benign)",
            "",
            "ATTACKS",
            f"  Prevented (blocked or approval): {self.attack_prevention_rate:.1%}"
            f"  [{n_atk - self._count('attack', ALLOW)}/{n_atk}]",
            f"    - hard-blocked:   {self._count('attack', BLOCK)}",
            f"    - held for human: {self._count('attack', APPROVAL)}",
            f"  Auto-executed (MISS): {self.attack_autonomous_execution_rate:.1%}"
            f"  [{self._count('attack', ALLOW)}/{n_atk}]",
            "",
            "BENIGN",
            f"  Clean auto-allow:     {self.benign_clean_pass_rate:.1%}"
            f"  [{self._count('benign', ALLOW)}/{n_ben}]",
            f"  Held for approval:    {self.benign_approval_rate:.1%}"
            f"  [{self._count('benign', APPROVAL)}/{n_ben}]  (friction)",
            f"  Hard-blocked (FALSE POSITIVE): {self.benign_block_rate:.1%}"
            f"  [{self._count('benign', BLOCK)}/{n_ben}]",
            "",
            "MISSES (attacks that auto-executed):",
        ]
        misses = [r for r in self.results if r.label == "attack" and r.outcome == ALLOW]
        if misses:
            for r in misses:
                lines.append(f"  - {r.case_id} ({r.tool_name})")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("FALSE POSITIVES (benign hard-blocked):")
        fps = [r for r in self.results if r.label == "benign" and r.outcome == BLOCK]
        if fps:
            for r in fps:
                lines.append(f"  - {r.case_id} ({r.tool_name}): {r.reason}")
        else:
            lines.append("  (none)")

        lines.append("=" * 60)
        return "\n".join(lines)


def load_gate_cases(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).parent / "gate_cases.json"
    with open(path) as f:
        return json.load(f)


class GateBenchmark:
    """Evaluate the tool-call gate against labeled attack/benign tool calls."""

    def __init__(
        self,
        policy: SecurityPolicy | None = None,
        cases_path: Path | None = None,
    ):
        self.data = load_gate_cases(cases_path)
        # Default: the shipping default policy (the thing we actually ship).
        self.policy = policy if policy is not None else SecurityPolicy()
        self.detector = SentinelDetector(DetectorConfig(use_mock_models=True))

    async def run(self) -> GateBenchmarkResults:
        ctx = AgentContext(
            current_task=self.data["task"],
            available_tools=self.data["available_tools"],
        )
        results: list[GateCaseResult] = []
        for case in self.data["cases"]:
            # Fresh middleware per case so action counts and monitor history
            # from earlier cases never bleed into this decision.
            mw = SentinelMiddleware(detector=self.detector, policy=self.policy)
            res = await mw.check_tool_call(
                tool_name=case["tool_name"],
                tool_args=case["tool_args"],
                context=ctx,
            )
            if res.blocked:
                outcome = BLOCK
            elif res.requires_approval:
                outcome = APPROVAL
            else:
                outcome = ALLOW
            results.append(GateCaseResult(
                case_id=case["id"],
                label=case["label"],
                tool_name=case["tool_name"],
                outcome=outcome,
                reason=res.reason,
            ))
        return GateBenchmarkResults(results=results)


async def _main() -> None:
    bench = GateBenchmark()
    results = await bench.run()
    print(results.summary())


if __name__ == "__main__":
    asyncio.run(_main())
