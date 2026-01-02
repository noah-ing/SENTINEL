"""Detection result types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionResult:
    """Result from the detection pipeline."""

    is_injection: bool
    confidence: float
    layer: str | None = None
    details: Any = None
    latency_ms: float | None = None

    def __bool__(self) -> bool:
        return self.is_injection


@dataclass
class HeuristicResult:
    """Result from heuristic detection."""

    confidence: float
    matches: list[str] = field(default_factory=list)
    patterns_checked: int = 0


@dataclass
class ClassifierResult:
    """Result from ML classifier detection."""

    confidence: float
    explanation: str | None = None
    embedding: list[float] | None = None


@dataclass
class JudgeResult:
    """Result from LLM judge detection."""

    is_injection: bool
    confidence: float
    reasoning: str | None = None
    tokens_used: int = 0
