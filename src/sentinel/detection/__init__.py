"""Detection engine for prompt injection attacks."""

from sentinel.detection.pipeline import SentinelDetector
from sentinel.detection.heuristic import HeuristicDetector
from sentinel.detection.classifier import ClassifierDetector
from sentinel.detection.llm_judge import LLMJudgeDetector
from sentinel.detection.result import DetectionResult

# SemanticDetector requires numpy, so lazy import
def __getattr__(name):
    if name == "SemanticDetector":
        from sentinel.detection.semantic import SemanticDetector
        return SemanticDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SentinelDetector",
    "HeuristicDetector",
    "ClassifierDetector",
    "LLMJudgeDetector",
    "SemanticDetector",
    "DetectionResult",
]
