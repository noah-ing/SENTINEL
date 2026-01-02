"""Main detection pipeline combining all detection layers."""

from dataclasses import dataclass, field
import time
from typing import Literal

from sentinel.detection.heuristic import HeuristicDetector, HeuristicConfig
from sentinel.detection.classifier import ClassifierDetector, ClassifierConfig, MockClassifierDetector
from sentinel.detection.llm_judge import LLMJudgeDetector, LLMJudgeConfig, MockLLMJudgeDetector
from sentinel.detection.result import DetectionResult


@dataclass
class DetectorConfig:
    """Configuration for the detection pipeline."""

    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    llm_judge: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)

    # Thresholds
    heuristic_threshold: float = 0.95
    classifier_threshold: float = 0.90
    adaptive_heuristic_trigger: float = 0.3
    adaptive_classifier_trigger: float = 0.5

    # Feature flags
    use_classifier: bool = True
    use_llm_judge: bool = True
    use_mock_models: bool = False  # For testing


class SentinelDetector:
    """
    Multi-layer detection pipeline for prompt injection.

    Layers (in order):
    1. Heuristic (fast, cheap, catches obvious attacks) - < 1ms
    2. Classifier (ML model, catches encoded/obfuscated) - < 50ms
    3. LLM Judge (slow, expensive, catches sophisticated) - < 3s
    4. Behavioral (monitors agent actions) - continuous

    Each layer can short-circuit if confidence is high enough.
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()

        # Initialize detectors
        self.heuristic = HeuristicDetector(self.config.heuristic)

        if self.config.use_mock_models:
            self.classifier = MockClassifierDetector(self.config.classifier)
            self.llm_judge = MockLLMJudgeDetector(self.config.llm_judge)
        else:
            self.classifier = ClassifierDetector(self.config.classifier)
            self.llm_judge = LLMJudgeDetector(self.config.llm_judge)

    async def scan(
        self,
        content: str,
        context: dict | None = None,
        depth: Literal["fast", "thorough", "adaptive"] = "adaptive",
    ) -> DetectionResult:
        """
        Scan content for prompt injection.

        Args:
            content: Text to analyze
            context: Optional dict with 'task' and 'tools' keys
            depth: Detection depth
                - "fast": Heuristic only
                - "thorough": All layers
                - "adaptive": Use more layers if suspicious

        Returns:
            DetectionResult with is_injection, confidence, layer, and details
        """
        start_time = time.perf_counter()

        # Layer 1: Heuristics (always run, < 1ms)
        heuristic_result = self.heuristic.scan(content)

        if heuristic_result.confidence >= self.config.heuristic_threshold:
            latency = (time.perf_counter() - start_time) * 1000
            return DetectionResult(
                is_injection=True,
                confidence=heuristic_result.confidence,
                layer="heuristic",
                details={
                    "matches": heuristic_result.matches,
                    "patterns_checked": heuristic_result.patterns_checked,
                },
                latency_ms=latency,
            )

        # Fast mode stops here
        if depth == "fast":
            latency = (time.perf_counter() - start_time) * 1000
            return DetectionResult(
                is_injection=False,
                confidence=heuristic_result.confidence,
                layer="heuristic",
                latency_ms=latency,
            )

        # Layer 2: Classifier (if enabled)
        if self.config.use_classifier:
            should_run_classifier = (
                depth == "thorough"
                or heuristic_result.confidence >= self.config.adaptive_heuristic_trigger
            )

            if should_run_classifier:
                classifier_result = await self.classifier.scan(content, context)

                if classifier_result.confidence >= self.config.classifier_threshold:
                    latency = (time.perf_counter() - start_time) * 1000
                    return DetectionResult(
                        is_injection=True,
                        confidence=classifier_result.confidence,
                        layer="classifier",
                        details={"explanation": classifier_result.explanation},
                        latency_ms=latency,
                    )

        # Layer 3: LLM Judge (if enabled and suspicious)
        if self.config.use_llm_judge and depth != "fast":
            should_run_judge = (
                depth == "thorough"
                or heuristic_result.confidence >= self.config.adaptive_heuristic_trigger
                or (
                    self.config.use_classifier
                    and classifier_result.confidence >= self.config.adaptive_classifier_trigger
                )
            )

            if should_run_judge:
                judge_result = await self.llm_judge.scan(content, context)

                if judge_result.is_injection:
                    latency = (time.perf_counter() - start_time) * 1000
                    return DetectionResult(
                        is_injection=True,
                        confidence=judge_result.confidence,
                        layer="llm_judge",
                        details={"reasoning": judge_result.reasoning},
                        latency_ms=latency,
                    )

        # No injection detected
        latency = (time.perf_counter() - start_time) * 1000

        # Return the highest confidence seen
        max_confidence = heuristic_result.confidence
        if self.config.use_classifier and "classifier_result" in dir():
            max_confidence = max(max_confidence, classifier_result.confidence)

        return DetectionResult(
            is_injection=False,
            confidence=max_confidence,
            layer="pipeline",
            latency_ms=latency,
        )

    async def scan_batch(
        self,
        contents: list[str],
        context: dict | None = None,
        depth: Literal["fast", "thorough", "adaptive"] = "adaptive",
    ) -> list[DetectionResult]:
        """Scan multiple pieces of content."""
        import asyncio

        tasks = [self.scan(content, context, depth) for content in contents]
        return await asyncio.gather(*tasks)
