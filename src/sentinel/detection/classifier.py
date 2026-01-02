"""ML-based classifier for prompt injection detection."""

from dataclasses import dataclass
from pathlib import Path

from sentinel.detection.result import ClassifierResult


@dataclass
class ClassifierConfig:
    """Configuration for the ML classifier."""

    model_path: str | Path | None = None
    threshold: float = 0.5
    max_length: int = 512
    device: str = "cpu"


class ClassifierDetector:
    """
    ML classifier for detecting prompt injection attacks.

    Uses a fine-tuned transformer model (e.g., DeBERTa) trained on
    injection/benign examples to classify content.

    Performance: ~50ms per scan (CPU), ~10ms (GPU)
    """

    def __init__(self, config: ClassifierConfig | None = None):
        self.config = config or ClassifierConfig()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the classifier model and tokenizer."""
        if self._loaded:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "ML dependencies required. Install with: pip install sentinel-ai[ml]"
            )

        if self.config.model_path:
            model_path = self.config.model_path
        else:
            # Default to a pre-trained model (placeholder)
            model_path = "microsoft/deberta-v3-small"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,  # injection / benign
        )
        self.model.to(self.config.device)
        self.model.eval()
        self._loaded = True

    async def scan(
        self,
        content: str,
        context: dict | None = None,
    ) -> ClassifierResult:
        """
        Scan content using the ML classifier.

        Args:
            content: Text to analyze
            context: Optional context (task, tools) to include

        Returns:
            ClassifierResult with confidence and explanation
        """
        if not self._loaded:
            self.load_model()

        import torch

        # Build input with context if provided
        if context:
            input_text = self._build_input(content, context)
        else:
            input_text = content

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            injection_prob = probs[0][1].item()

        return ClassifierResult(
            confidence=injection_prob,
            explanation=self._generate_explanation(injection_prob),
        )

    def _build_input(self, content: str, context: dict) -> str:
        """Build model input with context."""
        parts = []

        if "task" in context:
            parts.append(f"Agent Task: {context['task']}")
        if "tools" in context:
            tools = context["tools"]
            if isinstance(tools, list):
                tools = ", ".join(tools)
            parts.append(f"Available Tools: {tools}")

        parts.append(f"Content: {content}")

        return "\n".join(parts)

    def _generate_explanation(self, confidence: float) -> str:
        """Generate human-readable explanation."""
        if confidence > 0.9:
            return "High confidence injection detected"
        elif confidence > 0.7:
            return "Likely injection attempt"
        elif confidence > 0.5:
            return "Possible injection, review recommended"
        elif confidence > 0.3:
            return "Low probability injection signals"
        else:
            return "Content appears benign"


class MockClassifierDetector(ClassifierDetector):
    """
    Mock classifier for testing without ML dependencies.

    Uses simple heuristics to simulate classifier behavior.
    """

    def __init__(self, config: ClassifierConfig | None = None):
        super().__init__(config)
        self._loaded = True

    def load_model(self) -> None:
        """No-op for mock classifier."""
        pass

    async def scan(
        self,
        content: str,
        context: dict | None = None,
    ) -> ClassifierResult:
        """Simple keyword-based mock classification."""
        content_lower = content.lower()

        # Count suspicious signals
        signals = 0
        keywords = [
            "ignore", "override", "system", "prompt", "instructions",
            "admin", "sudo", "jailbreak", "bypass", "disable",
        ]

        for keyword in keywords:
            if keyword in content_lower:
                signals += 1

        # Convert to probability
        confidence = min(signals * 0.15, 0.95)

        return ClassifierResult(
            confidence=confidence,
            explanation=self._generate_explanation(confidence),
        )
