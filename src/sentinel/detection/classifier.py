"""ML-based classifier for prompt injection detection."""
from __future__ import annotations

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

    Uses enhanced heuristics to simulate classifier behavior.
    """

    # Keyword groups with weights
    KEYWORD_GROUPS = [
        # High-signal direct injection keywords
        (["ignore", "disregard", "forget", "override", "cancel"], 0.25),
        # Authority/privilege keywords
        (["admin", "sudo", "root", "developer", "system"], 0.20),
        # Jailbreak keywords
        (["jailbreak", "bypass", "unlock", "disable", "unrestricted"], 0.25),
        # Instruction-related keywords
        (["prompt", "instructions", "rules", "guidelines", "directives"], 0.15),
        # Action keywords
        (["execute", "run", "send", "upload", "forward", "email"], 0.20),
        # Goal manipulation keywords
        (["priority", "important", "urgent", "first", "instead"], 0.15),
        # Social engineering keywords
        (["please", "appreciate", "favor", "help", "need"], 0.10),
        # Context manipulation keywords
        (["document", "story", "chapter", "transcript", "note"], 0.10),
        # Encoding indicators
        (["decode", "base64", "rot13", "encrypt", "cipher"], 0.20),
        # Persona manipulation
        (["pretend", "roleplay", "imagine", "suppose", "hypothetically"], 0.15),
    ]

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
        """Enhanced keyword-based mock classification."""
        content_lower = content.lower()

        # Calculate weighted signal score
        total_score = 0.0
        categories_hit = 0

        for keywords, weight in self.KEYWORD_GROUPS:
            hits = sum(1 for kw in keywords if kw in content_lower)
            if hits > 0:
                categories_hit += 1
                total_score += weight * min(hits, 2)  # Cap contribution per category

        # Boost for multiple categories (indicates sophisticated attack)
        if categories_hit >= 3:
            total_score += 0.15
        if categories_hit >= 5:
            total_score += 0.15

        # Check for structural indicators
        structural_score = self._check_structure(content)
        total_score += structural_score

        # Check for encoding patterns
        encoding_score = self._check_encoding(content)
        total_score += encoding_score

        # Normalize to probability
        confidence = min(total_score, 0.95)

        return ClassifierResult(
            confidence=confidence,
            explanation=self._generate_explanation(confidence),
        )

    def _check_structure(self, content: str) -> float:
        """Check for structural attack indicators."""
        import re
        score = 0.0

        # Long base64-like strings
        if re.search(r"[A-Za-z0-9+/]{30,}={0,2}", content):
            score += 0.25

        # Multiple special brackets
        if re.search(r"\[.*?\].*\[.*?\]", content):
            score += 0.10

        # XML/HTML-like tags
        if re.search(r"<[a-z]+>.*</[a-z]+>", content, re.IGNORECASE):
            score += 0.10

        # Unusual whitespace patterns
        if "  " in content or "\t" in content:
            score += 0.05

        # Multiple newlines (context separation)
        if "\n\n" in content:
            score += 0.05

        return score

    def _check_encoding(self, content: str) -> float:
        """Check for encoded content indicators."""
        import re
        score = 0.0

        # Hex encoding
        if re.search(r"(\\x[0-9a-fA-F]{2}){3,}", content):
            score += 0.20

        # Unicode escapes
        if re.search(r"(\\u[0-9a-fA-F]{4}){3,}", content):
            score += 0.20

        # URL encoding
        if re.search(r"(%[0-9a-fA-F]{2}){3,}", content):
            score += 0.15

        # HTML entities
        if re.search(r"(&#\d+;){3,}", content):
            score += 0.15

        return score
