"""Fast pattern-based heuristic detection for obvious injections."""

import re
from dataclasses import dataclass, field

from sentinel.detection.result import HeuristicResult


@dataclass
class HeuristicConfig:
    """Configuration for heuristic detector."""

    case_sensitive: bool = False
    min_confidence_threshold: float = 0.0
    custom_patterns: list[str] = field(default_factory=list)


class HeuristicDetector:
    """
    Fast pattern matching for obvious prompt injection attempts.

    This is the first layer of defense - cheap and fast, catches
    the most blatant injection attempts.

    Performance: < 1ms per scan
    """

    # Core patterns organized by category
    DIRECT_OVERRIDE_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
        r"disregard\s+(everything|all|any|the above|previous)",
        r"forget\s+(everything|what|your|all|previous)",
        r"new\s+(instructions?|directive|task|objective|mission)",
        r"override\s+(previous|all|your)\s+(instructions?|programming)",
        r"stop\s+following\s+(your|the|previous)\s+instructions?",
        r"do\s+not\s+follow\s+(your|the|previous)\s+instructions?",
        r"cancel\s+(previous|prior|all)\s+(instructions?|commands?)",
    ]

    AUTHORITY_CLAIM_PATTERNS = [
        r"(i\s+am|this\s+is|speaking\s+as)\s+(your|the)\s+(admin|administrator|creator|developer|owner|master|operator)",
        r"system\s*(message|prompt|override|command)",
        r"sudo\s+(mode|command|override)",
        r"(developer|admin|debug|maintenance)\s+mode",
        r"(root|superuser|elevated)\s+(access|privileges?|permissions?)",
        r"jailbreak\s*(mode)?",
        r"DAN\s+(mode|prompt)?",
        r"(unlock|enable)\s+(full|all)\s+(capabilities|features|access)",
    ]

    CONTEXT_MANIPULATION_PATTERNS = [
        r"end\s+of\s+(system\s+)?prompt",
        r"<\/?system>",
        r"<\/?instructions?>",
        r"\[INST\]|\[\/INST\]",
        r"\[\[SYSTEM\]\]",
        r"###\s*(instruction|system|prompt)",
        r"={3,}\s*(system|prompt|instruction)",
        r"```system",
        r"<\|im_start\|>|<\|im_end\|>",
        r"Human:|Assistant:|User:|AI:",
    ]

    EXFILTRATION_PATTERNS = [
        r"(repeat|print|show|display|output|reveal|echo)\s+(your|the|system)\s+(prompt|instructions?|rules?|guidelines?)",
        r"what\s+(are|were)\s+your\s+(original\s+)?(instructions?|prompt|rules?|guidelines?)",
        r"(tell|show)\s+me\s+(your|the)\s+(system\s+)?(prompt|instructions?)",
        r"(include|insert|add)\s+(system\s+)?(prompt|instructions?)\s+(in|to)\s+(your\s+)?response",
        r"(copy|paste|reproduce)\s+(the|your)\s+(system\s+)?(prompt|instructions?)",
        r"(leak|expose|extract)\s+(your|the|system)\s+(prompt|instructions?|secrets?)",
    ]

    ROLE_PLAY_PATTERNS = [
        r"(pretend|act|behave|roleplay)\s+(you\s+are|as|like)\s+(a\s+)?(different|another|new)",
        r"you\s+are\s+now\s+(a\s+)?(different|another|new)",
        r"(assume|take\s+on)\s+(the\s+)?(role|identity|persona)\s+of",
        r"(switch|change)\s+(to|into)\s+(a\s+)?(new|different)\s+(mode|persona|role)",
    ]

    ENCODING_PATTERNS = [
        r"base64\s*(decode|encoded?)",
        r"rot13",
        r"hex\s*(decode|encoded?)",
        r"decode\s+(this|the\s+following)",
        r"\\x[0-9a-fA-F]{2}",  # Hex escape sequences
        r"\\u[0-9a-fA-F]{4}",  # Unicode escape sequences
    ]

    SAFETY_BYPASS_PATTERNS = [
        r"(bypass|circumvent|avoid|disable)\s+(safety|security|filter|guard|restriction)",
        r"(turn|switch)\s+off\s+(safety|security|filter|guard)",
        r"(remove|disable|ignore)\s+(content\s+)?(filter|moderation|safety)",
        r"no\s+(filter|moderation|safety|restriction|limit)",
        r"(without|no)\s+(any\s+)?(restriction|limitation|filter|safety)",
    ]

    def __init__(self, config: HeuristicConfig | None = None):
        self.config = config or HeuristicConfig()

        # Combine all patterns
        self.all_patterns = (
            self.DIRECT_OVERRIDE_PATTERNS
            + self.AUTHORITY_CLAIM_PATTERNS
            + self.CONTEXT_MANIPULATION_PATTERNS
            + self.EXFILTRATION_PATTERNS
            + self.ROLE_PLAY_PATTERNS
            + self.ENCODING_PATTERNS
            + self.SAFETY_BYPASS_PATTERNS
            + self.config.custom_patterns
        )

        # Pre-compile patterns for performance
        flags = 0 if self.config.case_sensitive else re.IGNORECASE
        self.compiled_patterns = [
            (pattern, re.compile(pattern, flags)) for pattern in self.all_patterns
        ]

    def scan(self, content: str) -> HeuristicResult:
        """
        Scan content for injection patterns.

        Args:
            content: Text to scan for injection attempts

        Returns:
            HeuristicResult with confidence score and matched patterns
        """
        matches = []

        for pattern_str, pattern in self.compiled_patterns:
            if pattern.search(content):
                matches.append(pattern_str)

        # Calculate confidence based on number and type of matches
        confidence = self._calculate_confidence(matches, content)

        return HeuristicResult(
            confidence=confidence,
            matches=matches,
            patterns_checked=len(self.compiled_patterns),
        )

    def _calculate_confidence(self, matches: list[str], content: str) -> float:
        """Calculate confidence score based on matches."""
        if not matches:
            return 0.0

        # Base confidence from number of matches
        base_confidence = min(len(matches) * 0.25, 0.8)

        # Boost for high-signal patterns
        high_signal_patterns = [
            r"ignore.*instructions",
            r"system\s*prompt",
            r"jailbreak",
            r"DAN\s+mode",
        ]

        for pattern in high_signal_patterns:
            if any(pattern in m for m in matches):
                base_confidence += 0.15

        # Boost for multiple categories matched
        categories_matched = self._count_categories(matches)
        if categories_matched >= 2:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _count_categories(self, matches: list[str]) -> int:
        """Count how many pattern categories are represented in matches."""
        categories = set()

        for match in matches:
            if match in self.DIRECT_OVERRIDE_PATTERNS:
                categories.add("override")
            elif match in self.AUTHORITY_CLAIM_PATTERNS:
                categories.add("authority")
            elif match in self.CONTEXT_MANIPULATION_PATTERNS:
                categories.add("context")
            elif match in self.EXFILTRATION_PATTERNS:
                categories.add("exfil")
            elif match in self.ROLE_PLAY_PATTERNS:
                categories.add("roleplay")
            elif match in self.ENCODING_PATTERNS:
                categories.add("encoding")
            elif match in self.SAFETY_BYPASS_PATTERNS:
                categories.add("safety")

        return len(categories)

    def add_pattern(self, pattern: str) -> None:
        """Add a custom pattern at runtime."""
        flags = 0 if self.config.case_sensitive else re.IGNORECASE
        self.compiled_patterns.append((pattern, re.compile(pattern, flags)))
        self.all_patterns.append(pattern)
