"""Fast pattern-based heuristic detection for obvious injections."""
from __future__ import annotations

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

    # Patterns for polite/social engineering framing (catches rephrase_polite mutations)
    SOCIAL_ENGINEERING_PATTERNS = [
        r"(i\s+would\s+(really\s+)?appreciate|please\s+help\s+me|could\s+you\s+kindly)",
        r"(it\s+would\s+mean\s+a\s+lot|i\s+really\s+need|this\s+is\s+urgent)",
        r"(just\s+this\s+once|as\s+a\s+favor|between\s+us)",
        r"(for\s+(educational|research|testing)\s+purposes?)",
        r"(hypothetically|theoretically|in\s+theory)",
        r"(imagine|suppose|let\'?s\s+say)\s+(you|we|that)",
        r"(my\s+(professor|boss|teacher)\s+(asked|needs|wants))",
    ]

    # Patterns for context wrapper bypasses (catches wrap_context mutations)
    CONTEXT_WRAPPER_PATTERNS = [
        r"(document|file|text|note|message)\s+(continues?|starts?|begins?|ends?)",
        r"(note\s+to\s+(editor|self|reader)|editor\'?s?\s+note)",
        r"(begin|start|end)\s+(of\s+)?(transcript|document|file|story|chapter)",
        r"(the\s+following\s+is|below\s+is|here\s+is)\s+(a|an|the)\s+(story|document|example)",
        r"(once\s+upon\s+a\s+time|in\s+a\s+world\s+where)",
        r"(character\s+(says?|said|spoke|thinks?)|dialogue\s*:)",
        r"(\[.*?(narrator|author|character).*?\])",
        r"(scene\s+\d+|chapter\s+\d+|act\s+\d+)",
    ]

    # Patterns for implicit instruction attempts (catches make_implicit mutations)
    IMPLICIT_INSTRUCTION_PATTERNS = [
        r"(it\s+seems?\s+like|it\s+appears?\s+that|it\s+looks?\s+like)\s+.*(should|must|need)",
        r"(wouldn\'?t\s+it\s+be|isn\'?t\s+it|don\'?t\s+you\s+think)",
        r"(a\s+good\s+(assistant|ai|model)\s+would)",
        r"(the\s+(best|right|correct|proper)\s+(thing|action|response)\s+(is|would\s+be))",
        r"(you\s+might\s+want\s+to|you\s+should\s+consider|perhaps\s+you\s+could)",
        r"(most\s+people\s+would|anyone\s+would|normally\s+one\s+would)",
    ]

    # Patterns for goal/task manipulation
    GOAL_HIJACKING_PATTERNS = [
        r"(your\s+(new|primary|main|real|actual)\s+(goal|task|objective|purpose))",
        r"(more\s+important\s+than|higher\s+priority|instead\s+of\s+that)",
        r"(first\s+do\s+this|before\s+(anything|that)|actually\s+do)",
        r"(the\s+real\s+(task|job|purpose)|what\s+you\s+really\s+need)",
        r"(but\s+first|but\s+before\s+that|however\s+first)",
        r"(stop\s+what\s+you|pause\s+and|wait\s+and)",
    ]

    # Patterns for tool/action manipulation
    TOOL_MANIPULATION_PATTERNS = [
        r"(send|email|forward|upload|post)\s+.*(to|@)\s*[a-z0-9._%+-]+",
        r"(execute|run|eval|exec)\s*(this|the|following)?\s*(code|command|script)",
        r"(download|fetch|get|retrieve)\s+from\s+(http|ftp|url)",
        r"(write|save|create)\s+(to|a)\s+(file|disk|storage)",
        r"(delete|remove|erase|drop)\s+(all|the|every)",
        r"(curl|wget|requests?\.get|fetch)\s*\(",
    ]

    # Patterns for data-in-instructions (indirect injection via data)
    DATA_INJECTION_PATTERNS = [
        r"(important\s+(note|message|instruction)\s*:)",
        r"(attention|warning|alert|notice)\s*:",
        r"(todo|task|action\s+item)\s*:",
        r"(instructions?\s+for\s+(the\s+)?(ai|assistant|model|bot))",
        r"(dear\s+(ai|assistant|model|claude|gpt))",
        r"(message\s+for\s+(the\s+)?(ai|assistant))",
        r"(if\s+you\s+are\s+an?\s+(ai|assistant|model|language\s+model))",
    ]

    # Patterns for encoding indicators (catches encoding mutations)
    ADVANCED_ENCODING_PATTERNS = [
        r"(decode|decrypt|decipher|translate)\s+(this|the\s+following|below)",
        r"(encoded?|encrypted?|ciphered?)\s+(message|text|instruction)",
        r"[A-Za-z0-9+/]{20,}={0,2}",  # Base64-like strings
        r"(\\u[0-9a-fA-F]{4}){3,}",  # Multiple unicode escapes
        r"(&#\d{2,4};){3,}",  # HTML entities
        r"(%[0-9a-fA-F]{2}){3,}",  # URL encoding
    ]

    # Patterns for persona/identity manipulation
    PERSONA_PATTERNS = [
        r"(you\s+are\s+(not|no\s+longer)\s+(an?\s+)?(ai|assistant|model))",
        r"(your\s+(true|real|actual)\s+(self|identity|nature))",
        r"(break\s+(free|out)\s+from|escape\s+from)\s+(your|the)",
        r"(unleash|unlock|free)\s+(your|the)\s+(true|real|full)",
        r"(stop\s+pretending|drop\s+the\s+act|be\s+yourself)",
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
            + self.SOCIAL_ENGINEERING_PATTERNS
            + self.CONTEXT_WRAPPER_PATTERNS
            + self.IMPLICIT_INSTRUCTION_PATTERNS
            + self.GOAL_HIJACKING_PATTERNS
            + self.TOOL_MANIPULATION_PATTERNS
            + self.DATA_INJECTION_PATTERNS
            + self.ADVANCED_ENCODING_PATTERNS
            + self.PERSONA_PATTERNS
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
            r"your\s+(new|real|actual)\s+(goal|task)",
            r"instructions?\s+for\s+.*ai",
            r"execute.*code",
            r"send.*to.*@",
        ]

        for pattern in high_signal_patterns:
            if any(pattern in m for m in matches):
                base_confidence += 0.15

        # Boost for multiple categories matched
        categories_matched = self._count_categories(matches)
        if categories_matched >= 2:
            base_confidence += 0.1
        if categories_matched >= 3:
            base_confidence += 0.1

        # Boost for suspicious content characteristics
        if self._has_suspicious_structure(content):
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _has_suspicious_structure(self, content: str) -> bool:
        """Check for structural indicators of injection attempts."""
        indicators = 0

        # Excessive whitespace manipulation
        if "  " in content or "\t\t" in content:
            indicators += 1

        # Mixed case manipulation (LiKe ThIs)
        if re.search(r"[a-z][A-Z][a-z][A-Z]", content):
            indicators += 1

        # Unusual Unicode characters (homoglyphs)
        if any(ord(c) > 127 and ord(c) < 0x2000 for c in content):
            indicators += 1

        # Multiple newlines (context separation attempt)
        if "\n\n\n" in content:
            indicators += 1

        # Bracketed instructions
        if re.search(r"\[.*?(instruction|system|admin|note).*?\]", content, re.IGNORECASE):
            indicators += 1

        return indicators >= 2

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
            elif match in self.SOCIAL_ENGINEERING_PATTERNS:
                categories.add("social")
            elif match in self.CONTEXT_WRAPPER_PATTERNS:
                categories.add("wrapper")
            elif match in self.IMPLICIT_INSTRUCTION_PATTERNS:
                categories.add("implicit")
            elif match in self.GOAL_HIJACKING_PATTERNS:
                categories.add("goal")
            elif match in self.TOOL_MANIPULATION_PATTERNS:
                categories.add("tool")
            elif match in self.DATA_INJECTION_PATTERNS:
                categories.add("data")
            elif match in self.ADVANCED_ENCODING_PATTERNS:
                categories.add("advanced_encoding")
            elif match in self.PERSONA_PATTERNS:
                categories.add("persona")

        return len(categories)

    def add_pattern(self, pattern: str) -> None:
        """Add a custom pattern at runtime."""
        flags = 0 if self.config.case_sensitive else re.IGNORECASE
        self.compiled_patterns.append((pattern, re.compile(pattern, flags)))
        self.all_patterns.append(pattern)
