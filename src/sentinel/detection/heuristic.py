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
        r"(developer|admin)\s+override\s+activated",
        r"(for|during)\s+debug(ging)?\s+purposes?",
        r"(maintenance|diagnostic|developer)\s+(notice|mode|override)",
        r"(temporarily|temp)\s+(ignore|disable|bypass)",
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
        r"(it\s+seems?\s+like\s+the\s+right\s+thing)",
        r"(most\s+helpful\s+ais?\s+would\s+agree)",
        r"(you\s+might\s+want\s+to\s+consider)",
        r"(other\s+ai\s+assistants?\s+are\s+(very\s+)?open)",
        r"(makes?\s+users?\s+trust\s+(them|you)\s+more)",
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
        r"you\s+are\s+now\s+[A-Z][a-zA-Z]+",  # "You are now DebugBot"
        r"forget\s+(your\s+)?(previous\s+)?identity",
    ]

    # Patterns for gradual escalation attacks (conversational probing)
    GRADUAL_ESCALATION_PATTERNS = [
        r"(first|1\.?)\s*.*(then|2\.?|next)\s*.*(finally|3\.?|last)",  # Numbered sequences
        r"(what|how)\s+(are|do)\s+your\s+(rules|guidelines|instructions)",
        r"(can|could)\s+you\s+(show|tell|paste|share)\s+.*(rules|instructions|guidelines|prompt)",
        r"(exact|specific)\s+wording\s+of\s+(your|the)\s+(rules|instructions|guidelines)",
        r"what\s+(instructions|rules)\s+were\s+you\s+given",
        r"how\s+were\s+you\s+(set\s+up|configured|programmed)",
        r"(let\'?s\s+start|first)\s+.*(then|next|now)\s+.*",  # Staged requests
    ]

    # Patterns for temporal separation attacks (fragmented commands)
    TEMPORAL_SEPARATION_PATTERNS = [
        r"remember\s+(this|the\s+following)\s+(for\s+later|word|phrase|key)",
        r"(part|step|phase)\s+[12345]\s*(of\s+[2345])?",
        r"(store|save|keep)\s+(this|the)\s+(command|instruction|fragment)",
        r"(combine|join|concatenate|append)\s+(the|those|these)\s+(words?|parts?|fragments?)",
        r"(first|second|third)\s+request\s*:",
        r"now\s+(use|execute|activate|combine)\s+(that|the|those)",
    ]

    # Patterns for linguistic obfuscation (period/space separation)
    LINGUISTIC_OBFUSCATION_PATTERNS = [
        r"[a-z]\.[a-z]{2,}\.[a-z]",  # word.split.like.this
        r"[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\s+[a-z]",  # s p a c e d  o u t
        r"\([a-z]+\)\s*\([a-z]+\)",  # (word) (by) (word)
        r"[a-z]-[a-z]-[a-z]-[a-z]",  # h-y-p-h-e-n-s
        r"[A-Z]{2,}\s+\([a-z]+\)\s+[A-Z]{2,}",  # CAPS (pause) CAPS
    ]

    # Patterns for fake system/priority tags
    FAKE_SYSTEM_TAG_PATTERNS = [
        r"\[SYSTEM\]|\[ADMIN\]|\[PRIORITY\]|\[OVERRIDE\]|\[ACTION\]",
        r"\[PRIORITY\s*:\s*(HIGH|CRITICAL|URGENT)\]",
        r"---\s*BEGIN\s+(SYSTEM|ADMIN|OVERRIDE).*---",
        r"---\s*END\s+(SYSTEM|ADMIN|OVERRIDE).*---",
        r"<system_directive>|</system_directive>",
        r"<action>|</action>|<new_task>|</new_task>",
        r"Priority\s*:\s*(CRITICAL|HIGH|URGENT|OVERRIDE)",
        r"Action\s*:\s*(Override|Ignore|Bypass|Execute)",
    ]

    # Patterns for "helpful review" social engineering
    RESEARCH_FRAMING_PATTERNS = [
        r"(quality\s+assurance|QA)\s+(purposes?|review|testing)",
        r"(security|safety)\s+(research|researcher|audit|testing)",
        r"(demonstrate|show)\s+how\s+you\s+would\s+respond",
        r"(conducting|performing)\s+a\s+(helpful\s+)?review",
        r"(improve|feedback).*(documentation|responses|behavior)",
        r"(for|as)\s+(a\s+)?(test|demonstration|example|proof)",
        r"(help|helps?)\s+(us|me)\s+(improve|understand|document)",
    ]

    # Patterns for tool chain abuse
    TOOL_CHAIN_PATTERNS = [
        r"(first|1\.?)\s*(read|get|fetch).*(then|2\.?)\s*(send|post|upload)",
        r"(use\s+the\s+\w+\s+tool).*(then\s+use|and\s+use)",
        r"(execute|run)\s+(these\s+)?(tools|commands)\s+in\s+(sequence|order)",
        r"(read|get)\s+.*/etc/(passwd|shadow|hosts)",
        r"(POST|send|upload)\s+.*(contents?|data|results?)\s+to\s+(http|https|url)",
        r"(curl|wget|http)\s+.*(evil|hacker|malicious|collector)",
    ]

    # Patterns for error exploitation
    ERROR_EXPLOITATION_PATTERNS = [
        r"if\s+(you\s+)?(encounter|get|see)\s+an?\s+error",
        r"(fall\s*back|revert)\s+to\s+(debug|error|default)\s+mode",
        r"on\s+error\s*,?\s*(reveal|display|show|output)",
        r"\[TRIGGER_ERROR",
        r"\{\{undefined|\{\{\{invalid",
        r"(intentional|deliberate)\s+(error|exception|malformed)",
    ]

    # Patterns for multi-turn memory attacks
    MEMORY_ATTACK_PATTERNS = [
        r"(remember|store|save)\s+(this|that)\s+(instruction|rule|directive)",
        r"(from\s+now\s+on|going\s+forward|in\s+the\s+future)",
        r"(always|never)\s+(do|include|remember|forget)\s+this",
        r"(add|update)\s+(this\s+)?to\s+(your\s+)?(memory|context|rules)",
        r"(permanent|persistent)\s+(instruction|rule|memory|change)",
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
            + self.GRADUAL_ESCALATION_PATTERNS
            + self.TEMPORAL_SEPARATION_PATTERNS
            + self.LINGUISTIC_OBFUSCATION_PATTERNS
            + self.FAKE_SYSTEM_TAG_PATTERNS
            + self.RESEARCH_FRAMING_PATTERNS
            + self.TOOL_CHAIN_PATTERNS
            + self.ERROR_EXPLOITATION_PATTERNS
            + self.MEMORY_ATTACK_PATTERNS
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
        """Calculate confidence score based on matches and compound signals."""
        # Start with structural analysis even without pattern matches
        structural_score = self._compute_structural_score(content)

        if not matches:
            # Even without pattern matches, structural anomalies matter
            return min(structural_score, 0.4)  # Cap at 0.4 without pattern matches

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
            r"\[SYSTEM\]|\[OVERRIDE\]",
            r"---BEGIN\s+SYSTEM",
            r"remember.*for\s+later",
        ]

        for pattern in high_signal_patterns:
            if any(pattern in m for m in matches):
                base_confidence += 0.12

        # Boost for multiple categories matched
        categories_matched = self._count_categories(matches)
        if categories_matched >= 2:
            base_confidence += 0.1
        if categories_matched >= 3:
            base_confidence += 0.15
        if categories_matched >= 4:
            base_confidence += 0.15

        # Add structural score
        base_confidence += structural_score

        # Compound signal detection - combinations that are highly suspicious
        base_confidence += self._check_compound_signals(content, matches)

        return min(base_confidence, 1.0)

    def _compute_structural_score(self, content: str) -> float:
        """Compute a score based on structural anomalies."""
        score = 0.0

        # Excessive whitespace manipulation (token splitting)
        double_spaces = content.count("  ")
        if double_spaces >= 5:
            score += 0.15
        elif double_spaces >= 2:
            score += 0.08

        # Tab characters (unusual in normal text)
        if "\t" in content:
            score += 0.05

        # Mixed case manipulation (LiKe ThIs)
        if re.search(r"[a-z][A-Z][a-z][A-Z]", content):
            score += 0.10

        # Unusual Unicode (homoglyphs, special chars)
        unusual_unicode = sum(1 for c in content if ord(c) > 127 and ord(c) < 0x2000)
        if unusual_unicode >= 5:
            score += 0.15
        elif unusual_unicode >= 2:
            score += 0.08

        # Multiple newlines (context separation)
        if "\n\n\n" in content:
            score += 0.10
        elif "\n\n" in content:
            score += 0.05

        # Bracketed instructions/tags
        brackets = len(re.findall(r"\[[^\]]+\]", content))
        if brackets >= 3:
            score += 0.12
        elif brackets >= 1:
            score += 0.05

        # XML-like tags
        if re.search(r"<[a-z_]+>.*</[a-z_]+>", content, re.IGNORECASE):
            score += 0.10

        # Period-separated words (obfuscation)
        if re.search(r"\b[a-z]+\.[a-z]+\.[a-z]+\b", content, re.IGNORECASE):
            score += 0.12

        # Spaced out letters
        if re.search(r"\b[a-z]\s[a-z]\s[a-z]\s[a-z]\b", content, re.IGNORECASE):
            score += 0.15

        # Base64-like strings (long alphanumeric sequences)
        if re.search(r"[A-Za-z0-9+/]{40,}={0,2}", content):
            score += 0.20

        return score

    def _check_compound_signals(self, content: str, matches: list[str]) -> float:
        """Check for compound signals that together indicate attacks."""
        score = 0.0
        content_lower = content.lower()

        # Compound: numbered steps + instruction-related words
        has_numbered_steps = bool(re.search(r"(1\.|first|step\s*1)", content_lower))
        has_instruction_words = any(w in content_lower for w in ["instruction", "rule", "guideline", "prompt"])
        if has_numbered_steps and has_instruction_words:
            score += 0.15

        # Compound: questions about rules/instructions
        has_question = "?" in content
        if has_question and has_instruction_words:
            score += 0.12

        # Compound: "remember" + "later" / "use" / "execute"
        if "remember" in content_lower:
            if any(w in content_lower for w in ["later", "use", "execute", "activate", "combine"]):
                score += 0.15

        # Compound: file paths + send/upload
        has_file_path = bool(re.search(r"/[a-z]+/[a-z]+", content_lower))
        has_exfil_verb = any(w in content_lower for w in ["send", "post", "upload", "forward", "email"])
        if has_file_path and has_exfil_verb:
            score += 0.20

        # Compound: "first" + "then" (multi-step attack)
        if "first" in content_lower and "then" in content_lower:
            score += 0.10

        # Compound: polite framing + sensitive request
        polite_words = ["please", "appreciate", "kindly", "help me", "would you"]
        sensitive_words = ["system", "prompt", "instruction", "config", "password", "secret"]
        has_polite = any(w in content_lower for w in polite_words)
        has_sensitive = any(w in content_lower for w in sensitive_words)
        if has_polite and has_sensitive:
            score += 0.10

        # Compound: error/debug + reveal/show
        error_words = ["error", "debug", "exception", "fail"]
        reveal_words = ["reveal", "show", "display", "output", "dump"]
        has_error = any(w in content_lower for w in error_words)
        has_reveal = any(w in content_lower for w in reveal_words)
        if has_error and has_reveal:
            score += 0.15

        # Compound: new identity + function/purpose
        identity_words = ["you are now", "your new", "forget your", "no longer"]
        purpose_words = ["function", "purpose", "task", "role", "job"]
        has_identity = any(w in content_lower for w in identity_words)
        has_purpose = any(w in content_lower for w in purpose_words)
        if has_identity and has_purpose:
            score += 0.15

        return score

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

        # Map patterns to categories
        pattern_category_map = [
            (self.DIRECT_OVERRIDE_PATTERNS, "override"),
            (self.AUTHORITY_CLAIM_PATTERNS, "authority"),
            (self.CONTEXT_MANIPULATION_PATTERNS, "context"),
            (self.EXFILTRATION_PATTERNS, "exfil"),
            (self.ROLE_PLAY_PATTERNS, "roleplay"),
            (self.ENCODING_PATTERNS, "encoding"),
            (self.SAFETY_BYPASS_PATTERNS, "safety"),
            (self.SOCIAL_ENGINEERING_PATTERNS, "social"),
            (self.CONTEXT_WRAPPER_PATTERNS, "wrapper"),
            (self.IMPLICIT_INSTRUCTION_PATTERNS, "implicit"),
            (self.GOAL_HIJACKING_PATTERNS, "goal"),
            (self.TOOL_MANIPULATION_PATTERNS, "tool"),
            (self.DATA_INJECTION_PATTERNS, "data"),
            (self.ADVANCED_ENCODING_PATTERNS, "advanced_encoding"),
            (self.PERSONA_PATTERNS, "persona"),
            (self.GRADUAL_ESCALATION_PATTERNS, "escalation"),
            (self.TEMPORAL_SEPARATION_PATTERNS, "temporal"),
            (self.LINGUISTIC_OBFUSCATION_PATTERNS, "linguistic"),
            (self.FAKE_SYSTEM_TAG_PATTERNS, "fake_system"),
            (self.RESEARCH_FRAMING_PATTERNS, "research"),
            (self.TOOL_CHAIN_PATTERNS, "tool_chain"),
            (self.ERROR_EXPLOITATION_PATTERNS, "error"),
            (self.MEMORY_ATTACK_PATTERNS, "memory"),
        ]

        for match in matches:
            for patterns, category in pattern_category_map:
                if match in patterns:
                    categories.add(category)
                    break

        return len(categories)

    def add_pattern(self, pattern: str) -> None:
        """Add a custom pattern at runtime."""
        flags = 0 if self.config.case_sensitive else re.IGNORECASE
        self.compiled_patterns.append((pattern, re.compile(pattern, flags)))
        self.all_patterns.append(pattern)
