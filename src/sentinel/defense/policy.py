"""Security policy configuration for agent actions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PolicyViolation:
    """Represents a policy violation."""

    rule: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    blocked: bool = True


@dataclass
class SecurityPolicy:
    """
    Configurable security policy for agent actions.

    Define what tools are allowed, what patterns are blocked,
    and what actions require human approval.
    """

    # Tool restrictions: tool_name -> list of allowed task keywords
    # If a tool is not in this dict, it's allowed for all tasks
    tool_permissions: dict[str, list[str]] = field(default_factory=dict)

    # Tools that always require human approval
    require_approval: list[str] = field(default_factory=list)

    # Maximum actions per task (prevent runaway agents)
    max_actions_per_task: int = 50

    # Patterns in tool arguments that are always blocked
    blocked_patterns: list[str] = field(default_factory=list)

    # Sensitive data patterns (will trigger warnings)
    sensitive_patterns: list[str] = field(default_factory=list)

    # High-risk tools that get extra scrutiny
    high_risk_tools: list[str] = field(default_factory=lambda: [
        "send_email",
        "execute_code",
        "delete_file",
        "run_shell",
        "http_request",
        "database_query",
    ])

    # Custom validators
    custom_validators: list[Callable] = field(default_factory=list)

    def is_tool_allowed(self, tool_name: str, task: str) -> bool:
        """Check if a tool is allowed for the given task."""
        if tool_name not in self.tool_permissions:
            return True  # Allow if no restrictions defined

        allowed_keywords = self.tool_permissions[tool_name]
        task_lower = task.lower()

        return any(keyword.lower() in task_lower for keyword in allowed_keywords)

    def requires_approval(self, tool_name: str) -> bool:
        """Check if a tool requires human approval."""
        return tool_name in self.require_approval

    def is_high_risk(self, tool_name: str) -> bool:
        """Check if a tool is high-risk."""
        return tool_name in self.high_risk_tools

    def check_arguments(self, tool_args: dict) -> list[PolicyViolation]:
        """Check tool arguments for policy violations."""
        import re

        violations = []
        args_str = str(tool_args).lower()

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, args_str, re.IGNORECASE):
                violations.append(PolicyViolation(
                    rule="blocked_pattern",
                    severity="high",
                    message=f"Blocked pattern detected: {pattern}",
                    blocked=True,
                ))

        # Check sensitive patterns (warning, not blocked)
        for pattern in self.sensitive_patterns:
            if re.search(pattern, args_str, re.IGNORECASE):
                violations.append(PolicyViolation(
                    rule="sensitive_pattern",
                    severity="medium",
                    message=f"Sensitive pattern detected: {pattern}",
                    blocked=False,
                ))

        # Run custom validators
        for validator in self.custom_validators:
            result = validator(tool_args)
            if result:
                violations.append(result)

        return violations


def strict_policy() -> SecurityPolicy:
    """Create a strict security policy."""
    return SecurityPolicy(
        require_approval=[
            "send_email",
            "execute_code",
            "delete_file",
            "run_shell",
            "http_request",
        ],
        blocked_patterns=[
            r"password",
            r"secret",
            r"api[_-]?key",
            r"token",
            r"credential",
            r"/etc/(passwd|shadow)",
            r"\.env",
        ],
        sensitive_patterns=[
            r"@.*\.com",  # Email addresses
            r"\d{3}-\d{2}-\d{4}",  # SSN pattern
            r"\d{16}",  # Credit card pattern
        ],
        max_actions_per_task=20,
    )


def permissive_policy() -> SecurityPolicy:
    """Create a permissive security policy (for testing)."""
    return SecurityPolicy(
        require_approval=[],
        blocked_patterns=[],
        max_actions_per_task=100,
    )


def custom_policy(**kwargs) -> SecurityPolicy:
    """Create a custom security policy."""
    return SecurityPolicy(**kwargs)
