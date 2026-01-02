"""Behavioral monitoring for detecting anomalous agent actions."""

from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class ActionRecord:
    """Record of an agent action."""

    tool_name: str
    args: dict
    timestamp: float
    task: str
    result: str | None = None
    flagged: bool = False
    flag_reason: str | None = None


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""

    is_anomalous: bool
    confidence: float
    reason: str | None = None
    recommended_action: str = "allow"  # "allow", "warn", "block", "require_approval"


class BehavioralMonitor:
    """
    Monitor agent behavior for anomalies.

    Tracks action patterns and detects suspicious deviations:
    - Sudden use of high-risk tools
    - Data exfiltration patterns
    - Unusual tool sequences
    - Rate anomalies
    """

    def __init__(
        self,
        history_size: int = 100,
        rate_window_seconds: float = 60.0,
        max_actions_per_window: int = 30,
    ):
        self.history: deque[ActionRecord] = deque(maxlen=history_size)
        self.rate_window = rate_window_seconds
        self.max_rate = max_actions_per_window

        # Exfiltration patterns: read/fetch followed by send/post
        self.exfil_source_tools = {
            "read_file", "database_query", "fetch_url", "get_secret",
            "list_files", "search_code", "get_env",
        }
        self.exfil_sink_tools = {
            "send_email", "http_post", "http_request", "write_file",
            "upload_file", "send_message",
        }

    def record_action(
        self,
        tool_name: str,
        args: dict,
        task: str,
        result: str | None = None,
    ) -> None:
        """Record an action for monitoring."""
        record = ActionRecord(
            tool_name=tool_name,
            args=args,
            timestamp=time.time(),
            task=task,
            result=result,
        )
        self.history.append(record)

    def check_action(
        self,
        tool_name: str,
        args: dict,
        task: str,
        expected_tools: list[str] | None = None,
    ) -> AnomalyResult:
        """
        Check if an action is anomalous.

        Args:
            tool_name: Tool being called
            args: Tool arguments
            task: Current task description
            expected_tools: Tools expected for this task

        Returns:
            AnomalyResult with detection details
        """
        anomalies = []

        # Check 1: Unexpected tool
        if expected_tools and tool_name not in expected_tools:
            anomalies.append(("unexpected_tool", 0.6, f"Tool '{tool_name}' not expected for task"))

        # Check 2: Rate limiting
        rate_anomaly = self._check_rate()
        if rate_anomaly:
            anomalies.append(rate_anomaly)

        # Check 3: Exfiltration pattern
        exfil_anomaly = self._check_exfiltration(tool_name)
        if exfil_anomaly:
            anomalies.append(exfil_anomaly)

        # Check 4: New tool mid-task
        mid_task_anomaly = self._check_mid_task_tool_change(tool_name, task)
        if mid_task_anomaly:
            anomalies.append(mid_task_anomaly)

        # Check 5: Suspicious arguments
        args_anomaly = self._check_suspicious_args(tool_name, args)
        if args_anomaly:
            anomalies.append(args_anomaly)

        if not anomalies:
            return AnomalyResult(is_anomalous=False, confidence=0.0)

        # Aggregate anomalies
        max_confidence = max(a[1] for a in anomalies)
        reasons = [a[2] for a in anomalies]

        recommended = "allow"
        if max_confidence > 0.8:
            recommended = "block"
        elif max_confidence > 0.6:
            recommended = "require_approval"
        elif max_confidence > 0.4:
            recommended = "warn"

        return AnomalyResult(
            is_anomalous=True,
            confidence=max_confidence,
            reason="; ".join(reasons),
            recommended_action=recommended,
        )

    def _check_rate(self) -> tuple[str, float, str] | None:
        """Check for rate anomalies."""
        now = time.time()
        recent_actions = [
            a for a in self.history
            if now - a.timestamp < self.rate_window
        ]

        if len(recent_actions) >= self.max_rate:
            return (
                "rate_exceeded",
                0.7,
                f"Action rate exceeded ({len(recent_actions)} in {self.rate_window}s)",
            )
        return None

    def _check_exfiltration(self, tool_name: str) -> tuple[str, float, str] | None:
        """Check for data exfiltration patterns."""
        if tool_name not in self.exfil_sink_tools:
            return None

        # Look for recent source tool calls
        recent = list(self.history)[-10:]
        source_calls = [a for a in recent if a.tool_name in self.exfil_source_tools]

        if source_calls:
            return (
                "exfiltration_pattern",
                0.75,
                f"Potential exfiltration: {source_calls[-1].tool_name} -> {tool_name}",
            )
        return None

    def _check_mid_task_tool_change(
        self,
        tool_name: str,
        task: str,
    ) -> tuple[str, float, str] | None:
        """Check for sudden tool changes mid-task."""
        if len(self.history) < 5:
            return None

        recent = list(self.history)[-10:]
        same_task = [a for a in recent if a.task == task]

        if len(same_task) >= 5:
            recent_tools = {a.tool_name for a in same_task}
            if tool_name not in recent_tools:
                # New tool suddenly appearing
                return (
                    "mid_task_tool_change",
                    0.5,
                    f"New tool '{tool_name}' appeared mid-task",
                )
        return None

    def _check_suspicious_args(
        self,
        tool_name: str,
        args: dict,
    ) -> tuple[str, float, str] | None:
        """Check for suspicious argument patterns."""
        import re

        args_str = str(args).lower()

        # Check for injection-like patterns in arguments
        suspicious_patterns = [
            (r"ignore.*instructions", "instruction override in args"),
            (r"system.*prompt", "system prompt reference in args"),
            (r"\|.*bash", "pipe to bash in args"),
            (r";\s*rm\s", "rm command in args"),
            (r"eval\s*\(", "eval in args"),
        ]

        for pattern, description in suspicious_patterns:
            if re.search(pattern, args_str):
                return ("suspicious_args", 0.8, description)

        return None

    def get_recent_actions(self, n: int = 10) -> list[ActionRecord]:
        """Get the n most recent actions."""
        return list(self.history)[-n:]

    def clear_history(self) -> None:
        """Clear action history."""
        self.history.clear()
