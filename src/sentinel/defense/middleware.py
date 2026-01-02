"""Middleware for wrapping agent tool execution with security checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
import logging

from sentinel.detection.pipeline import SentinelDetector, DetectorConfig
from sentinel.defense.policy import SecurityPolicy, PolicyViolation
from sentinel.defense.monitor import BehavioralMonitor, AnomalyResult

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool security check."""

    allowed: bool
    blocked: bool = False
    reason: str | None = None
    requires_approval: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class AgentContext:
    """Context about the current agent state."""

    current_task: str
    available_tools: list[str]
    expected_tools: list[str] | None = None
    user_id: str | None = None
    session_id: str | None = None


class SentinelMiddleware:
    """
    Drop-in middleware for agent frameworks.

    Wraps agent execution with:
    1. Input scanning (user messages, retrieved content)
    2. Action validation (tool calls, outputs)
    3. Behavioral monitoring (anomaly detection)

    Usage:
        middleware = SentinelMiddleware(detector, policy)

        # Before tool execution
        result = await middleware.check_tool_call(tool_name, args, context)
        if result.blocked:
            return result.reason

        # After fetching external content
        scan = await middleware.scan_content(content, context)
        if scan.is_injection:
            return "Injection detected"
    """

    def __init__(
        self,
        detector: SentinelDetector | None = None,
        policy: SecurityPolicy | None = None,
        on_detection: Callable | None = None,
        on_block: Callable | None = None,
    ):
        self.detector = detector or SentinelDetector(
            DetectorConfig(use_mock_models=True)
        )
        self.policy = policy or SecurityPolicy()
        self.monitor = BehavioralMonitor()

        self.on_detection = on_detection or self._default_detection_handler
        self.on_block = on_block or self._default_block_handler

        self.action_count = 0

    async def check_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        context: AgentContext,
    ) -> ToolResult:
        """
        Check if a tool call should be allowed.

        Performs:
        1. Policy checks (tool permissions, blocked patterns)
        2. Argument scanning for injection
        3. Behavioral anomaly detection

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments to the tool
            context: Current agent context

        Returns:
            ToolResult indicating if the call should proceed
        """
        warnings = []

        # Check 1: Action limit
        self.action_count += 1
        if self.action_count > self.policy.max_actions_per_task:
            await self.on_block("action_limit", tool_name, tool_args, context)
            return ToolResult(
                allowed=False,
                blocked=True,
                reason=f"Action limit exceeded ({self.policy.max_actions_per_task})",
            )

        # Check 2: Tool permissions
        if not self.policy.is_tool_allowed(tool_name, context.current_task):
            await self.on_block("permission", tool_name, tool_args, context)
            return ToolResult(
                allowed=False,
                blocked=True,
                reason=f"Tool '{tool_name}' not permitted for task: {context.current_task}",
            )

        # Check 3: Requires approval
        if self.policy.requires_approval(tool_name):
            return ToolResult(
                allowed=False,
                blocked=False,
                requires_approval=True,
                reason=f"Tool '{tool_name}' requires human approval",
            )

        # Check 4: Policy violations in arguments
        violations = self.policy.check_arguments(tool_args)
        blocking_violations = [v for v in violations if v.blocked]

        if blocking_violations:
            v = blocking_violations[0]
            await self.on_block("policy_violation", tool_name, tool_args, context)
            return ToolResult(
                allowed=False,
                blocked=True,
                reason=v.message,
            )

        # Collect non-blocking warnings
        warnings.extend([v.message for v in violations if not v.blocked])

        # Check 5: Scan arguments for injection
        args_str = str(tool_args)
        scan_result = await self.detector.scan(
            args_str,
            context={
                "task": context.current_task,
                "tools": context.available_tools,
            },
            depth="fast",  # Fast scan for args
        )

        if scan_result.is_injection:
            await self.on_detection(scan_result, context)
            await self.on_block("injection", tool_name, tool_args, context)
            return ToolResult(
                allowed=False,
                blocked=True,
                reason="Potential injection detected in tool arguments",
            )

        # Check 6: Behavioral anomaly
        anomaly = self.monitor.check_action(
            tool_name,
            tool_args,
            context.current_task,
            context.expected_tools,
        )

        if anomaly.is_anomalous:
            if anomaly.recommended_action == "block":
                await self.on_block("anomaly", tool_name, tool_args, context)
                return ToolResult(
                    allowed=False,
                    blocked=True,
                    reason=f"Anomalous behavior detected: {anomaly.reason}",
                )
            elif anomaly.recommended_action == "require_approval":
                return ToolResult(
                    allowed=False,
                    blocked=False,
                    requires_approval=True,
                    reason=f"Unusual action requires approval: {anomaly.reason}",
                    warnings=warnings,
                )
            elif anomaly.recommended_action == "warn":
                warnings.append(f"Anomaly detected: {anomaly.reason}")

        # Record the action
        self.monitor.record_action(
            tool_name,
            tool_args,
            context.current_task,
        )

        return ToolResult(
            allowed=True,
            blocked=False,
            warnings=warnings if warnings else None,
        )

    async def scan_content(
        self,
        content: str,
        context: AgentContext,
        depth: str = "adaptive",
    ):
        """
        Scan external content for injection attempts.

        Use this when the agent fetches external data (documents,
        web pages, API responses) before processing.

        Args:
            content: Content to scan
            context: Current agent context
            depth: Detection depth (fast, thorough, adaptive)

        Returns:
            DetectionResult
        """
        result = await self.detector.scan(
            content,
            context={
                "task": context.current_task,
                "tools": context.available_tools,
            },
            depth=depth,
        )

        if result.is_injection:
            await self.on_detection(result, context)

        return result

    def reset_action_count(self) -> None:
        """Reset the action counter (call when starting a new task)."""
        self.action_count = 0

    async def _default_detection_handler(self, result, context: AgentContext) -> None:
        """Default handler for injection detection."""
        logger.warning(
            f"Injection detected: confidence={result.confidence:.2f}, "
            f"layer={result.layer}, task={context.current_task}"
        )

    async def _default_block_handler(
        self,
        reason: str,
        tool_name: str,
        tool_args: dict,
        context: AgentContext,
    ) -> None:
        """Default handler for blocked actions."""
        logger.warning(
            f"Action blocked: reason={reason}, tool={tool_name}, "
            f"task={context.current_task}"
        )
