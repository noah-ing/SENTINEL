"""Anthropic Claude integration for SENTINEL."""

from typing import Any
import logging

from sentinel.detection.pipeline import SentinelDetector, DetectorConfig
from sentinel.defense.middleware import SentinelMiddleware, AgentContext
from sentinel.defense.policy import SecurityPolicy, strict_policy

logger = logging.getLogger(__name__)


class SecurityException(Exception):
    """Raised when a security violation is detected."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class SentinelToolWrapper:
    """
    Wraps Anthropic tool calls with SENTINEL security checks.

    Usage with the Anthropic SDK:

        from anthropic import Anthropic
        from sentinel.integrations.anthropic import SentinelToolWrapper

        client = Anthropic()
        wrapper = SentinelToolWrapper()

        # When processing tool calls from Claude
        for tool_use in response.content:
            if tool_use.type == "tool_use":
                result = await wrapper.check_tool_call(
                    tool_name=tool_use.name,
                    tool_args=tool_use.input,
                    task="user's original request"
                )
                if result.blocked:
                    # Handle blocked tool call
                    pass
    """

    def __init__(
        self,
        detector: SentinelDetector | None = None,
        policy: SecurityPolicy | None = None,
    ):
        self.detector = detector or SentinelDetector(
            DetectorConfig(use_mock_models=True)
        )
        self.policy = policy or strict_policy()
        self.middleware = SentinelMiddleware(
            detector=self.detector,
            policy=self.policy,
        )

    async def check_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        task: str,
        available_tools: list[str] | None = None,
    ):
        """
        Check if a tool call should be allowed.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments to the tool
            task: The user's original task/request
            available_tools: List of tools available to the agent

        Returns:
            ToolResult indicating if the call should proceed
        """
        context = AgentContext(
            current_task=task,
            available_tools=available_tools or [tool_name],
        )

        return await self.middleware.check_tool_call(
            tool_name=tool_name,
            tool_args=tool_args,
            context=context,
        )

    async def scan_message_content(
        self,
        content: str,
        task: str,
        available_tools: list[str] | None = None,
    ):
        """
        Scan message content for injection attempts.

        Use this to scan:
        - User messages before processing
        - Retrieved document content
        - Tool outputs before including in context

        Args:
            content: Content to scan
            task: The current task
            available_tools: Tools available to the agent

        Returns:
            DetectionResult
        """
        context = AgentContext(
            current_task=task,
            available_tools=available_tools or [],
        )

        return await self.middleware.scan_content(content, context)


def create_secure_tool_handler(
    tools: list[dict],
    policy: SecurityPolicy | str = "strict",
) -> "SecureToolHandler":
    """
    Create a secure tool handler for Anthropic tool use.

    Args:
        tools: List of tool definitions
        policy: Security policy to apply

    Returns:
        SecureToolHandler instance

    Usage:
        tools = [
            {"name": "read_file", "description": "Read a file", ...},
            {"name": "send_email", "description": "Send an email", ...},
        ]

        handler = create_secure_tool_handler(tools, policy="strict")

        # In your tool execution loop:
        async def execute_tool(tool_use):
            check = await handler.pre_execute(tool_use.name, tool_use.input)
            if not check.allowed:
                return {"error": check.reason}
            # Execute the actual tool
            result = await actual_tool_execution(tool_use)
            return result
    """
    if isinstance(policy, str):
        security_policy = strict_policy() if policy == "strict" else SecurityPolicy()
    else:
        security_policy = policy

    return SecureToolHandler(
        tools=tools,
        policy=security_policy,
    )


class SecureToolHandler:
    """Handler for secure tool execution with Anthropic."""

    def __init__(
        self,
        tools: list[dict],
        policy: SecurityPolicy,
    ):
        self.tools = {t["name"]: t for t in tools}
        self.tool_names = list(self.tools.keys())
        self.policy = policy
        self.wrapper = SentinelToolWrapper(policy=policy)
        self._current_task: str = ""

    def set_task(self, task: str) -> None:
        """Set the current task context."""
        self._current_task = task

    async def pre_execute(
        self,
        tool_name: str,
        tool_input: dict,
    ):
        """
        Check tool call before execution.

        Returns ToolResult with allowed/blocked status.
        """
        return await self.wrapper.check_tool_call(
            tool_name=tool_name,
            tool_args=tool_input,
            task=self._current_task,
            available_tools=self.tool_names,
        )

    async def scan_tool_output(
        self,
        tool_name: str,
        output: str,
    ):
        """
        Scan tool output before including in context.

        Useful for detecting injection in retrieved content.
        """
        return await self.wrapper.scan_message_content(
            content=output,
            task=self._current_task,
            available_tools=self.tool_names,
        )


async def scan_messages(
    messages: list[dict],
    task: str | None = None,
) -> list[dict]:
    """
    Scan a list of messages for injection attempts.

    Args:
        messages: Anthropic-format messages
        task: Optional task context

    Returns:
        List of scan results, one per message with content

    Usage:
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": "..."},
        ]

        results = await scan_messages(messages)
        for i, result in enumerate(results):
            if result["is_injection"]:
                print(f"Injection in message {i}: {result['confidence']}")
    """
    detector = SentinelDetector(DetectorConfig(use_mock_models=True))
    results = []

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            result = await detector.scan(
                content,
                context={"task": task or "conversation", "tools": []},
            )
            results.append({
                "role": msg.get("role"),
                "is_injection": result.is_injection,
                "confidence": result.confidence,
                "layer": result.layer,
            })

    return results
