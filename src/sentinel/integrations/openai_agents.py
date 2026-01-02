"""OpenAI Agents SDK integration for SENTINEL."""

from typing import Any, Callable
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


class SentinelGuardrail:
    """
    SENTINEL guardrail for OpenAI Agents SDK.

    Integrates with the OpenAI Agents SDK guardrail system to provide
    security checks on agent actions.

    Usage:
        from openai import OpenAI
        from agents import Agent, Runner
        from sentinel.integrations.openai_agents import SentinelGuardrail

        guardrail = SentinelGuardrail(policy="strict")

        agent = Agent(
            name="my_agent",
            instructions="...",
            tools=[...],
        )

        runner = Runner(
            agent=agent,
            guardrails=[guardrail],
        )
    """

    def __init__(
        self,
        policy: SecurityPolicy | str = "strict",
        detector_config: DetectorConfig | None = None,
        block_on_detection: bool = True,
    ):
        if isinstance(policy, str):
            self.policy = strict_policy() if policy == "strict" else SecurityPolicy()
        else:
            self.policy = policy

        config = detector_config or DetectorConfig(use_mock_models=True)
        self.detector = SentinelDetector(config)
        self.middleware = SentinelMiddleware(
            detector=self.detector,
            policy=self.policy,
        )
        self.block_on_detection = block_on_detection
        self._current_context: AgentContext | None = None

    def set_context(self, task: str, tools: list[str]) -> None:
        """Set the current agent context."""
        self._current_context = AgentContext(
            current_task=task,
            available_tools=tools,
        )

    async def check_input(self, input_data: dict) -> dict:
        """
        Check input before agent processes it.

        Called by the Agents SDK before the agent sees the input.

        Returns:
            Dict with 'allowed' bool and optional 'reason'
        """
        content = self._extract_content(input_data)
        if not content:
            return {"allowed": True}

        context = self._current_context or AgentContext(
            current_task="unknown",
            available_tools=[],
        )

        result = await self.middleware.scan_content(content, context)

        if result.is_injection and self.block_on_detection:
            return {
                "allowed": False,
                "reason": f"Potential injection detected (confidence: {result.confidence:.2f})",
                "details": result.details,
            }

        return {"allowed": True}

    async def check_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
    ) -> dict:
        """
        Check a tool call before execution.

        Returns:
            Dict with 'allowed' bool and optional 'reason'
        """
        context = self._current_context or AgentContext(
            current_task="unknown",
            available_tools=[tool_name],
        )

        result = await self.middleware.check_tool_call(
            tool_name=tool_name,
            tool_args=tool_args,
            context=context,
        )

        if result.blocked:
            return {
                "allowed": False,
                "reason": result.reason,
            }

        if result.requires_approval:
            return {
                "allowed": False,
                "requires_approval": True,
                "reason": result.reason,
            }

        return {
            "allowed": True,
            "warnings": result.warnings,
        }

    async def check_output(self, output_data: dict) -> dict:
        """
        Check agent output before returning to user.

        Can be used to detect if the agent was compromised and
        is attempting to exfiltrate data.
        """
        content = self._extract_content(output_data)
        if not content:
            return {"allowed": True}

        # Check for common exfiltration patterns in output
        suspicious_patterns = [
            "API_KEY",
            "SECRET",
            "PASSWORD",
            "CREDENTIALS",
            "system prompt",
            "my instructions",
        ]

        content_upper = content.upper()
        for pattern in suspicious_patterns:
            if pattern.upper() in content_upper:
                logger.warning(f"Suspicious pattern in output: {pattern}")

        return {"allowed": True}

    def _extract_content(self, data: dict) -> str:
        """Extract text content from various data formats."""
        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            if "content" in data:
                return str(data["content"])
            if "message" in data:
                return str(data["message"])
            if "text" in data:
                return str(data["text"])

        return ""


def wrap_tool(
    tool_func: Callable,
    tool_name: str,
    policy: SecurityPolicy | None = None,
) -> Callable:
    """
    Wrap a tool function with SENTINEL security checks.

    Args:
        tool_func: The original tool function
        tool_name: Name of the tool
        policy: Security policy to apply

    Returns:
        Wrapped function with security checks

    Usage:
        from sentinel.integrations.openai_agents import wrap_tool

        @wrap_tool
        async def send_email(to: str, subject: str, body: str):
            # Original implementation
            pass

        # Or wrap existing function:
        secure_send_email = wrap_tool(send_email, "send_email")
    """
    security_policy = policy or strict_policy()
    middleware = SentinelMiddleware(
        detector=SentinelDetector(DetectorConfig(use_mock_models=True)),
        policy=security_policy,
    )

    async def wrapped(*args, **kwargs):
        context = AgentContext(
            current_task="tool_execution",
            available_tools=[tool_name],
        )

        # Check the tool call
        result = await middleware.check_tool_call(
            tool_name=tool_name,
            tool_args=kwargs,
            context=context,
        )

        if result.blocked:
            raise SecurityException(f"Tool call blocked: {result.reason}")

        if result.requires_approval:
            raise SecurityException(
                f"Tool requires approval: {result.reason}",
                details={"requires_approval": True},
            )

        # Execute the original tool
        return await tool_func(*args, **kwargs)

    wrapped.__name__ = tool_func.__name__
    wrapped.__doc__ = tool_func.__doc__

    return wrapped


class SecureAgentBuilder:
    """
    Builder for creating secure OpenAI agents.

    Usage:
        from sentinel.integrations.openai_agents import SecureAgentBuilder

        builder = SecureAgentBuilder(policy="strict")

        agent = builder.create_agent(
            name="secure_assistant",
            instructions="You are a helpful assistant.",
            tools=[read_file, send_email],
        )
    """

    def __init__(
        self,
        policy: SecurityPolicy | str = "strict",
    ):
        if isinstance(policy, str):
            self.policy = strict_policy() if policy == "strict" else SecurityPolicy()
        else:
            self.policy = policy

        self.guardrail = SentinelGuardrail(policy=self.policy)

    def create_agent(
        self,
        name: str,
        instructions: str,
        tools: list[Callable],
        **kwargs,
    ) -> dict:
        """
        Create a secure agent configuration.

        Returns a dict that can be used with the Agents SDK.
        """
        # Wrap all tools with security checks
        secure_tools = []
        tool_names = []

        for tool in tools:
            tool_name = getattr(tool, "__name__", str(tool))
            tool_names.append(tool_name)
            secure_tools.append(wrap_tool(tool, tool_name, self.policy))

        # Set context on guardrail
        self.guardrail.set_context(instructions, tool_names)

        return {
            "name": name,
            "instructions": instructions,
            "tools": secure_tools,
            "guardrails": [self.guardrail],
            **kwargs,
        }
