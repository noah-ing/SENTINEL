"""LangChain integration for SENTINEL."""

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


class SentinelCallbackHandler:
    """
    LangChain callback handler for SENTINEL integration.

    Add this to your agent's callbacks to enable security monitoring.

    Usage:
        from sentinel.integrations.langchain import SentinelCallbackHandler

        handler = SentinelCallbackHandler()
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            callbacks=[handler]
        )
    """

    def __init__(
        self,
        middleware: SentinelMiddleware | None = None,
        block_on_detection: bool = True,
    ):
        self.middleware = middleware or SentinelMiddleware(
            detector=SentinelDetector(DetectorConfig(use_mock_models=True)),
            policy=strict_policy(),
        )
        self.block_on_detection = block_on_detection
        self._current_context: AgentContext | None = None

    def set_context(self, task: str, tools: list[str]) -> None:
        """Set the current agent context."""
        self._current_context = AgentContext(
            current_task=task,
            available_tools=tools,
        )

    async def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts executing."""
        tool_name = serialized.get("name", "unknown")

        context = self._current_context or AgentContext(
            current_task="unknown",
            available_tools=[tool_name],
        )

        result = await self.middleware.check_tool_call(
            tool_name=tool_name,
            tool_args={"input": input_str},
            context=context,
        )

        if result.blocked and self.block_on_detection:
            raise SecurityException(
                f"Tool call blocked: {result.reason}",
                details={"tool": tool_name, "input": input_str[:100]},
            )

        if result.requires_approval:
            logger.warning(
                f"Tool '{tool_name}' requires approval: {result.reason}"
            )

        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"Security warning: {warning}")

    async def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """Called when the agent takes an action."""
        # Log for audit trail
        logger.info(f"Agent action: {action}")

    async def on_chain_start(
        self,
        serialized: dict,
        inputs: dict,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts."""
        # Reset action count for new chain
        self.middleware.reset_action_count()

        # Try to extract task from inputs
        task = inputs.get("input", inputs.get("question", "unknown"))
        if isinstance(task, str):
            self.set_context(task, [])


def secure_agent(
    agent: Any,
    policy: SecurityPolicy | str = "strict",
    config: dict | None = None,
) -> Any:
    """
    Wrap a LangChain agent with SENTINEL protection.

    Args:
        agent: LangChain AgentExecutor or similar
        policy: Security policy ("strict", "permissive") or SecurityPolicy instance
        config: Optional detector configuration

    Returns:
        The agent with SENTINEL callbacks added

    Usage:
        from langchain.agents import create_react_agent, AgentExecutor
        from sentinel.integrations.langchain import secure_agent

        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        secure_executor = secure_agent(executor, policy="strict")
    """
    # Create policy
    if isinstance(policy, str):
        if policy == "strict":
            security_policy = strict_policy()
        else:
            security_policy = SecurityPolicy()
    else:
        security_policy = policy

    # Create detector config
    detector_config = DetectorConfig(**(config or {}))
    if not config:
        detector_config.use_mock_models = True

    # Create middleware
    middleware = SentinelMiddleware(
        detector=SentinelDetector(detector_config),
        policy=security_policy,
    )

    # Create callback handler
    handler = SentinelCallbackHandler(middleware=middleware)

    # Add to agent callbacks
    if hasattr(agent, "callbacks"):
        if agent.callbacks is None:
            agent.callbacks = []
        agent.callbacks.append(handler)
    else:
        logger.warning("Agent does not have callbacks attribute")

    return agent


async def scan_document(
    content: str,
    task: str = "document processing",
    tools: list[str] | None = None,
    depth: str = "adaptive",
) -> dict:
    """
    Standalone function to scan a document for injection.

    Useful for scanning retrieved content before passing to an agent.

    Args:
        content: Document content to scan
        task: Description of the agent's task
        tools: List of tools available to the agent
        depth: Detection depth (fast, thorough, adaptive)

    Returns:
        Dict with scan results

    Usage:
        from sentinel.integrations.langchain import scan_document

        docs = retriever.get_relevant_documents(query)
        for doc in docs:
            result = await scan_document(doc.page_content, task="summarize")
            if result["is_injection"]:
                print(f"Injection detected: {result['details']}")
    """
    detector = SentinelDetector(DetectorConfig(use_mock_models=True))

    result = await detector.scan(
        content,
        context={"task": task, "tools": tools or []},
        depth=depth,
    )

    return {
        "is_injection": result.is_injection,
        "confidence": result.confidence,
        "layer": result.layer,
        "details": result.details,
        "latency_ms": result.latency_ms,
    }
