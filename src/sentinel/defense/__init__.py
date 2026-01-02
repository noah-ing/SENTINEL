"""Defense layer for protecting agents from prompt injection."""

from sentinel.defense.middleware import SentinelMiddleware, ToolResult
from sentinel.defense.policy import SecurityPolicy
from sentinel.defense.monitor import BehavioralMonitor

__all__ = [
    "SentinelMiddleware",
    "SecurityPolicy",
    "BehavioralMonitor",
    "ToolResult",
]
