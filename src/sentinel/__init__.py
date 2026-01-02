"""
SENTINEL: Security ENhanced Testing and Injection Neutralization for Evolved Learned agents

Prompt injection detection and defense for agentic AI systems.
"""

__version__ = "0.1.0"

from sentinel.detection.pipeline import SentinelDetector
from sentinel.detection.result import DetectionResult
from sentinel.defense.middleware import SentinelMiddleware
from sentinel.defense.policy import SecurityPolicy

__all__ = [
    "SentinelDetector",
    "DetectionResult",
    "SentinelMiddleware",
    "SecurityPolicy",
]
