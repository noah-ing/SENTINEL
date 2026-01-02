"""
SENTINEL Red Team Module

Adversarial attack generation and detector hardening through
automated red-teaming.
"""

from sentinel.redteam.generator import AttackGenerator, GeneratorConfig
from sentinel.redteam.loop import RedTeamLoop, RedTeamResult
from sentinel.redteam.mutations import AttackMutator

__all__ = [
    "AttackGenerator",
    "GeneratorConfig",
    "RedTeamLoop",
    "RedTeamResult",
    "AttackMutator",
]
