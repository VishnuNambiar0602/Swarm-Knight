"""Swarm Intelligence Module - Multi-LLM collaboration with debate/refinement."""

from .models import (
    SwarmParticipant,
    SwarmSession,
    DebateRound,
    Critique,
    SwarmResult,
)
from .orchestrator import SwarmOrchestrator
from .providers import OpenSourceProvider, OpenRouterProvider, OllamaProvider

__all__ = [
    "SwarmParticipant",
    "SwarmSession", 
    "DebateRound",
    "Critique",
    "SwarmResult",
    "SwarmOrchestrator",
    "OpenSourceProvider",
    "OpenRouterProvider",
    "OllamaProvider",
]

# CLI entry point
def cli():
    from .cli import app
    app()
