"""Swarm-Knight - Multi-LLM Collaboration via Debate/Refinement."""

from .models import (
    SwarmParticipant,
    SwarmSession,
    DebateRound,
    Critique,
    SwarmResult,
    SwarmConfig,
    AgentReputation,
    MemoryEntry,
    MemoryType,
    TaskPlan,
    ParticipantRole,
    ProviderType,
    SwarmStatus,
)
from .consensus import ConsensusEngine, consensus_engine
from .orchestrator import SwarmOrchestrator, swarm_orchestrator
from .memory import MemorySystem, memory_system
from .reputation import ReputationSystem, reputation_system
from .dynamic_agents import DynamicAgentGenerator, dynamic_generator
from .parallel import ParallelEngine, parallel_engine, TaskResult
from .cache import CacheSystem, cache_system
from .providers import OpenSourceProvider, OpenRouterProvider, OllamaProvider

__all__ = [
    "SwarmParticipant", "SwarmSession", "DebateRound", "Critique", "SwarmResult",
    "SwarmConfig", "AgentReputation", "MemoryEntry", "MemoryType", "TaskPlan",
    "ParticipantRole", "ProviderType", "SwarmStatus",
    "ConsensusEngine", "consensus_engine",
    "SwarmOrchestrator", "swarm_orchestrator",
    "MemorySystem", "memory_system",
    "ReputationSystem", "reputation_system",
    "DynamicAgentGenerator", "dynamic_generator",
    "ParallelEngine", "parallel_engine", "TaskResult",
    "CacheSystem", "cache_system",
    "OpenSourceProvider", "OpenRouterProvider", "OllamaProvider",
]

__version__ = "2.0.0"


def cli():
    from .cli import app
    app()
