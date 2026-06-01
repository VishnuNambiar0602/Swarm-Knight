"""Swarm-Knight - Production-ready multi-LLM collaboration."""

from .models import (
    Agent, AgentRole, Critique, Judgment, Round, Session, SessionStatus,
    SwarmConfig, SwarmResult, TaskPlan, AgentReputation, MemoryEntry,
    MemoryType, Metrics, FREE_MODELS,
)
from .knight import Knight, knight
from .memory import Memory, memory
from .reputation import Reputation, reputation
from .metrics import MetricsCollector, metrics
from .cache import Cache, cache
from .retry import RetryQueue, retry_queue
from .parallel import ParallelEngine
from .providers import OpenSourceProvider, OpenRouterProvider, OllamaProvider, create_provider

__version__ = "2.1.0"

__all__ = [
    "Agent", "AgentRole", "Critique", "Judgment", "Round", "Session",
    "SessionStatus", "SwarmConfig", "SwarmResult", "TaskPlan",
    "AgentReputation", "MemoryEntry", "MemoryType", "Metrics",
    "Knight", "knight",
    "Memory", "memory",
    "Reputation", "reputation",
    "MetricsCollector", "metrics",
    "Cache", "cache",
    "RetryQueue", "retry_queue",
    "ParallelEngine",
    "OpenSourceProvider", "OpenRouterProvider", "OllamaProvider", "create_provider",
]

def cli():
    from .cli import app
    app()
