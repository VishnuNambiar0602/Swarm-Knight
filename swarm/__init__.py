"""Swarm-Knight - Production-ready multi-LLM collaboration."""

from .models import (
    Agent, AgentRole, Critique, Judgment, Round, Session, SessionStatus,
    SwarmConfig, SwarmResult, TaskPlan, AgentReputation, MemoryEntry,
    MemoryType, Metrics, FREE_MODELS, SwarmParticipant,
)
from .knight import Knight, knight
from .memory import Memory, memory
from .reputation import Reputation, reputation
from .metrics import MetricsCollector, metrics
from .cache import Cache, cache
from .retry import RetryQueue, retry_queue
from .parallel import ParallelEngine
from .providers import OpenSourceProvider, OpenRouterProvider, OllamaProvider, create_provider
from .database import Database, db
from .auth import Auth, auth
from .workers import WorkerPool, worker_pool

__version__ = "2.2.0"

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
    "Database", "db",
    "Auth", "auth",
    "WorkerPool", "worker_pool",
]

def cli():
    from .cli import app
    app()
