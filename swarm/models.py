"""Swarm-Knight Core Models - Extended for consensus, memory, reputation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ParticipantRole(str, Enum):
    GENERATOR = "generator"
    CRITIC = "critic"
    REFINER = "refiner"
    COORDINATOR = "coordinator"
    PLANNER = "planner"
    REVIEWER = "reviewer"


class ProviderType(str, Enum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    VLLM = "vllm"
    TOGETHER = "together"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class SwarmParticipant(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    provider: ProviderType
    model: str
    role: ParticipantRole = ParticipantRole.GENERATOR
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    supports_tools: bool = False
    supports_vision: bool = False
    context_window: int = 128_000
    # Dynamic fields
    auto_generated: bool = False
    task_description: Optional[str] = None


class Critique(BaseModel):
    critic_id: str
    target_id: str
    content: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=10.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class DebateRound(BaseModel):
    round_number: int
    solutions: dict[str, str] = Field(default_factory=dict)
    critiques: list[Critique] = Field(default_factory=list)
    refinements: dict[str, str] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    consensus_score: float = Field(default=0.0, ge=0.0, le=1.0)
    improvement_delta: float = 0.0
    confidence_scores: dict[str, float] = Field(default_factory=dict)


class SwarmStatus(str, Enum):
    INITIALIZING = "initializing"
    PLANNING = "planning"
    GENERATING = "generating"
    DEBATING = "debating"
    REFINING = "refining"
    MERGING = "merging"
    CONSENSUS = "consensus"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


class SwarmConfig(BaseModel):
    max_rounds: int = Field(default=5, ge=1, le=20)
    consensus_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    early_stop_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    min_improvement: float = Field(default=0.02, ge=0.0, le=1.0)
    enable_parallel: bool = True
    enable_memory: bool = True
    enable_reputation: bool = True
    enable_dynamic_agents: bool = True
    max_agents: int = Field(default=8, ge=2, le=20)
    timeout_seconds: int = Field(default=600, ge=30)
    cache_enabled: bool = True


class SwarmSession(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    task: str
    description: Optional[str] = None
    participants: list[SwarmParticipant] = Field(default_factory=list)
    config: SwarmConfig = Field(default_factory=SwarmConfig)
    rounds: list[DebateRound] = Field(default_factory=list)
    current_round: int = 0
    status: SwarmStatus = SwarmStatus.INITIALIZING
    final_output: Optional[str] = None
    best_solution: Optional[str] = None
    best_participant_id: Optional[str] = None
    consensus_reached: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_tokens_used: int = 0
    total_api_calls: int = 0
    error_message: Optional[str] = None
    # Planning metadata
    task_type: Optional[str] = None
    required_capabilities: list[str] = Field(default_factory=list)


class SwarmResult(BaseModel):
    session_id: str
    output: str
    best_participant_id: str
    best_participant_name: str
    rounds_completed: int
    consensus_reached: bool
    final_consensus_score: float
    all_solutions: dict[str, str] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    tokens_used: int = 0
    agent_scores: dict[str, float] = Field(default_factory=dict)
    memory_saved: bool = False


# Agent Reputation Model
class AgentReputation(BaseModel):
    agent_id: str
    agent_name: str
    model: str
    total_tasks: int = 0
    successful_tasks: int = 0
    total_score: float = 0.0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    hallucination_count: int = 0
    consensus_contributions: int = 0
    last_used: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)


# Memory Models
class MemoryType(str, Enum):
    LONG_TERM = "long_term"
    PROJECT = "project"
    AGENT_SPECIFIC = "agent_specific"
    TASK_HISTORY = "task_history"


class MemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    memory_type: MemoryType
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    agent_id: Optional[str] = None
    project_id: Optional[str] = None
    relevance_score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)
    accessed_at: Optional[datetime] = None
    access_count: int = 0


class TaskPlan(BaseModel):
    task_type: str
    subtasks: list[str] = Field(default_factory=list)
    required_roles: list[ParticipantRole] = Field(default_factory=list)
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    estimated_rounds: int = 3


# Free OpenRouter Models
FREE_MODELS = {
    "planner": {"model": "nvidia/nemotron-3-super-49b:free", "name": "Nemotron Planner", "ctx": 1_000_000},
    "architect": {"model": "openai/gpt-oss-120b:free", "name": "GPT-OSS Architect", "ctx": 128_000},
    "designer": {"model": "minimax/minimax-m2.5:free", "name": "MiniMax Designer", "ctx": 1_000_000},
    "stylist": {"model": "google/gemma-4-31b:free", "name": "Gemma Stylist", "ctx": 128_000},
    "coder": {"model": "poolside/poolside-laguna-m-1:free", "name": "Laguna Coder", "ctx": 128_000},
    "fast": {"model": "poolside/poolside-laguna-xs-2:free", "name": "Laguna Fast", "ctx": 128_000},
    "assembler": {"model": "moonshotai/kimi-k2.6:free", "name": "Kimi Assembler", "ctx": 1_000_000},
    "reviewer": {"model": "nvidia/nemotron-3-super-49b:free", "name": "Nemotron Reviewer", "ctx": 1_000_000},
    "extractor": {"model": "nvidia/nemotron-nano-12b-2-vl:free", "name": "Nemotron Extractor", "ctx": 128_000},
    "tagger": {"model": "z-ai/glm-4.5-air:free", "name": "GLM Tagger", "ctx": 128_000},
}
