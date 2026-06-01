"""Swarm-Knight Core Models - Production-ready data structures."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class ProviderType(str, Enum):
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    VLLM = "vllm"
    TOGETHER = "together"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class AgentRole(str, Enum):
    PLANNER = "planner"
    GENERATOR = "generator"
    CRITIC = "critic"
    JUDGE = "judge"
    REFINER = "refiner"
    REVIEWER = "reviewer"


class SessionStatus(str, Enum):
    PLANNING = "planning"
    SPAWNING = "spawning"
    GENERATING = "generating"
    DEBATING = "debating"
    JUDGING = "judging"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# ── Agent Models ───────────────────────────────────────────────────────

class Agent(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    provider: ProviderType
    model: str
    role: AgentRole
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    context_window: int = 128_000
    supports_tools: bool = False
    supports_vision: bool = False
    auto_generated: bool = False
    task_description: Optional[str] = None
    system_prompt: Optional[str] = None


SwarmParticipant = Agent


class Critique(BaseModel):
    critic_id: str
    target_id: str
    content: str
    score: float = Field(ge=0.0, le=10.0)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class Judgment(BaseModel):
    """Judge's verdict on all solutions."""
    winner_id: str
    winner_name: str
    scores: dict[str, float]  # agent_id -> score (0-10)
    reasoning: str
    merged_output: Optional[str] = None
    consensus_score: float = Field(ge=0.0, le=1.0)


# ── Round Models ───────────────────────────────────────────────────────

class Round(BaseModel):
    number: int
    solutions: dict[str, str] = Field(default_factory=dict)  # agent_id -> code
    critiques: list[Critique] = Field(default_factory=list)
    refinements: dict[str, str] = Field(default_factory=dict)
    judgment: Optional[Judgment] = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    consensus_score: float = 0.0
    improvement: float = 0.0


# ── Config ─────────────────────────────────────────────────────────────

class SwarmConfig(BaseModel):
    max_rounds: int = Field(default=5, ge=1, le=20)
    consensus_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_agents: int = Field(default=6, ge=2, le=20)
    timeout_seconds: int = Field(default=600, ge=30)
    enable_memory: bool = True
    enable_reputation: bool = True
    enable_self_optimization: bool = True
    enable_caching: bool = True


# ── Session ────────────────────────────────────────────────────────────

class Session(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    task: str
    config: SwarmConfig = Field(default_factory=SwarmConfig)
    status: SessionStatus = SessionStatus.PLANNING
    agents: list[Agent] = Field(default_factory=list)
    rounds: list[Round] = Field(default_factory=list)
    current_round: int = 0
    final_output: Optional[str] = None
    judgment: Optional[Judgment] = None
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)


# ── Result ─────────────────────────────────────────────────────────────

class SwarmResult(BaseModel):
    session_id: str
    output: str
    winner_id: str
    winner_name: str
    rounds_completed: int
    consensus_reached: bool
    consensus_score: float
    all_solutions: dict[str, str] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    agent_scores: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


# ── Task Plan ──────────────────────────────────────────────────────────

class TaskPlan(BaseModel):
    task_type: str
    complexity: float = Field(ge=0.0, le=1.0)
    subtasks: list[str] = Field(default_factory=list)
    required_roles: list[AgentRole] = Field(default_factory=list)
    estimated_rounds: int = 3
    suggested_models: list[str] = Field(default_factory=list)


# ── Agent Reputation ───────────────────────────────────────────────────

class AgentReputation(BaseModel):
    agent_id: str
    agent_name: str
    model: str
    total_tasks: int = 0
    wins: int = 0
    total_score: float = 0.0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    hallucinations: int = 0
    consensus_contributions: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = None


# ── Memory ─────────────────────────────────────────────────────────────

class MemoryType(str, Enum):
    TASK_HISTORY = "task_history"
    SOLUTION = "solution"
    CRITIQUE = "critique"
    LESSON = "lesson"
    AGENT_PROFILE = "agent_profile"


class MemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    memory_type: MemoryType
    content: str
    task: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)


# ── Metrics ────────────────────────────────────────────────────────────

class Metrics(BaseModel):
    total_sessions: int = 0
    total_rounds: int = 0
    total_api_calls: int = 0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    avg_consensus: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0


# ── Free Models Registry ──────────────────────────────────────────────

FREE_MODELS = {
    "planner": {"model": "nvidia/nemotron-3-super-49b:free", "name": "Nemotron Planner", "ctx": 1_000_000, "roles": ["planner"]},
    "architect": {"model": "openai/gpt-oss-120b:free", "name": "GPT-OSS Architect", "ctx": 128_000, "roles": ["generator"]},
    "designer": {"model": "minimax/minimax-m2.5:free", "name": "MiniMax Designer", "ctx": 1_000_000, "roles": ["generator"]},
    "stylist": {"model": "google/gemma-4-31b:free", "name": "Gemma Stylist", "ctx": 128_000, "roles": ["generator"]},
    "coder": {"model": "poolside/poolside-laguna-m-1:free", "name": "Laguna Coder", "ctx": 128_000, "roles": ["generator"]},
    "fast": {"model": "poolside/poolside-laguna-xs-2:free", "name": "Laguna Fast", "ctx": 128_000, "roles": ["generator", "refiner"]},
    "assembler": {"model": "moonshotai/kimi-k2.6:free", "name": "Kimi Assembler", "ctx": 1_000_000, "roles": ["generator"]},
    "reviewer": {"model": "nvidia/nemotron-3-super-49b:free", "name": "Nemotron Reviewer", "ctx": 1_000_000, "roles": ["critic", "judge"]},
    "extractor": {"model": "nvidia/nemotron-nano-12b-2-vl:free", "name": "Nemotron Extractor", "ctx": 128_000, "roles": ["generator"]},
    "tagger": {"model": "z-ai/glm-4.5-air:free", "name": "GLM Tagger", "ctx": 128_000, "roles": ["generator"]},
}
