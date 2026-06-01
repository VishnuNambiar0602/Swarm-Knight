"""Swarm data models for multi-LLM collaboration."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ParticipantRole(str, Enum):
    """Roles a swarm participant can take."""
    GENERATOR = "generator"
    CRITIC = "critic"
    REFINER = "refiner"
    COORDINATOR = "coordinator"


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    VLLM = "vllm"
    TOGETHER = "together"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class SwarmParticipant(BaseModel):
    """A single LLM participant in the swarm."""
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    provider: ProviderType
    model: str
    role: ParticipantRole = ParticipantRole.GENERATOR
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    # Provider-specific capabilities
    supports_tools: bool = False
    supports_vision: bool = False
    context_window: int = 128_000


class Critique(BaseModel):
    """A critique from one participant about another's solution."""
    critic_id: str
    target_id: str
    content: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=10.0)


class DebateRound(BaseModel):
    """A single round of debate/refinement."""
    round_number: int
    solutions: dict[str, str] = Field(default_factory=dict)
    critiques: list[Critique] = Field(default_factory=list)
    refinements: dict[str, str] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    consensus_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SwarmStatus(str, Enum):
    """Status of a swarm session."""
    INITIALIZING = "initializing"
    GENERATING = "generating"
    DEBATING = "debating"
    REFINING = "refining"
    CONSENSUS = "consensus"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


class SwarmConfig(BaseModel):
    """Configuration for swarm behavior."""
    max_rounds: int = Field(default=3, ge=1, le=10)
    consensus_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    min_improvement: float = Field(default=0.05, ge=0.0, le=1.0)
    enable_parallel_generation: bool = True
    enable_cross_critique: bool = True
    timeout_seconds: int = Field(default=300, ge=30)


class SwarmSession(BaseModel):
    """A swarm collaboration session."""
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
    # Metadata
    total_tokens_used: int = 0
    total_api_calls: int = 0
    error_message: Optional[str] = None


class SwarmResult(BaseModel):
    """Result of a swarm collaboration."""
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


# Pre-configured free OpenRouter models for e-commerce swarm
FREE_OPENROUTER_MODELS = {
    "planner": SwarmParticipant(
        name="Nemotron Planner",
        provider=ProviderType.OPENROUTER,
        model="nvidia/nemotron-3-super-49b:free",
        role=ParticipantRole.COORDINATOR,
        context_window=1_000_000,
        supports_tools=True,
    ),
    "architect": SwarmParticipant(
        name="GPT-OSS Architect",
        provider=ProviderType.OPENROUTER,
        model="openai/gpt-oss-120b:free",
        role=ParticipantRole.GENERATOR,
        supports_tools=True,
    ),
    "designer": SwarmParticipant(
        name="MiniMax Designer",
        provider=ProviderType.OPENROUTER,
        model="minimax/minimax-m2.5:free",
        role=ParticipantRole.GENERATOR,
        context_window=1_000_000,
    ),
    "stylist": SwarmParticipant(
        name="Gemma Stylist",
        provider=ProviderType.OPENROUTER,
        model="google/gemma-4-31b:free",
        role=ParticipantRole.GENERATOR,
        supports_vision=True,
    ),
    "coder": SwarmParticipant(
        name="Laguna Coder",
        provider=ProviderType.OPENROUTER,
        model="poolside/poolside-laguna-m-1:free",
        role=ParticipantRole.GENERATOR,
        supports_tools=True,
    ),
    "itercoder": SwarmParticipant(
        name="Laguna Fast",
        provider=ProviderType.OPENROUTER,
        model="poolside/poolside-laguna-xs-2:free",
        role=ParticipantRole.REFINER,
    ),
    "assembler": SwarmParticipant(
        name="Kimi Assembler",
        provider=ProviderType.OPENROUTER,
        model="moonshotai/kimi-k2.6:free",
        role=ParticipantRole.GENERATOR,
        context_window=1_000_000,
    ),
    "reviewer": SwarmParticipant(
        name="Nemotron Reviewer",
        provider=ProviderType.OPENROUTER,
        model="nvidia/nemotron-3-super-49b:free",
        role=ParticipantRole.CRITIC,
        context_window=1_000_000,
    ),
    "extractor": SwarmParticipant(
        name="Nemotron Extractor",
        provider=ProviderType.OPENROUTER,
        model="nvidia/nemotron-nano-12b-2-vl:free",
        role=ParticipantRole.GENERATOR,
        supports_vision=True,
    ),
    "tagger": SwarmParticipant(
        name="GLM Tagger",
        provider=ProviderType.OPENROUTER,
        model="z-ai/glm-4.5-air:free",
        role=ParticipantRole.GENERATOR,
        supports_tools=True,
    ),
}


def get_ecommerce_swarm_config() -> list[SwarmParticipant]:
    """Get pre-configured participants for e-commerce development."""
    return [
        FREE_OPENROUTER_MODELS["planner"],
        FREE_OPENROUTER_MODELS["architect"],
        FREE_OPENROUTER_MODELS["designer"],
        FREE_OPENROUTER_MODELS["coder"],
        FREE_OPENROUTER_MODELS["assembler"],
        FREE_OPENROUTER_MODELS["reviewer"],
    ]


def get_coding_swarm_config() -> list[SwarmParticipant]:
    """Get pre-configured participants for general coding tasks."""
    return [
        FREE_OPENROUTER_MODELS["coder"],
        FREE_OPENROUTER_MODELS["itercoder"],
        FREE_OPENROUTER_MODELS["assembler"],
        FREE_OPENROUTER_MODELS["reviewer"],
    ]
