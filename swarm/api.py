"""Swarm API endpoints for multi-LLM collaboration."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .models import (
    ParticipantRole,
    ProviderType,
    SwarmConfig,
    SwarmParticipant,
    SwarmStatus,
    get_coding_swarm_config,
    get_ecommerce_swarm_config,
)
from .orchestrator import swarm_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/swarm", tags=["swarm"])


# Request/Response Models

class CreateSwarmRequest(BaseModel):
    """Request to create a new swarm session."""
    task: str
    description: Optional[str] = None
    preset: Optional[str] = Field(None, description="Preset config: 'ecommerce' or 'coding'")
    participants: Optional[list[dict]] = None
    config: Optional[dict] = None


class ParticipantConfig(BaseModel):
    """Configuration for a single participant."""
    name: str
    provider: str = "openrouter"
    model: str
    role: str = "generator"
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class SwarmStatusResponse(BaseModel):
    """Response with swarm session status."""
    id: str
    status: str
    current_round: int
    max_rounds: int
    consensus_reached: bool
    num_participants: int
    rounds_completed: int
    created_at: str


class SwarmResultResponse(BaseModel):
    """Response with swarm result."""
    session_id: str
    output: str
    best_participant_id: str
    best_participant_name: str
    rounds_completed: int
    consensus_reached: bool
    final_consensus_score: float
    duration_seconds: float


# API Endpoints

@router.post("/create", response_model=SwarmStatusResponse)
async def create_swarm(request: CreateSwarmRequest):
    """Create a new swarm session."""
    try:
        # Parse participants if provided
        participants = None
        if request.participants:
            participants = [
                SwarmParticipant(
                    name=p.get("name", "Unnamed"),
                    provider=ProviderType(p.get("provider", "openrouter")),
                    model=p["model"],
                    role=ParticipantRole(p.get("role", "generator")),
                    api_key=p.get("api_key"),
                    base_url=p.get("base_url"),
                )
                for p in request.participants
            ]

        # Parse config if provided
        config = None
        if request.config:
            config = SwarmConfig(**request.config)

        session = await swarm_orchestrator.create_session(
            task=request.task,
            participants=participants,
            config=config,
            preset=request.preset,
        )

        return SwarmStatusResponse(
            id=session.id,
            status=session.status,
            current_round=session.current_round,
            max_rounds=session.config.max_rounds,
            consensus_reached=session.consensus_reached,
            num_participants=len(session.participants),
            rounds_completed=len(session.rounds),
            created_at=session.created_at.isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to create swarm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/start")
async def start_swarm(session_id: str):
    """Start a swarm session."""
    session = swarm_orchestrator.active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status not in (SwarmStatus.INITIALIZING, SwarmStatus.STOPPED):
        raise HTTPException(status_code=400, detail="Session already running")

    # Run in background task
    async def _run():
        try:
            await swarm_orchestrator.run_session(session_id)
        except Exception as e:
            logger.error(f"Swarm session {session_id} failed: {e}")

    asyncio.create_task(_run())

    return {"status": "started", "session_id": session_id}


@router.get("/{session_id}/status", response_model=SwarmStatusResponse)
async def get_status(session_id: str):
    """Get swarm session status."""
    status = await swarm_orchestrator.get_session_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")

    return SwarmStatusResponse(**status)


@router.get("/{session_id}/result", response_model=SwarmResultResponse)
async def get_result(session_id: str):
    """Get swarm result (only if completed)."""
    session = swarm_orchestrator.active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != SwarmStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Session not completed")

    return SwarmResultResponse(
        session_id=session.id,
        output=session.final_output or "",
        best_participant_id=session.best_participant_id or "",
        best_participant_name=swarm_orchestrator._get_participant_name(
            session, session.best_participant_id or ""
        ),
        rounds_completed=len(session.rounds),
        consensus_reached=session.consensus_reached,
        final_consensus_score=session.rounds[-1].consensus_score if session.rounds else 0.0,
        duration_seconds=(
            (session.completed_at - session.created_at).total_seconds()
            if session.completed_at
            else 0.0
        ),
    )


@router.post("/{session_id}/stop")
async def stop_swarm(session_id: str):
    """Stop a running swarm session."""
    await swarm_orchestrator.stop_session(session_id)
    return {"status": "stopped", "session_id": session_id}


@router.delete("/{session_id}")
async def delete_swarm(session_id: str):
    """Delete a swarm session."""
    await swarm_orchestrator.cleanup_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@router.get("/presets")
async def get_presets():
    """Get available swarm presets."""
    return {
        "presets": [
            {
                "id": "ecommerce",
                "name": "E-Commerce Development",
                "description": "Full e-commerce site with specialized roles",
                "num_participants": 6,
            },
            {
                "id": "coding",
                "name": "General Coding",
                "description": "Coding tasks with debate and refinement",
                "num_participants": 4,
            },
        ]
    }


@router.get("/providers")
async def get_providers():
    """Get available LLM providers."""
    return {
        "providers": [
            {
                "id": "openrouter",
                "name": "OpenRouter",
                "description": "Access to many free models",
                "requires_api_key": True,
                "free_models": True,
            },
            {
                "id": "ollama",
                "name": "Ollama",
                "description": "Local LLM inference",
                "requires_api_key": False,
                "requires_local": True,
            },
            {
                "id": "together",
                "name": "Together AI",
                "description": "Cloud LLM inference",
                "requires_api_key": True,
            },
            {
                "id": "groq",
                "name": "Groq",
                "description": "Fast LLM inference",
                "requires_api_key": True,
            },
        ]
    }


@router.get("/models/{provider}")
async def get_models(provider: str):
    """Get available models for a provider."""
    try:
        provider_type = ProviderType(provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    # Return pre-configured free models for OpenRouter
    if provider_type == ProviderType.OPENROUTER:
        return {
            "models": [
                {"id": "nvidia/nemotron-3-super-49b:free", "name": "Nemotron 3 Super", "free": True},
                {"id": "openai/gpt-oss-120b:free", "name": "GPT-OSS 120B", "free": True},
                {"id": "minimax/minimax-m2.5:free", "name": "MiniMax M2.5", "free": True},
                {"id": "google/gemma-4-31b:free", "name": "Gemma 4 31B", "free": True},
                {"id": "poolside/poolside-laguna-m-1:free", "name": "Laguna M.1", "free": True},
                {"id": "poolside/poolside-laguna-xs-2:free", "name": "Laguna XS.2", "free": True},
                {"id": "moonshotai/kimi-k2.6:free", "name": "Kimi K2.6", "free": True},
                {"id": "nvidia/nemotron-nano-12b-2-vl:free", "name": "Nemotron Nano 12B VL", "free": True},
                {"id": "z-ai/glm-4.5-air:free", "name": "GLM 4.5 Air", "free": True},
            ]
        }

    # For other providers, try to query them (requires API key)
    return {"models": [], "note": "API key required to list models"}


# WebSocket endpoint for real-time swarm updates

@router.websocket("/ws/{session_id}")
async def swarm_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time swarm updates."""
    await websocket.accept()

    async def progress_callback(event: str, data: dict):
        try:
            await websocket.send_json({"event": event, "data": data})
        except Exception:
            pass

    swarm_orchestrator.set_progress_callback(session_id, progress_callback)

    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            # Client can send commands like "stop"
            if data == "stop":
                await swarm_orchestrator.stop_session(session_id)
    except WebSocketDisconnect:
        pass
    finally:
        if session_id in swarm_orchestrator._progress_callbacks:
            del swarm_orchestrator._progress_callbacks[session_id]
