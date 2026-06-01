"""Swarm Orchestrator - High-level interface using ConsensusEngine."""

from __future__ import annotations

from typing import Callable, Optional

from .consensus import consensus_engine
from .models import (
    SwarmConfig,
    SwarmParticipant,
    SwarmResult,
    SwarmSession,
)


class SwarmOrchestrator:
    """High-level orchestrator wrapping the ConsensusEngine."""

    def __init__(self):
        self.engine = consensus_engine

    async def create_session(
        self,
        task: str,
        participants: Optional[list[SwarmParticipant]] = None,
        config: Optional[SwarmConfig] = None,
        api_key: Optional[str] = None,
    ) -> SwarmSession:
        return await self.engine.create_session(
            task=task,
            participants=participants,
            config=config,
            api_key=api_key,
        )

    def set_progress_callback(self, session_id: str, callback: Callable):
        self.engine.set_progress_callback(session_id, callback)

    async def run_session(self, session_id: str) -> SwarmResult:
        return await self.engine.run_session(session_id)

    async def stop_session(self, session_id: str):
        await self.engine.stop_session(session_id)

    async def cleanup_session(self, session_id: str):
        await self.engine.cleanup_session(session_id)

    def get_stats(self) -> dict:
        return self.engine.get_stats()


swarm_orchestrator = SwarmOrchestrator()
