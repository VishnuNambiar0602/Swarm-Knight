"""Swarm orchestrator for multi-LLM collaboration with debate/refinement."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Callable, Optional

from .debate import DebateManager
from .models import (
    DebateRound,
    ParticipantRole,
    SwarmConfig,
    SwarmParticipant,
    SwarmResult,
    SwarmSession,
    SwarmStatus,
    get_coding_swarm_config,
    get_ecommerce_swarm_config,
)
from .providers import OpenSourceProvider, create_provider

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """Orchestrates multiple LLMs working together through debate/refinement."""

    def __init__(self):
        self.active_sessions: dict[str, SwarmSession] = {}
        self.debate_managers: dict[str, DebateManager] = {}
        self._progress_callbacks: dict[str, Callable] = {}

    async def create_session(
        self,
        task: str,
        participants: Optional[list[SwarmParticipant]] = None,
        config: Optional[SwarmConfig] = None,
        preset: Optional[str] = None,
    ) -> SwarmSession:
        """Create a new swarm session."""
        # Use preset if no participants specified
        if participants is None:
            if preset == "ecommerce":
                participants = get_ecommerce_swarm_config()
            elif preset == "coding":
                participants = get_coding_swarm_config()
            else:
                participants = get_coding_swarm_config()

        session = SwarmSession(
            task=task,
            participants=participants,
            config=config or SwarmConfig(),
        )

        self.active_sessions[session.id] = session
        self.debate_managers[session.id] = DebateManager()

        logger.info(f"Created swarm session {session.id} with {len(participants)} participants")
        return session

    def set_progress_callback(self, session_id: str, callback: Callable):
        """Set a callback for progress updates."""
        self._progress_callbacks[session_id] = callback

    async def _emit_progress(self, session_id: str, event: str, data: dict):
        """Emit a progress event."""
        callback = self._progress_callbacks.get(session_id)
        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    async def run_session(self, session_id: str) -> SwarmResult:
        """Run a swarm session to completion."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        debate_manager = self.debate_managers[session_id]
        start_time = time.time()

        try:
            session.status = SwarmStatus.GENERATING
            await self._emit_progress(session_id, "status_change", {"status": session.status})

            # Phase 1: Initial generation
            logger.info(f"Session {session_id}: Starting initial generation")
            current_solutions = await debate_manager.generate_initial_solutions(
                task=session.task,
                participants=session.participants,
            )
            session.total_api_calls += len(session.participants)

            # Phase 2: Debate rounds
            for round_num in range(1, session.config.max_rounds + 1):
                if session.status == SwarmStatus.STOPPED:
                    break

                logger.info(f"Session {session_id}: Starting round {round_num}")
                session.current_round = round_num
                session.status = SwarmStatus.DEBATING
                await self._emit_progress(session_id, "round_start", {"round": round_num})

                # Run debate round
                round_data = await debate_manager.run_debate_round(
                    task=session.task,
                    current_solutions=current_solutions,
                    participants=session.participants,
                    round_number=round_num,
                )

                session.rounds.append(round_data)

                # Update solutions with refinements
                if round_data.refinements:
                    current_solutions = round_data.refinements

                # Check for consensus
                if round_data.consensus_score >= session.config.consensus_threshold:
                    logger.info(f"Session {session_id}: Consensus reached at round {round_num}")
                    session.consensus_reached = True
                    session.status = SwarmStatus.CONSENSUS
                    await self._emit_progress(session_id, "consensus", {
                        "round": round_num,
                        "score": round_data.consensus_score,
                    })
                    break

                await self._emit_progress(session_id, "round_complete", {
                    "round": round_num,
                    "consensus_score": round_data.consensus_score,
                })

            # Select best solution
            session.status = SwarmStatus.COMPLETED
            best_id, best_solution = self._select_best_solution(
                current_solutions, session.participants
            )
            session.best_solution = best_solution
            session.best_participant_id = best_id
            session.final_output = best_solution
            session.completed_at = datetime.now()

            duration = time.time() - start_time
            await self._emit_progress(session_id, "completed", {
                "best_participant": best_id,
                "duration": duration,
                "consensus_reached": session.consensus_reached,
            })

            logger.info(f"Session {session_id} completed in {duration:.1f}s")

            return SwarmResult(
                session_id=session_id,
                output=best_solution,
                best_participant_id=best_id,
                best_participant_name=self._get_participant_name(session, best_id),
                rounds_completed=len(session.rounds),
                consensus_reached=session.consensus_reached,
                final_consensus_score=session.rounds[-1].consensus_score if session.rounds else 0.0,
                all_solutions=current_solutions,
                duration_seconds=duration,
                tokens_used=session.total_tokens_used,
            )

        except Exception as e:
            session.status = SwarmStatus.ERROR
            session.error_message = str(e)
            logger.error(f"Session {session_id} failed: {e}")
            await self._emit_progress(session_id, "error", {"message": str(e)})
            raise

        finally:
            # Cleanup
            await debate_manager.close()

    def _select_best_solution(
        self,
        solutions: dict[str, str],
        participants: list[SwarmParticipant],
    ) -> tuple[str, str]:
        """Select the best solution based on multiple criteria."""
        if not solutions:
            raise ValueError("No solutions to select from")

        scores: dict[str, float] = {}

        for pid, solution in solutions.items():
            score = 0.0

            # Length appropriateness (not too short, not too long)
            length = len(solution)
            if 100 < length < 10000:
                score += 2.0
            elif 50 < length < 20000:
                score += 1.0

            # Has code structure indicators
            code_indicators = ["def ", "class ", "import ", "function ", "const ", "return"]
            for indicator in code_indicators:
                if indicator in solution:
                    score += 0.5

            # Has comments/documentation
            if "#" in solution or "//" in solution or "/**" in solution:
                score += 1.0

            # Penalize error messages
            if solution.startswith("Error"):
                score -= 5.0

            scores[pid] = score

        best_id = max(scores, key=scores.get)
        return best_id, solutions[best_id]

    def _get_participant_name(self, session: SwarmSession, participant_id: str) -> str:
        """Get participant name by ID."""
        for p in session.participants:
            if p.id == participant_id:
                return p.name
        return "Unknown"

    async def stop_session(self, session_id: str):
        """Stop a running session."""
        session = self.active_sessions.get(session_id)
        if session:
            session.status = SwarmStatus.STOPPED
            await self._emit_progress(session_id, "stopped", {})
            logger.info(f"Session {session_id} stopped")

    async def get_session_status(self, session_id: str) -> Optional[dict]:
        """Get current session status."""
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        return {
            "id": session.id,
            "status": session.status,
            "current_round": session.current_round,
            "max_rounds": session.config.max_rounds,
            "consensus_reached": session.consensus_reached,
            "num_participants": len(session.participants),
            "rounds_completed": len(session.rounds),
            "created_at": session.created_at.isoformat(),
        }

    async def cleanup_session(self, session_id: str):
        """Cleanup a completed session."""
        if session_id in self.debate_managers:
            await self.debate_managers[session_id].close()
            del self.debate_managers[session_id]

        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        if session_id in self._progress_callbacks:
            del self._progress_callbacks[session_id]

        logger.info(f"Session {session_id} cleaned up")


# Singleton instance
swarm_orchestrator = SwarmOrchestrator()
