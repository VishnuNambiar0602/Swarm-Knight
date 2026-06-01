"""Consensus Engine - Core architecture replacing the orchestrator."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Callable, Optional

from .cache import cache_system
from .debate import DebateManager
from .dynamic_agents import dynamic_generator
from .memory import memory_system
from .models import (
    DebateRound,
    ParticipantRole,
    SwarmConfig,
    SwarmParticipant,
    SwarmResult,
    SwarmSession,
    SwarmStatus,
    TaskPlan,
)
from .parallel import parallel_engine, TaskResult
from .providers import OpenSourceProvider, create_provider
from .reputation import reputation_system

logger = logging.getLogger(__name__)


class ConsensusEngine:
    """Core engine with consensus-based termination and dynamic agent generation."""

    def __init__(self):
        self.active_sessions: dict[str, SwarmSession] = {}
        self._progress_callbacks: dict[str, Callable] = {}
        self.debate_managers: dict[str, DebateManager] = {}

    async def create_session(
        self,
        task: str,
        participants: Optional[list[SwarmParticipant]] = None,
        config: Optional[SwarmConfig] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> SwarmSession:
        config = config or SwarmConfig()

        if not participants and config.enable_dynamic_agents:
            participants = dynamic_generator.generate_agents(
                task, max_agents=config.max_agents, api_key=api_key
            )
        elif not participants:
            from .models import FREE_MODELS
            participants = self._get_default_participants(api_key)

        if config.enable_reputation:
            scores = {r.agent_id: reputation_system.get_agent_score(r.agent_id) for r in participants}
            participants = dynamic_generator.rank_agents_by_reputation(participants, scores)

        session = SwarmSession(
            task=task,
            participants=participants,
            config=config,
        )

        plan = dynamic_generator.analyze_task(task)
        session.task_type = plan.task_type
        session.required_capabilities = plan.subtasks

        self.active_sessions[session.id] = session
        self.debate_managers[session.id] = DebateManager()

        if config.enable_memory:
            memory_system.store(
                content=f"Created session for: {task}",
                memory_type=MemoryType.TASK_HISTORY,
                metadata={"session_id": session.id, "task_type": plan.task_type},
                project_id=project_id,
            )

        logger.info(f"Session {session.id}: Created with {len(participants)} agents ({plan.task_type})")
        return session

    def _get_default_participants(self, api_key: Optional[str]) -> list[SwarmParticipant]:
        from .models import FREE_MODELS
        return [
            SwarmParticipant(name="Laguna Coder", provider=ProviderType.OPENROUTER, model=FREE_MODELS["coder"]["model"], role=ParticipantRole.GENERATOR, api_key=api_key),
            SwarmParticipant(name="Kimi Assembler", provider=ProviderType.OPENROUTER, model=FREE_MODELS["assembler"]["model"], role=ParticipantRole.GENERATOR, api_key=api_key),
            SwarmParticipant(name="Nemotron Reviewer", provider=ProviderType.OPENROUTER, model=FREE_MODELS["reviewer"]["model"], role=ParticipantRole.CRITIC, api_key=api_key),
        ]

    def set_progress_callback(self, session_id: str, callback: Callable):
        self._progress_callbacks[session_id] = callback

    async def _emit(self, session_id: str, event: str, data: dict):
        cb = self._progress_callbacks.get(session_id)
        if cb:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event, data)
                else:
                    cb(event, data)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    async def run_session(self, session_id: str) -> SwarmResult:
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        debate_manager = self.debate_managers[session_id]
        start_time = time.time()

        try:
            session.status = SwarmStatus.GENERATING
            await self._emit(session_id, "status_change", {"status": session.status})

            plan = dynamic_generator.analyze_task(session.task)
            estimated_rounds = min(plan.estimated_rounds, session.config.max_rounds)

            logger.info(f"Session {session_id}: Starting with {len(session.participants)} agents")
            current_solutions = await self._parallel_generate(
                session.task, session.participants, session
            )

            best_score = 0.0
            no_improvement_count = 0

            for round_num in range(1, estimated_rounds + 1):
                if session.status == SwarmStatus.STOPPED:
                    break

                session.current_round = round_num
                session.status = SwarmStatus.DEBATING
                await self._emit(session_id, "round_start", {"round": round_num})

                round_data = await debate_manager.run_debate_round(
                    task=session.task,
                    current_solutions=current_solutions,
                    participants=session.participants,
                    round_number=round_num,
                )

                session.rounds.append(round_data)

                if round_data.refinements:
                    current_solutions = round_data.refinements

                round_score = round_data.consensus_score
                improvement = round_score - best_score

                if round_score > best_score:
                    best_score = round_score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                await self._emit(session_id, "round_complete", {
                    "round": round_num,
                    "consensus_score": round_score,
                    "improvement": improvement,
                })

                if round_score >= session.config.early_stop_threshold:
                    logger.info(f"Session {session_id}: Early consensus at round {round_num} ({round_score:.2%})")
                    session.consensus_reached = True
                    session.status = SwarmStatus.CONSENSUS
                    await self._emit(session_id, "consensus", {"round": round_num, "score": round_score})
                    break

                if no_improvement_count >= 2:
                    logger.info(f"Session {session_id}: No improvement for 2 rounds, stopping")
                    break

            session.status = SwarmStatus.MERGING
            await self._emit(session_id, "merging", {})

            best_id, best_solution = self._select_best_solution(current_solutions, session.participants)
            merged_output = await self._merge_solutions(current_solutions, session.participants, session)

            session.best_solution = best_solution
            session.best_participant_id = best_id
            session.final_output = merged_output or best_solution
            session.status = SwarmStatus.COMPLETED
            session.completed_at = datetime.now()

            duration = time.time() - start_time
            agent_scores = {}
            for p in session.participants:
                agent_scores[p.name] = reputation_system.get_agent_score(p.id)

            if session.config.enable_memory:
                memory_system.store_task_result(
                    task=session.task,
                    result=session.final_output[:1000],
                    agent_id=best_id,
                    score=best_score,
                )

            if session.config.enable_reputation:
                for p in session.participants:
                    score = 1.0 if p.id == best_id else 0.5
                    reputation_system.record_task(
                        agent_id=p.id,
                        agent_name=p.name,
                        model=p.model,
                        score=score,
                        latency_ms=duration * 1000 / len(session.participants),
                        contributed_to_consensus=session.consensus_reached,
                    )

            await self._emit(session_id, "completed", {
                "best_participant": best_id,
                "duration": duration,
                "consensus_reached": session.consensus_reached,
            })

            return SwarmResult(
                session_id=session_id,
                output=session.final_output or "",
                best_participant_id=best_id,
                best_participant_name=self._get_name(session, best_id),
                rounds_completed=len(session.rounds),
                consensus_reached=session.consensus_reached,
                final_consensus_score=best_score,
                all_solutions=current_solutions,
                duration_seconds=duration,
                tokens_used=session.total_tokens_used,
                agent_scores=agent_scores,
                memory_saved=session.config.enable_memory,
            )

        except Exception as e:
            session.status = SwarmStatus.ERROR
            session.error_message = str(e)
            logger.error(f"Session {session_id} failed: {e}")
            await self._emit(session_id, "error", {"message": str(e)})
            raise
        finally:
            await debate_manager.close()

    async def _parallel_generate(
        self,
        task: str,
        participants: list[SwarmParticipant],
        session: SwarmSession,
    ) -> dict[str, str]:
        async def _gen(provider: OpenSourceParticipant, participant: SwarmParticipant) -> str:
            cached = cache_system.get_response(task, participant.model)
            if cached:
                return cached
            result = await provider.generate_with_retry(
                prompt=task,
                system_prompt=self._get_system_prompt(participant),
            )
            cache_system.set_response(task, participant.model, result)
            return result

        task_results = await parallel_engine.run_parallel(
            _gen, participants, timeout=session.config.timeout_seconds
        )

        solutions = {}
        for tr in task_results:
            if tr.success:
                solutions[tr.agent_id] = tr.result
            else:
                logger.warning(f"Agent {tr.agent_name} failed: {tr.error}")

        if not solutions:
            raise RuntimeError("All agents failed to generate solutions")

        return solutions

    async def _merge_solutions(
        self,
        solutions: dict[str, str],
        participants: list[SwarmParticipant],
        session: SwarmSession,
    ) -> Optional[str]:
        if len(solutions) == 1:
            return next(iter(solutions.values()))

        merger = next(
            (p for p in participants if p.role in (ParticipantRole.GENERATOR, ParticipantRole.ASSEMBLER)),
            participants[0] if participants else None,
        )
        if not merger:
            return None

        solutions_text = "\n\n---\n\n".join(
            f"Solution from {self._get_name(session, pid)}:\n{sol[:2000]}"
            for pid, sol in solutions.items()
        )

        merge_prompt = f"Original task: {session.task}\n\nMultiple solutions were generated:\n\n{solutions_text}\n\nMerge the best parts into one cohesive solution. Output only the final merged code."

        try:
            provider = create_provider(merger)
            result = await provider.generate_with_retry(prompt=merge_prompt)
            await provider.close()
            return result
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return None

    def _select_best_solution(
        self, solutions: dict[str, str], participants: list[SwarmParticipant]
    ) -> tuple[str, str]:
        if not solutions:
            raise ValueError("No solutions to select")

        scores = {}
        for pid, solution in solutions.items():
            score = 0.0
            length = len(solution)
            if 100 < length < 10000:
                score += 2.0
            elif 50 < length < 20000:
                score += 1.0

            code_indicators = ["def ", "class ", "import ", "function ", "const ", "return"]
            for ind in code_indicators:
                if ind in solution:
                    score += 0.5

            if "#" in solution or "//" in solution:
                score += 1.0

            if solution.startswith("Error"):
                score -= 5.0

            rep_score = reputation_system.get_agent_score(pid)
            score += rep_score * 2

            scores[pid] = score

        best_id = max(scores, key=scores.get)
        return best_id, solutions[best_id]

    def _get_system_prompt(self, participant: SwarmParticipant) -> str:
        prompts = {
            ParticipantRole.GENERATOR: "You are an expert developer. Generate high-quality, complete, production-ready code.",
            ParticipantRole.CRITIC: "You are an expert code reviewer. Provide thorough, constructive criticism.",
            ParticipantRole.REFINER: "You are an expert at improving code. Refine and optimize solutions.",
            ParticipantRole.COORDINATOR: "You are a technical lead. Coordinate and synthesize solutions.",
            ParticipantRole.PLANNER: "You are a technical architect. Plan and design system architecture.",
            ParticipantRole.REVIEWER: "You are a QA expert. Review for quality, edge cases, and best practices.",
        }
        return prompts.get(participant.role, prompts[ParticipantRole.GENERATOR])

    def _get_name(self, session: SwarmSession, pid: str) -> str:
        for p in session.participants:
            if p.id == pid:
                return p.name
        return "Unknown"

    async def stop_session(self, session_id: str):
        session = self.active_sessions.get(session_id)
        if session:
            session.status = SwarmStatus.STOPPED
            await self._emit(session_id, "stopped", {})

    async def cleanup_session(self, session_id: str):
        if session_id in self.debate_managers:
            await self.debate_managers[session_id].close()
            del self.debate_managers[session_id]
        self.active_sessions.pop(session_id, None)
        self._progress_callbacks.pop(session_id, None)

    def get_stats(self) -> dict:
        return {
            "active_sessions": len(self.active_sessions),
            "cache": cache_system.get_stats(),
            "reputation": reputation_system.get_stats(),
            "memory": memory_system.get_stats(),
        }


consensus_engine = ConsensusEngine()
