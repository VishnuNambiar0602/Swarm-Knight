"""Core Engine - The Knight Loop.

Planner → Generate Agents → Parallel Execution → Debate → Judge → Memory → Reputation → Self-Optimize
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Optional
from uuid import uuid4

from .cache import cache
from .memory import memory
from .metrics import metrics
from .models import (
    Agent,
    AgentRole,
    Critique,
    Judgment,
    MemoryType,
    Round,
    Session,
    SessionStatus,
    SwarmConfig,
    SwarmResult,
    TaskPlan,
)
from .parallel import ParallelEngine
from .providers import create_provider
from .reputation import reputation
from .retry import retry_queue

logger = logging.getLogger("swarm.knight")


class Knight:
    """The core loop: Planner → Agents → Parallel → Debate → Judge → Memory → Reputation → Self-Optimize."""

    def __init__(self):
        self.parallel = ParallelEngine()
        self._callbacks: dict[str, Callable] = {}

    def on(self, event: str, callback: Callable):
        self._callbacks[event] = callback

    async def _emit(self, event: str, data: dict):
        cb = self._callbacks.get(event)
        if cb:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event, data)
                else:
                    cb(event, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    # ── Step 1: Plan ───────────────────────────────────────────────────

    async def plan(self, task: str, config: SwarmConfig) -> TaskPlan:
        """Analyze the task and create a plan."""
        logger.info(f"[PLAN] Analyzing task: {task[:80]}...")
        await self._emit("plan_start", {"task": task})

        task_lower = task.lower()
        scores = {
            "code": sum(1 for kw in ["code", "implement", "build", "create", "function", "class", "api"] if kw in task_lower),
            "design": sum(1 for kw in ["design", "layout", "ui", "style", "css", "visual"] if kw in task_lower),
            "review": sum(1 for kw in ["review", "audit", "fix", "debug", "test"] if kw in task_lower),
            "plan": sum(1 for kw in ["plan", "architect", "structure", "schema"] if kw in task_lower),
            "data": sum(1 for kw in ["data", "parse", "extract", "transform", "csv", "json"] if kw in task_lower),
        }
        task_type = max(scores, key=scores.get) if any(scores.values()) else "code"
        complexity = min(1.0, len(task.split()) / 50 + 0.3)

        roles_map = {
            "code": [AgentRole.GENERATOR, AgentRole.GENERATOR, AgentRole.CRITIC, AgentRole.JUDGE],
            "design": [AgentRole.GENERATOR, AgentRole.GENERATOR, AgentRole.CRITIC, AgentRole.JUDGE],
            "review": [AgentRole.CRITIC, AgentRole.CRITIC, AgentRole.JUDGE, AgentRole.GENERATOR],
            "plan": [AgentRole.PLANNER, AgentRole.GENERATOR, AgentRole.CRITIC, AgentRole.JUDGE],
            "data": [AgentRole.GENERATOR, AgentRole.GENERATOR, AgentRole.REVIEWER, AgentRole.JUDGE],
        }

        plan = TaskPlan(
            task_type=task_type,
            complexity=complexity,
            required_roles=roles_map.get(task_type, roles_map["code"]),
            estimated_rounds=3 if complexity < 0.5 else 5,
            subtasks=self._decompose(task, task_type),
        )

        await self._emit("plan_complete", {"plan": plan.model_dump()})
        logger.info(f"[PLAN] Type: {task_type}, Complexity: {complexity:.0%}, Rounds: {plan.estimated_rounds}")
        return plan

    def _decompose(self, task: str, task_type: str) -> list[str]:
        decompositions = {
            "code": ["Implement core logic", "Handle edge cases", "Write tests", "Review code quality"],
            "design": ["Create layout structure", "Define visual hierarchy", "Ensure accessibility"],
            "review": ["Identify issues", "Suggest improvements", "Validate fixes"],
            "plan": ["Analyze requirements", "Design architecture", "Create implementation plan"],
            "data": ["Parse input data", "Validate and transform", "Generate output"],
        }
        return decompositions.get(task_type, decompositions["code"])

    # ── Step 2: Generate Agents ────────────────────────────────────────

    async def spawn_agents(
        self, task: str, plan: TaskPlan, config: SwarmConfig, api_key: Optional[str] = None
    ) -> list[Agent]:
        """Dynamically generate agents based on the plan."""
        logger.info(f"[SPAWN] Generating {len(plan.required_roles)} agents...")
        await self._emit("spawn_start", {"roles": [r.value for r in plan.required_roles]})

        from .models import FREE_MODELS

        agents = []
        used_models = set()

        for i, role in enumerate(plan.required_roles):
            best_model = self._pick_model(role, plan, used_models, api_key)
            if best_model:
                agents.append(best_model)
                used_models.add(best_model.model)

        if config.enable_reputation:
            agents = self._rank_by_reputation(agents)

        await self._emit("spawn_complete", {"agents": [a.name for a in agents]})
        logger.info(f"[SPAWN] Agents: {[a.name for a in agents]}")
        return agents

    def _pick_model(
        self, role: AgentRole, plan: TaskPlan, used: set, api_key: Optional[str]
    ) -> Optional[Agent]:
        from .models import FREE_MODELS

        for key, info in FREE_MODELS.items():
            if info["model"] in used:
                continue
            if role.value in info["roles"] or "generator" in info["roles"]:
                return Agent(
                    name=info["name"],
                    provider="openrouter",
                    model=info["model"],
                    role=role,
                    api_key=api_key,
                    context_window=info["ctx"],
                    auto_generated=True,
                    task_description=plan.subtasks[0] if plan.subtasks else None,
                )
        return None

    def _rank_by_reputation(self, agents: list[Agent]) -> list[Agent]:
        return sorted(
            agents,
            key=lambda a: reputation.score(a.id),
            reverse=True,
        )

    # ── Step 3: Parallel Execution ─────────────────────────────────────

    async def generate_solutions(
        self, task: str, agents: list[Agent], config: SwarmConfig
    ) -> dict[str, str]:
        """All agents generate solutions in parallel."""
        logger.info(f"[GENERATE] {len(agents)} agents generating solutions...")
        await self._emit("generate_start", {"agent_count": len(agents)})

        results: dict[str, str] = {}

        async def _gen_one(agent: Agent) -> tuple[str, str, float]:
            start = time.time()
            try:
                if config.enable_caching:
                    cached = cache.get("solution", task, agent.model)
                    if cached:
                        return agent.id, cached, 0.0

                provider = create_provider(agent)
                system = self._system_prompt(agent.role)
                result = await provider.generate_with_retry(
                    prompt=task, system_prompt=system
                )
                latency = (time.time() - start) * 1000

                if config.enable_caching:
                    cache.set("solution", task, agent.model, result)

                await provider.close()
                return agent.id, result, latency
            except Exception as e:
                latency = (time.time() - start) * 1000
                logger.error(f"[GENERATE] {agent.name} failed: {e}")
                retry_queue.add(agent.id, task, agent)
                return agent.id, "", latency

        tasks = [_gen_one(a) for a in agents]
        completed = await asyncio.gather(*tasks, return_exceptions=False)

        for agent_id, result, latency in completed:
            if result:
                results[agent_id] = result
                metrics.record_api_call(latency_ms=latency)
            else:
                metrics.record_error()

        await self._emit("generate_complete", {"solutions": len(results)})
        logger.info(f"[GENERATE] {len(results)}/{len(agents)} solutions generated")
        return results

    def _system_prompt(self, role: AgentRole) -> str:
        prompts = {
            AgentRole.PLANNER: "You are a technical architect. Design system architecture and plan implementation.",
            AgentRole.GENERATOR: "You are an expert developer. Generate complete, production-ready code.",
            AgentRole.CRITIC: "You are an expert reviewer. Provide thorough, constructive criticism with specific suggestions.",
            AgentRole.JUDGE: "You are a senior technical judge. Evaluate solutions objectively and select the best one.",
            AgentRole.REFINER: "You are an expert at improving code. Refine and optimize solutions.",
            AgentRole.REVIEWER: "You are a QA expert. Review for quality, edge cases, and best practices.",
        }
        return prompts.get(role, prompts[AgentRole.GENERATOR])

    # ── Step 4: Debate ─────────────────────────────────────────────────

    async def debate(
        self, task: str, solutions: dict[str, str], agents: list[Agent], round_num: int
    ) -> list[Critique]:
        """Each agent critiques all other solutions."""
        logger.info(f"[DEBATE] Round {round_num}: Cross-critique phase...")
        await self._emit("debate_start", {"round": round_num, "solutions": len(solutions)})

        critiques: list[Critique] = []
        agent_map = {a.id: a for a in agents}
        critics = [a for a in agents if a.role in (AgentRole.CRITIC, AgentRole.JUDGE, AgentRole.REVIEWER)]
        if not critics:
            critics = agents[:2]

        async def _critique_one(critic: Agent, target_id: str, target_solution: str) -> Critique:
            try:
                provider = create_provider(critic)
                prompt = f"""Task: {task}

Solution to critique (by {agent_map.get(target_id, Agent(name='Unknown', provider='openrouter', model='', role=AgentRole.GENERATOR)).name}):

{target_solution[:3000]}

Provide a detailed critique. Rate 0-10. List strengths, weaknesses, suggestions."""
                result = await provider.generate_with_retry(
                    prompt=prompt,
                    system_prompt="You are an expert code reviewer. Be thorough and constructive.",
                )
                await provider.close()
                score = self._extract_score(result)
                return Critique(
                    critic_id=critic.id,
                    target_id=target_id,
                    content=result,
                    score=score,
                    strengths=self._extract_list(result, "strength"),
                    weaknesses=self._extract_list(result, "weakness"),
                    suggestions=self._extract_list(result, "suggest"),
                )
            except Exception as e:
                logger.error(f"[DEBATE] Critique failed: {e}")
                return Critique(critic_id=critic.id, target_id=target_id, content=str(e), score=0.0)

        tasks = []
        for critic in critics:
            for target_id, solution in solutions.items():
                if target_id != critic.id:
                    tasks.append(_critique_one(critic, target_id, solution))

        results = await asyncio.gather(*tasks, return_exceptions=False)
        critiques = [r for r in results if isinstance(r, Critique)]

        await self._emit("debate_complete", {"critiques": len(critiques)})
        logger.info(f"[DEBATE] {len(critiques)} critiques generated")
        return critiques

    def _extract_score(self, text: str) -> float:
        import re
        match = re.search(r"(?:score|rating)[:\s]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if match:
            return min(10.0, max(0.0, float(match.group(1))))
        return 5.0

    def _extract_list(self, text: str, keyword: str) -> list[str]:
        import re
        pattern = rf"(?:{keyword}s?|{keyword}ed)[:\s]*\n((?:[-*]\s*.+\n?)+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return [line.strip().lstrip("-* ") for line in match.group(1).strip().split("\n") if line.strip()]
        return []

    # ── Step 5: Judge ──────────────────────────────────────────────────

    async def judge(
        self, task: str, solutions: dict[str, str], critiques: list[Critique], agents: list[Agent]
    ) -> Judgment:
        """Judge selects the best solution and merges if needed."""
        logger.info("[JUDGE] Evaluating all solutions...")
        await self._emit("judge_start", {})

        agent_map = {a.id: a for a in agents}
        judge = next((a for a in agents if a.role == AgentRole.JUDGE), agents[0])

        solutions_text = "\n\n---\n\n".join(
            f"=== {agent_map.get(pid, Agent(name='Unknown', provider='openrouter', model='', role=AgentRole.GENERATOR)).name} ===\n{sol[:2000]}"
            for pid, sol in solutions.items()
        )

        critiques_text = "\n\n".join(
            f"Critic on {agent_map.get(c.target_id, Agent(name='Unknown', provider='openrouter', model='', role=AgentRole.GENERATOR)).name} (Score: {c.score}/10):\n{c.content[:1000]}"
            for c in critiques[:10]
        )

        judge_prompt = f"""Task: {task}

=== SOLUTIONS ===
{solutions_text}

=== CRITIQUES ===
{critiques_text}

Evaluate all solutions. Choose the BEST one. Provide:
1. Winner ID (the agent ID)
2. Score for each agent (0-10)
3. Your reasoning
4. A merged solution combining the best parts

Output format:
WINNER: <agent_id>
SCORES: <id>:<score>, <id>:<score>, ...
MERGED:
<merged code>"""

        try:
            provider = create_provider(judge)
            result = await provider.generate_with_retry(
                prompt=judge_prompt,
                system_prompt="You are a senior technical judge. Be objective and thorough.",
            )
            await provider.close()

            winner_id, scores, merged = self._parse_judgment(result, solutions)

            return Judgment(
                winner_id=winner_id,
                winner_name=agent_map.get(winner_id, Agent(name="Unknown", provider="openrouter", model="", role=AgentRole.GENERATOR)).name,
                scores=scores,
                reasoning=result,
                merged_output=merged,
                consensus_score=self._calc_consensus(scores),
            )
        except Exception as e:
            logger.error(f"[JUDGE] Failed: {e}")
            best_id = max(solutions, key=lambda x: len(solutions[x]))
            return Judgment(
                winner_id=best_id,
                winner_name=agent_map.get(best_id, Agent(name="Unknown", provider="openrouter", model="", role=AgentRole.GENERATOR)).name,
                scores={k: 5.0 for k in solutions},
                reasoning=f"Fallback: {str(e)}",
                consensus_score=0.5,
            )

    def _parse_judgment(self, text: str, solutions: dict[str, str]) -> tuple[str, dict[str, float], Optional[str]]:
        import re

        winner_match = re.search(r"WINNER:\s*(\w+)", text)
        winner_id = winner_match.group(1) if winner_match else next(iter(solutions))

        scores = {}
        scores_match = re.findall(r"(\w+):\s*(\d+(?:\.\d+)?)", text)
        for aid, score in scores_match:
            if aid in solutions:
                scores[aid] = min(10.0, max(0.0, float(score)))
        for aid in solutions:
            if aid not in scores:
                scores[aid] = 5.0

        merged = None
        merged_match = re.search(r"MERGED:\s*\n([\s\S]+)$", text)
        if merged_match:
            merged = merged_match.group(1).strip()

        return winner_id, scores, merged

    def _calc_consensus(self, scores: dict[str, float]) -> float:
        if not scores:
            return 0.0
        values = list(scores.values())
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        return max(0.0, min(1.0, 1.0 - variance / 25.0))

    # ── Step 6: Memory Update ──────────────────────────────────────────

    async def update_memory(
        self, task: str, session: Session, judgment: Judgment
    ):
        """Store task, solution, and lessons in memory."""
        if not session.config.enable_memory:
            return

        logger.info("[MEMORY] Updating memory...")
        await self._emit("memory_start", {})

        memory.store(
            MemoryType.TASK_HISTORY,
            content=task,
            task=task,
            session_id=session.id,
            score=judgment.consensus_score,
            metadata={"winner": judgment.winner_id, "rounds": len(session.rounds)},
        )

        if judgment.merged_output:
            memory.store(
                MemoryType.SOLUTION,
                content=judgment.merged_output,
                task=task,
                agent_id=judgment.winner_id,
                session_id=session.id,
                score=10.0,
            )

        for critique in session.rounds[-1].critiques if session.rounds else []:
            memory.store(
                MemoryType.CRITIQUE,
                content=critique.content,
                task=task,
                agent_id=critique.critic_id,
                session_id=session.id,
                score=critique.score,
            )

        if judgment.consensus_score < 0.5:
            memory.store(
                MemoryType.LESSON,
                content=f"Low consensus ({judgment.consensus_score:.0%}) for: {task}. Consider different approach.",
                task=task,
                session_id=session.id,
            )

        await self._emit("memory_complete", {})
        logger.info("[MEMORY] Memory updated")

    # ── Step 7: Reputation Update ──────────────────────────────────────

    async def update_reputation(self, session: Session, judgment: Judgment):
        """Update agent reputation scores."""
        if not session.config.enable_reputation:
            return

        logger.info("[REPUTATION] Updating reputation...")
        await self._emit("reputation_start", {})

        agent_map = {a.id: a for a in session.agents}
        for agent_id, score in judgment.scores.items():
            agent = agent_map.get(agent_id)
            if agent:
                is_winner = agent_id == judgment.winner_id
                reputation.record(
                    agent_id=agent_id,
                    agent_name=agent.name,
                    model=agent.model,
                    score=score,
                    is_winner=is_winner,
                    contributed_to_consensus=score >= 7.0,
                )

        await self._emit("reputation_complete", {})
        logger.info("[REPUTATION] Reputation updated")

    # ── Step 8: Self-Optimization ──────────────────────────────────────

    async def self_optimize(self, session: Session, judgment: Judgment):
        """Learn from this session to improve future runs."""
        if not session.config.enable_self_optimization:
            return

        logger.info("[OPTIMIZE] Running self-optimization...")
        await self._emit("optimize_start", {})

        stats = reputation.get_stats()
        avg_score = stats.get("avg_score", 0.5)

        if judgment.consensus_score < 0.5:
            logger.info("[OPTIMIZE] Low consensus - will suggest more diverse agents next time")
            memory.store(
                MemoryType.LESSON,
                content=f"Task '{session.task[:50]}' had low consensus. Need more diverse agent selection.",
                task=session.task,
                session_id=session.id,
            )

        if len(session.rounds) >= session.config.max_rounds:
            logger.info("[OPTIMIZE] Hit max rounds - will suggest increasing max_rounds")
            memory.store(
                MemoryType.LESSON,
                content=f"Task needed more rounds. Consider increasing max_rounds for similar tasks.",
                task=session.task,
                session_id=session.id,
            )

        await self._emit("optimize_complete", {"stats": stats})
        logger.info("[OPTIMIZE] Self-optimization complete")

    # ── Main Loop ──────────────────────────────────────────────────────

    async def run(
        self,
        task: str,
        config: Optional[SwarmConfig] = None,
        api_key: Optional[str] = None,
        agents: Optional[list[Agent]] = None,
    ) -> SwarmResult:
        """Execute the full Knight loop."""
        config = config or SwarmConfig()
        session = Session(task=task, config=config)
        start_time = time.time()

        try:
            # 1. Plan
            session.status = SessionStatus.PLANNING
            plan = await self.plan(task, config)

            # 2. Spawn Agents
            session.status = SessionStatus.SPAWNING
            if not agents:
                session.agents = await self.spawn_agents(task, plan, config, api_key)
            else:
                session.agents = agents

            best_score = 0.0
            no_improvement = 0

            for round_num in range(1, plan.estimated_rounds + 1):
                if session.status == SessionStatus.STOPPED:
                    break

                session.current_round = round_num
                round_data = Round(number=round_num)

                # 3. Parallel Execution
                session.status = SessionStatus.GENERATING
                solutions = await self.generate_solutions(task, session.agents, config)
                round_data.solutions = solutions

                # 4. Debate
                session.status = SessionStatus.DEBATING
                critiques = await self.debate(task, solutions, session.agents, round_num)
                round_data.critiques = critiques

                # 5. Judge
                session.status = SessionStatus.JUDGING
                judgment = await self.judge(task, solutions, critiques, session.agents)
                round_data.judgment = judgment
                round_data.consensus_score = judgment.consensus_score
                round_data.completed_at = datetime.now()

                session.rounds.append(round_data)

                # Check improvement
                if judgment.consensus_score > best_score:
                    best_score = judgment.consensus_score
                    no_improvement = 0
                else:
                    no_improvement += 1

                await self._emit("round_complete", {
                    "round": round_num,
                    "consensus": judgment.consensus_score,
                    "winner": judgment.winner_name,
                })

                # Early stop
                if judgment.consensus_score >= config.consensus_threshold:
                    logger.info(f"[LOOP] Consensus reached at round {round_num}")
                    break
                if no_improvement >= 2:
                    logger.info(f"[LOOP] No improvement for 2 rounds, stopping")
                    break

            # 6. Memory Update
            session.status = SessionStatus.OPTIMIZING
            session.judgment = judgment
            await self.update_memory(task, session, judgment)

            # 7. Reputation Update
            await self.update_reputation(session, judgment)

            # 8. Self-Optimize
            await self.self_optimize(session, judgment)

            # Finalize
            session.status = SessionStatus.COMPLETED
            session.completed_at = datetime.now()
            session.final_output = judgment.merged_output or solutions.get(judgment.winner_id, "")

            duration = time.time() - start_time
            metrics.record_session(duration, len(session.rounds))

            return SwarmResult(
                session_id=session.id,
                output=session.final_output,
                winner_id=judgment.winner_id,
                winner_name=judgment.winner_name,
                rounds_completed=len(session.rounds),
                consensus_reached=best_score >= config.consensus_threshold,
                consensus_score=best_score,
                all_solutions=solutions,
                duration_seconds=duration,
                agent_scores=judgment.scores,
                metrics=metrics.get_snapshot(),
            )

        except Exception as e:
            session.status = SessionStatus.FAILED
            session.error = str(e)
            logger.error(f"[LOOP] Failed: {e}")
            metrics.record_error()
            raise

    async def stop(self, session: Session):
        session.status = SessionStatus.STOPPED

    async def close(self):
        await self.parallel.close()


knight = Knight()
