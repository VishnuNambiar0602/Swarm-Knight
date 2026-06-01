"""Agent Reputation System - Track performance, accuracy, and contribution."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import AgentReputation

logger = logging.getLogger(__name__)


class ReputationSystem:
    """Track and manage agent reputation scores."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path.home() / ".swarm-knight" / "reputation"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.reputations: dict[str, AgentReputation] = {}
        self._load_all()

    def _load_all(self):
        for file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text())
                rep = AgentReputation(**data)
                self.reputations[rep.agent_id] = rep
            except Exception as e:
                logger.warning(f"Failed to load reputation {file}: {e}")

    def get_reputation(self, agent_id: str) -> AgentReputation:
        if agent_id not in self.reputations:
            self.reputations[agent_id] = AgentReputation(
                agent_id=agent_id, agent_name="", model=""
            )
        return self.reputations[agent_id]

    def record_task(
        self,
        agent_id: str,
        agent_name: str,
        model: str,
        score: float,
        latency_ms: float,
        is_hallucination: bool = False,
        contributed_to_consensus: bool = False,
    ):
        rep = self.get_reputation(agent_id)
        rep.agent_name = agent_name
        rep.model = model
        rep.total_tasks += 1
        rep.total_score += score
        rep.avg_score = rep.total_score / rep.total_tasks
        rep.avg_latency_ms = (
            (rep.avg_latency_ms * (rep.total_tasks - 1) + latency_ms) / rep.total_tasks
        )
        if is_hallucination:
            rep.hallucination_count += 1
        if contributed_to_consensus:
            rep.consensus_contributions += 1
        rep.last_used = datetime.now()
        self._persist(rep)

    def record_success(self, agent_id: str, score: float):
        rep = self.get_reputation(agent_id)
        rep.successful_tasks += 1
        rep.total_score += score
        rep.avg_score = rep.total_score / max(rep.total_tasks, 1)
        self._persist(rep)

    def record_failure(self, agent_id: str):
        rep = self.get_reputation(agent_id)
        rep.total_tasks += 1
        self._persist(rep)

    def get_top_agents(self, n: int = 5) -> list[AgentReputation]:
        sorted_reps = sorted(
            self.reputations.values(),
            key=lambda r: r.avg_score * 0.6 + (1 - r.avg_latency_ms / 10000) * 0.2 + (r.consensus_contributions / max(r.total_tasks, 1)) * 0.2,
            reverse=True,
        )
        return sorted_reps[:n]

    def get_recommended_agents(
        self, task_type: str, n: int = 4
    ) -> list[AgentReputation]:
        if not self.reputations:
            return []
        return self.get_top_agents(n)

    def get_agent_score(self, agent_id: str) -> float:
        rep = self.get_reputation(agent_id)
        if rep.total_tasks == 0:
            return 0.5
        accuracy = rep.successful_tasks / rep.total_tasks
        latency_factor = max(0, 1 - rep.avg_latency_ms / 30000)
        consensus_factor = rep.consensus_contributions / max(rep.total_tasks, 1)
        hallucination_penalty = rep.hallucination_count / max(rep.total_tasks, 1) * 0.5
        return max(0.0, min(1.0, accuracy * 0.5 + latency_factor * 0.2 + consensus_factor * 0.3 - hallucination_penalty))

    def should_use_agent(self, agent_id: str, threshold: float = 0.3) -> bool:
        score = self.get_agent_score(agent_id)
        return score >= threshold

    def get_stats(self) -> dict:
        if not self.reputations:
            return {"total_agents": 0, "avg_score": 0, "avg_latency": 0}
        scores = [self.get_agent_score(r.agent_id) for r in self.reputations.values()]
        latencies = [r.avg_latency_ms for r in self.reputations.values()]
        return {
            "total_agents": len(self.reputations),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "top_agents": [r.agent_name for r in self.get_top_agents(3)],
        }

    def _persist(self, rep: AgentReputation):
        file = self.storage_dir / f"{rep.agent_id}.json"
        file.write_text(json.dumps(rep.model_dump(), default=str, indent=2))


reputation_system = ReputationSystem()
