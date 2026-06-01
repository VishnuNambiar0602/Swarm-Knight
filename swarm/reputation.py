"""Reputation System - Track agent performance and reliability."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import AgentReputation

logger = logging.getLogger("swarm.reputation")


class Reputation:
    """Track and query agent reputation scores."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path.home() / ".swarm-knight" / "reputation"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._reputations: dict[str, AgentReputation] = {}
        self._load()

    def _load(self):
        for file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text())
                rep = AgentReputation(**data)
                self._reputations[rep.agent_id] = rep
            except Exception as e:
                logger.warning(f"Failed to load reputation: {e}")

    def record(
        self,
        agent_id: str,
        agent_name: str,
        model: str,
        score: float,
        is_winner: bool = False,
        contributed_to_consensus: bool = False,
    ):
        if agent_id not in self._reputations:
            self._reputations[agent_id] = AgentReputation(
                agent_id=agent_id, agent_name=agent_name, model=model
            )
        rep = self._reputations[agent_id]
        rep.total_tasks += 1
        rep.total_score += score
        rep.avg_score = rep.total_score / rep.total_tasks
        if is_winner:
            rep.wins += 1
        if contributed_to_consensus:
            rep.consensus_contributions += 1
        rep.last_used = datetime.now()
        self._persist(rep)

    def score(self, agent_id: str) -> float:
        rep = self._reputations.get(agent_id)
        if not rep or rep.total_tasks == 0:
            return 0.5
        win_rate = rep.wins / rep.total_tasks
        consensus_rate = rep.consensus_contributions / rep.total_tasks
        return win_rate * 0.5 + rep.avg_score / 10 * 0.3 + consensus_rate * 0.2

    def get(self, agent_id: str) -> Optional[AgentReputation]:
        return self._reputations.get(agent_id)

    def top(self, n: int = 5) -> list[AgentReputation]:
        return sorted(
            self._reputations.values(),
            key=lambda r: self.score(r.agent_id),
            reverse=True,
        )[:n]

    def get_stats(self) -> dict:
        if not self._reputations:
            return {"total_agents": 0, "avg_score": 0}
        scores = [self.score(r.agent_id) for r in self._reputations.values()]
        return {
            "total_agents": len(self._reputations),
            "avg_score": sum(scores) / len(scores),
            "top_agents": [r.agent_name for r in self.top(3)],
        }

    def _persist(self, rep: AgentReputation):
        file = self.storage_dir / f"{rep.agent_id}.json"
        file.write_text(json.dumps(rep.model_dump(), default=str, indent=2))


reputation = Reputation()
