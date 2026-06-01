"""Reputation System - Agent performance tracking backed by SQLite."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from .database import db
from .models import AgentReputation

logger = logging.getLogger("swarm.reputation")


class Reputation:
    """Track agent reputation backed by SQLite."""

    def __init__(self):
        pass

    def record(
        self,
        agent_id: str,
        agent_name: str,
        model: str,
        score: float,
        is_winner: bool = False,
        contributed_to_consensus: bool = False,
    ):
        existing = db.get_reputation(agent_id)
        if existing:
            total = existing["total_tasks"] + 1
            wins = existing["wins"] + (1 if is_winner else 0)
            total_score = existing["total_score"] + score
            consensus = existing["consensus_contributions"] + (1 if contributed_to_consensus else 0)
        else:
            total = 1
            wins = 1 if is_winner else 0
            total_score = score
            consensus = 1 if contributed_to_consensus else 0

        db.save_reputation({
            "agent_id": agent_id,
            "agent_name": agent_name,
            "model": model,
            "total_tasks": total,
            "wins": wins,
            "total_score": total_score,
            "avg_score": total_score / total,
            "consensus_contributions": consensus,
            "last_used": datetime.now().isoformat(),
        })

    def score(self, agent_id: str) -> float:
        rep = db.get_reputation(agent_id)
        if not rep or rep["total_tasks"] == 0:
            return 0.5
        win_rate = rep["wins"] / rep["total_tasks"]
        consensus_rate = rep["consensus_contributions"] / rep["total_tasks"]
        return win_rate * 0.5 + rep["avg_score"] / 10 * 0.3 + consensus_rate * 0.2

    def get(self, agent_id: str) -> Optional[dict]:
        return db.get_reputation(agent_id)

    def top(self, n: int = 5) -> list[dict]:
        reps = db.list_reputations()
        return reps[:n]

    def get_stats(self) -> dict:
        reps = db.list_reputations()
        if not reps:
            return {"total_agents": 0, "avg_score": 0}
        scores = [self.score(r["agent_id"]) for r in reps]
        return {
            "total_agents": len(reps),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "top_agents": [r["agent_name"] for r in self.top(3)],
        }


reputation = Reputation()
