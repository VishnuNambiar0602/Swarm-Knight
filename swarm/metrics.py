"""Metrics Collection - Track performance, latency, and usage."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsCollector:
    total_sessions: int = 0
    total_rounds: int = 0
    total_api_calls: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    latencies: list[float] = field(default_factory=list)
    round_latencies: list[float] = field(default_factory=list)
    consensus_scores: list[float] = field(default_factory=list)
    agent_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    agent_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    start_time: float = field(default_factory=time.time)

    def record_session(self, duration_seconds: float, rounds: int):
        self.total_sessions += 1
        self.total_rounds += rounds
        self.latencies.append(duration_seconds * 1000)

    def record_api_call(self, latency_ms: float = 0.0, tokens: int = 0):
        self.total_api_calls += 1
        self.total_tokens += tokens
        if latency_ms > 0:
            self.round_latencies.append(latency_ms)

    def record_round(self, consensus_score: float, duration_ms: float):
        self.consensus_scores.append(consensus_score)
        self.round_latencies.append(duration_ms)

    def record_error(self, error_type: str = "unknown"):
        self.total_errors += 1
        self.errors_by_type[error_type] += 1

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    def record_agent_win(self, agent_id: str):
        self.agent_wins[agent_id] += 1

    def record_agent_call(self, agent_id: str):
        self.agent_calls[agent_id] += 1

    def get_snapshot(self) -> dict[str, Any]:
        uptime = time.time() - self.start_time
        cache_total = self.cache_hits + self.cache_misses
        return {
            "uptime_seconds": uptime,
            "total_sessions": self.total_sessions,
            "total_rounds": self.total_rounds,
            "total_api_calls": self.total_api_calls,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "avg_latency_ms": sum(self.latencies) / len(self.latencies) if self.latencies else 0,
            "avg_round_latency_ms": sum(self.round_latencies) / len(self.round_latencies) if self.round_latencies else 0,
            "avg_consensus": sum(self.consensus_scores) / len(self.consensus_scores) if self.consensus_scores else 0,
            "cache_hit_rate": f"{self.cache_hits / cache_total:.1%}" if cache_total > 0 else "0%",
            "errors_by_type": dict(self.errors_by_type),
            "top_agents": sorted(self.agent_wins.items(), key=lambda x: x[1], reverse=True)[:5],
        }

    def reset(self):
        self.__init__()


metrics = MetricsCollector()
