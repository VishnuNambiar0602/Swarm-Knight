"""Tests for Swarm-Knight core modules."""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.models import (
    Agent,
    AgentReputation,
    AgentRole,
    Critique,
    Judgment,
    MemoryEntry,
    MemoryType,
    ProviderType,
    Round,
    Session,
    SessionStatus,
    SwarmConfig,
)


class TestModels:
    def test_agent_creation(self):
        agent = Agent(
            id="test-1",
            name="TestAgent",
            provider=ProviderType.OPENROUTER,
            model="test/model",
            role=AgentRole.PLANNER,
            system_prompt="You are a planner",
        )
        assert agent.id == "test-1"
        assert agent.name == "TestAgent"
        assert agent.model == "test/model"
        assert agent.role == AgentRole.PLANNER

    def test_session_creation(self):
        session = Session(
            id="sess-1",
            task="Build a REST API",
            status=SessionStatus.PLANNING,
        )
        assert session.id == "sess-1"
        assert session.task == "Build a REST API"
        assert session.status == SessionStatus.PLANNING

    def test_round_creation(self):
        r = Round(
            number=1,
            solutions={"agent-1": "Generated code"},
            critiques=[],
            refinements={},
            consensus_score=0.0,
            improvement=0.0,
        )
        assert r.number == 1
        assert "agent-1" in r.solutions

    def test_critique_creation(self):
        c = Critique(
            critic_id="agent-2",
            target_id="agent-1",
            content="Looks good",
            score=8.5,
            strengths=["Good structure"],
            weaknesses=["Needs tests"],
            suggestions=["Add unit tests"],
        )
        assert c.score == 8.5
        assert c.critic_id == "agent-2"

    def test_judgment_creation(self):
        j = Judgment(
            winner_id="agent-1",
            winner_name="TestAgent",
            scores={"agent-1": 0.9, "agent-2": 0.75},
            reasoning="Better overall solution",
            consensus_score=0.885,
        )
        assert j.winner_id == "agent-1"
        assert j.consensus_score == 0.885

    def test_memory_entry(self):
        m = MemoryEntry(
            id="mem-1",
            memory_type=MemoryType.TASK_HISTORY,
            content="API task result",
            task="Build API",
            agent_id="agent-1",
            session_id="sess-1",
            score=8.0,
            metadata={"key": "value"},
            created_at=datetime.now(),
        )
        assert m.memory_type == MemoryType.TASK_HISTORY
        assert m.score == 8.0

    def test_agent_reputation(self):
        r = AgentReputation(
            agent_id="agent-1",
            agent_name="TestAgent",
            model="test/model",
            total_tasks=5,
            wins=3,
            total_score=28.5,
            avg_score=5.7,
            consensus_contributions=2,
            last_used=datetime.now(),
        )
        assert r.total_tasks == 5
        assert r.wins == 3

    def test_swarm_config(self):
        c = SwarmConfig(
            max_rounds=3,
            consensus_threshold=0.85,
            auto_stop=True,
            providers=["openrouter"],
        )
        assert c.max_rounds == 3
        assert c.consensus_threshold == 0.85
