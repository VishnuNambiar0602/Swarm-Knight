"""Tests for Database module."""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.database import Database


@pytest.fixture
def test_db(tmp_path):
    db = Database(db_path=tmp_path / "test_swarm")
    yield db
    db.close()


class TestDatabase:
    def test_init_creates_db(self, test_db):
        assert test_db.db_file.exists()

    def test_save_and_get_session(self, test_db):
        session = {
            "id": "sess-1",
            "task": "Build API",
            "status": "planning",
            "config": None,
            "agents": None,
            "rounds": None,
            "final_output": None,
            "judgment": None,
            "winner_id": None,
            "winner_name": None,
            "consensus_score": 0,
            "duration_ms": 0,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None,
        }
        test_db.save_session(session)
        result = test_db.get_session("sess-1")
        assert result is not None
        assert result["task"] == "Build API"

    def test_list_sessions(self, test_db):
        for i in range(3):
            test_db.save_session({
                "id": f"sess-{i}",
                "task": f"Task {i}",
                "status": "planning",
                "config": None,
                "agents": None,
                "rounds": None,
                "final_output": None,
                "judgment": None,
                "winner_id": None,
                "winner_name": None,
                "consensus_score": 0,
                "duration_ms": 0,
                "created_at": datetime.now().isoformat(),
                "completed_at": None,
                "error": None,
            })
        sessions = test_db.list_sessions()
        assert len(sessions) == 3

    def test_delete_session(self, test_db):
        test_db.save_session({
            "id": "sess-del",
            "task": "Delete me",
            "status": "planning",
            "config": None,
            "agents": None,
            "rounds": None,
            "final_output": None,
            "judgment": None,
            "winner_id": None,
            "winner_name": None,
            "consensus_score": 0,
            "duration_ms": 0,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None,
        })
        test_db.delete_session("sess-del")
        assert test_db.get_session("sess-del") is None

    def test_save_and_search_memory(self, test_db):
        test_db.save_memory({
            "id": "mem-1",
            "memory_type": "task_history",
            "content": "Built REST API successfully",
            "task": "Build API",
            "agent_id": "agent-1",
            "session_id": "sess-1",
            "score": 8.5,
            "metadata": {},
            "created_at": datetime.now().isoformat(),
        })
        results = test_db.search_memories("REST API")
        assert len(results) >= 1
        assert results[0]["content"] == "Built REST API successfully"

    def test_save_and_get_reputation(self, test_db):
        test_db.save_reputation({
            "agent_id": "agent-1",
            "agent_name": "TestAgent",
            "model": "test/model",
            "total_tasks": 5,
            "wins": 3,
            "total_score": 28.5,
            "avg_score": 5.7,
            "consensus_contributions": 2,
            "last_used": datetime.now().isoformat(),
        })
        rep = test_db.get_reputation("agent-1")
        assert rep is not None
        assert rep["total_tasks"] == 5

    def test_save_and_validate_api_key(self, test_db):
        test_db.save_api_key("sk-test123", name="test-key")
        assert test_db.validate_api_key("sk-test123") is True
        assert test_db.validate_api_key("sk-invalid") is False

    def test_record_metric(self, test_db):
        test_db.record_metric("sess-1", "api_call", {"latency_ms": 150})
        metrics = test_db.get_metrics("sess-1")
        assert len(metrics) == 1
        assert metrics[0]["event_type"] == "api_call"

    def test_get_stats(self, test_db):
        stats = test_db.get_stats()
        assert "sessions" in stats
        assert "memories" in stats
        assert "db_size_mb" in stats
