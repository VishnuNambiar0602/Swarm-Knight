"""Database Layer - Local SQLite with async support."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("swarm.db")

DEFAULT_DB_DIR = Path.home() / ".swarm-knight"


class Database:
    """Lightweight SQLite database for local storage."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_dir = db_path or DEFAULT_DB_DIR
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_file = self.db_dir / "swarm.db"
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            # Run migrations
            self._migrate(conn)
            conn.commit()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_file))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    def _migrate(self, conn: sqlite3.Connection):
        current = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if not current:
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO meta (key, value) VALUES ('version', '1')")

        version = conn.execute("SELECT value FROM meta WHERE key='version'").fetchone()
        ver = int(version[0]) if version else 0

        if ver < 1:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    status TEXT DEFAULT 'planning',
                    config TEXT,
                    agents TEXT,
                    rounds TEXT,
                    final_output TEXT,
                    judgment TEXT,
                    winner_id TEXT,
                    winner_name TEXT,
                    consensus_score REAL DEFAULT 0,
                    duration_ms REAL DEFAULT 0,
                    created_at TEXT,
                    completed_at TEXT,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    task TEXT,
                    agent_id TEXT,
                    session_id TEXT,
                    score REAL DEFAULT 0,
                    metadata TEXT,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reputations (
                    agent_id TEXT PRIMARY KEY,
                    agent_name TEXT,
                    model TEXT,
                    total_tasks INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    total_score REAL DEFAULT 0,
                    avg_score REAL DEFAULT 0,
                    consensus_contributions INTEGER DEFAULT 0,
                    last_used TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT,
                    last_used TEXT,
                    active INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    event_type TEXT,
                    data TEXT,
                    timestamp TEXT
                )
            """)
            conn.execute("UPDATE meta SET value='1' WHERE key='version'")
            conn.commit()

    # ── Sessions ───────────────────────────────────────────────────────

    def save_session(self, session_data: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (id, task, status, config, agents, rounds, final_output,
                    judgment, winner_id, winner_name, consensus_score,
                    duration_ms, created_at, completed_at, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_data["id"],
                    session_data["task"],
                    session_data.get("status", "planning"),
                    json.dumps(session_data.get("config")),
                    json.dumps(session_data.get("agents")),
                    json.dumps(session_data.get("rounds")),
                    session_data.get("final_output"),
                    json.dumps(session_data.get("judgment")),
                    session_data.get("winner_id"),
                    session_data.get("winner_name"),
                    session_data.get("consensus_score", 0),
                    session_data.get("duration_ms", 0),
                    session_data.get("created_at", datetime.now().isoformat()),
                    session_data.get("completed_at"),
                    session_data.get("error"),
                ),
            )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_session(self, session_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            conn.commit()

    # ── Memories ───────────────────────────────────────────────────────

    def save_memory(self, memory_data: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO memories
                   (id, memory_type, content, task, agent_id, session_id, score, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory_data["id"],
                    memory_data["memory_type"],
                    memory_data["content"],
                    memory_data.get("task"),
                    memory_data.get("agent_id"),
                    memory_data.get("session_id"),
                    memory_data.get("score", 0),
                    json.dumps(memory_data.get("metadata", {})),
                    memory_data.get("created_at", datetime.now().isoformat()),
                ),
            )
            conn.commit()

    def search_memories(
        self, query: str, memory_type: Optional[str] = None, limit: int = 10
    ) -> list[dict]:
        with self._conn() as conn:
            if memory_type:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE memory_type=? AND content LIKE ? ORDER BY score DESC LIMIT ?",
                    (memory_type, f"%{query}%", limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE content LIKE ? ORDER BY score DESC LIMIT ?",
                    (f"%{query}%", limit),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_task_history(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE memory_type='task_history' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Reputations ────────────────────────────────────────────────────

    def save_reputation(self, rep_data: dict):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO reputations
                   (agent_id, agent_name, model, total_tasks, wins, total_score,
                    avg_score, consensus_contributions, last_used)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rep_data["agent_id"],
                    rep_data.get("agent_name"),
                    rep_data.get("model"),
                    rep_data.get("total_tasks", 0),
                    rep_data.get("wins", 0),
                    rep_data.get("total_score", 0),
                    rep_data.get("avg_score", 0),
                    rep_data.get("consensus_contributions", 0),
                    rep_data.get("last_used", datetime.now().isoformat()),
                ),
            )
            conn.commit()

    def get_reputation(self, agent_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM reputations WHERE agent_id=?", (agent_id,)).fetchone()
            return dict(row) if row else None

    def list_reputations(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM reputations ORDER BY avg_score DESC").fetchall()
            return [dict(r) for r in rows]

    # ── API Keys ───────────────────────────────────────────────────────

    def save_api_key(self, key: str, name: str = "default"):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO api_keys (key, name, created_at, active) VALUES (?, ?, ?, 1)",
                (key, name, datetime.now().isoformat()),
            )
            conn.commit()

    def validate_api_key(self, key: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT active FROM api_keys WHERE key=?", (key,)
            ).fetchone()
            if row and row["active"]:
                conn.execute(
                    "UPDATE api_keys SET last_used=? WHERE key=?",
                    (datetime.now().isoformat(), key),
                )
                conn.commit()
                return True
            return False

    def list_api_keys(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, name, created_at, last_used, active FROM api_keys").fetchall()
            return [dict(r) for r in rows]

    def delete_api_key(self, key: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM api_keys WHERE key=?", (key,))
            conn.commit()

    # ── Metrics ────────────────────────────────────────────────────────

    def record_metric(self, session_id: str, event_type: str, data: dict):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO metrics (session_id, event_type, data, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, event_type, json.dumps(data), datetime.now().isoformat()),
            )
            conn.commit()

    def get_metrics(self, session_id: Optional[str] = None, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM metrics WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM metrics ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    # ── Stats ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._conn() as conn:
            sessions = conn.execute("SELECT COUNT(*) as c FROM sessions").fetchone()["c"]
            memories = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
            agents = conn.execute("SELECT COUNT(*) as c FROM reputations").fetchone()["c"]
            keys = conn.execute("SELECT COUNT(*) as c FROM api_keys WHERE active=1").fetchone()["c"]
            db_size = os.path.getsize(self.db_file) if self.db_file.exists() else 0
            return {
                "sessions": sessions,
                "memories": memories,
                "agents": agents,
                "api_keys": keys,
                "db_size_mb": round(db_size / (1024 * 1024), 2),
            }

    def close(self):
        pass


# Schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


db = Database()
