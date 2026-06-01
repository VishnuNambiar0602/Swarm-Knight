"""Memory System - Persistent storage backed by SQLite."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .database import db
from .models import MemoryEntry, MemoryType

logger = logging.getLogger("swarm.memory")


class Memory:
    """Persistent memory with semantic retrieval via SQLite."""

    def __init__(self):
        pass

    def store(
        self,
        memory_type: MemoryType,
        content: str,
        task: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        score: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            id=uuid4().hex,
            memory_type=memory_type,
            content=content,
            task=task,
            agent_id=agent_id,
            session_id=session_id,
            score=score,
            metadata=metadata or {},
            created_at=datetime.now(),
        )
        db.save_memory(entry.model_dump())
        return entry

    def recall(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        type_str = memory_type.value if memory_type else None
        rows = db.search_memories(query, memory_type=type_str, limit=top_k)
        return [MemoryEntry(**r) for r in rows]

    def get_similar_tasks(self, task: str, top_k: int = 3) -> list[MemoryEntry]:
        return self.recall(task, MemoryType.TASK_HISTORY, top_k)

    def get_stats(self) -> dict:
        return db.get_stats()


memory = Memory()
