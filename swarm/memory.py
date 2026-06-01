"""Memory System - Persistent storage for lessons, solutions, and context."""

from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import MemoryEntry, MemoryType

logger = logging.getLogger("swarm.memory")


class Memory:
    """Persistent memory with semantic retrieval."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path.home() / ".swarm-knight" / "memory"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[MemoryEntry] = []
        self._load()

    def _load(self):
        for file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text())
                self._entries.append(MemoryEntry(**data))
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}")

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
            memory_type=memory_type,
            content=content,
            task=task,
            agent_id=agent_id,
            session_id=session_id,
            score=score,
            metadata=metadata or {},
            embedding=self._embed(content),
        )
        self._entries.append(entry)
        self._persist(entry)
        return entry

    def recall(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        query_emb = self._embed(query)
        scored = []
        for entry in self._entries:
            if memory_type and entry.memory_type != memory_type:
                continue
            sim = self._cosine(query_emb, entry.embedding or [])
            scored.append((sim, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def get_similar_tasks(self, task: str, top_k: int = 3) -> list[MemoryEntry]:
        return self.recall(task, MemoryType.TASK_HISTORY, top_k)

    def get_stats(self) -> dict:
        by_type = {}
        for entry in self._entries:
            t = entry.memory_type.value
            by_type[t] = by_type.get(t, 0) + 1
        return {"total": len(self._entries), "by_type": by_type}

    def _embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.lower().encode()).digest()
        return [b / 255.0 for b in h[:32]]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def _persist(self, entry: MemoryEntry):
        file = self.storage_dir / f"{entry.id}.json"
        file.write_text(json.dumps(entry.model_dump(), default=str, indent=2))


memory = Memory()
