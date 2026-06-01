"""Memory System - Long-term, Project, Agent-specific memory with semantic retrieval."""

from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .models import MemoryEntry, MemoryType

logger = logging.getLogger(__name__)


class MemoryIndex:
    """Simple in-memory vector index for semantic search."""

    def __init__(self):
        self._entries: dict[str, MemoryEntry] = {}
        self._embeddings: dict[str, list[float]] = {}

    def add(self, entry: MemoryEntry) -> None:
        self._entries[entry.id] = entry
        if entry.embedding:
            self._embeddings[entry.id] = entry.embedding

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[MemoryEntry, float]]:
        if not self._embeddings:
            return []

        scores: list[tuple[str, float]] = []
        for entry_id, emb in self._embeddings.items():
            score = self._cosine_similarity(query_embedding, emb)
            scores.append((entry_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for entry_id, score in scores[:top_k]:
            entry = self._entries.get(entry_id)
            if entry:
                entry.relevance_score = score
                entry.access_count += 1
                entry.accessed_at = datetime.now()
                results.append((entry, score))

        return results

    def get_by_type(self, memory_type: MemoryType) -> list[MemoryEntry]:
        return [e for e in self._entries.values() if e.memory_type == memory_type]

    def get_by_agent(self, agent_id: str) -> list[MemoryEntry]:
        return [e for e in self._entries.values() if e.agent_id == agent_id]

    def get_by_project(self, project_id: str) -> list[MemoryEntry]:
        return [e for e in self._entries.values() if e.project_id == project_id]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class MemorySystem:
    """Persistent memory system with semantic retrieval."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path.home() / ".swarm-knight" / "memory"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.long_term = MemoryIndex()
        self.project_memory: dict[str, MemoryIndex] = {}
        self.agent_memory: dict[str, MemoryIndex] = {}
        self.task_history: list[MemoryEntry] = []

        self._load_all()

    def _load_all(self):
        """Load all memories from disk."""
        for file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text())
                entry = MemoryEntry(**data)
                self._add_to_index(entry)
            except Exception as e:
                logger.warning(f"Failed to load memory {file}: {e}")

    def _add_to_index(self, entry: MemoryEntry):
        if entry.memory_type == MemoryType.LONG_TERM:
            self.long_term.add(entry)
        elif entry.memory_type == MemoryType.PROJECT:
            pid = entry.project_id or "default"
            if pid not in self.project_memory:
                self.project_memory[pid] = MemoryIndex()
            self.project_memory[pid].add(entry)
        elif entry.memory_type == MemoryType.AGENT_SPECIFIC:
            aid = entry.agent_id or "default"
            if aid not in self.agent_memory:
                self.agent_memory[aid] = MemoryIndex()
            self.agent_memory[aid].add(entry)
        elif entry.memory_type == MemoryType.TASK_HISTORY:
            self.task_history.append(entry)

    def store(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[dict] = None,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        embedding: Optional[list[float]] = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            agent_id=agent_id,
            project_id=project_id,
            embedding=embedding,
        )
        self._add_to_index(entry)
        self._persist(entry)
        return entry

    def recall(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        results = []

        if memory_type == MemoryType.LONG_TERM or memory_type is None:
            results.extend(self._search_index(self.long_term, query, top_k))

        if (memory_type == MemoryType.PROJECT or memory_type is None) and project_id:
            idx = self.project_memory.get(project_id)
            if idx:
                results.extend(self._search_index(idx, query, top_k))

        if (memory_type == MemoryType.AGENT_SPECIFIC or memory_type is None) and agent_id:
            idx = self.agent_memory.get(agent_id)
            if idx:
                results.extend(self._search_index(idx, query, top_k))

        if memory_type == MemoryType.TASK_HISTORY or memory_type is None:
            for entry in self.task_history[-50:]:
                if query.lower() in entry.content.lower():
                    results.append(entry)

        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results[:top_k]

    def _search_index(self, index: MemoryIndex, query: str, top_k: int) -> list[MemoryEntry]:
        query_emb = self._simple_embedding(query)
        pairs = index.search(query_emb, top_k)
        return [entry for entry, _ in pairs]

    def store_task_result(
        self,
        task: str,
        result: str,
        agent_id: str,
        score: float,
        project_id: Optional[str] = None,
    ):
        content = f"Task: {task}\nResult: {result[:500]}\nScore: {score}"
        self.store(
            content=content,
            memory_type=MemoryType.TASK_HISTORY,
            metadata={"score": score, "task": task},
            agent_id=agent_id,
            project_id=project_id,
        )

    def get_similar_tasks(self, task: str, top_k: int = 3) -> list[MemoryEntry]:
        return self.recall(task, memory_type=MemoryType.TASK_HISTORY, top_k=top_k)

    def get_agent_history(self, agent_id: str) -> list[MemoryEntry]:
        return self.agent_memory.get(agent_id, MemoryIndex()).get_by_type(
            MemoryType.AGENT_SPECIFIC
        ) if agent_id in self.agent_memory else []

    def _simple_embedding(self, text: str) -> list[float]:
        h = hashlib.sha256(text.lower().encode()).digest()
        return [b / 255.0 for b in h[:32]]

    def _persist(self, entry: MemoryEntry):
        file = self.storage_dir / f"{entry.id}.json"
        file.write_text(json.dumps(entry.model_dump(), default=str, indent=2))

    def get_stats(self) -> dict:
        return {
            "long_term": len(self.long_term._entries),
            "projects": {k: len(v._entries) for k, v in self.project_memory.items()},
            "agents": {k: len(v._entries) for k, v in self.agent_memory.items()},
            "task_history": len(self.task_history),
        }


memory_system = MemorySystem()
