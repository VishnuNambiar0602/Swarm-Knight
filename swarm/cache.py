"""Performance Cache - Response caching, embedding cache, tool-result cache."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LRUCache:
    """LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, ts = self._cache[key]
            if time.time() - ts < self.ttl_seconds:
                self._cache.move_to_end(key)
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any):
        if key in self._cache:
            del self._cache[key]
        self._cache[key] = (value, time.time())
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def invalidate(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self):
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)


class CacheSystem:
    """Multi-layer cache for responses, embeddings, and tool results."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path.home() / ".swarm-knight" / "cache"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.response_cache = LRUCache(max_size=500, ttl_seconds=7200)
        self.embedding_cache = LRUCache(max_size=2000, ttl_seconds=86400)
        self.tool_cache = LRUCache(max_size=300, ttl_seconds=1800)
        self.plan_cache = LRUCache(max_size=100, ttl_seconds=3600)

        self._stats = {"hits": 0, "misses": 0}

    def get_response(self, prompt: str, model: str) -> Optional[str]:
        key = self._make_key("response", prompt, model)
        result = self.response_cache.get(key)
        if result:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1
        return result

    def set_response(self, prompt: str, model: str, response: str):
        key = self._make_key("response", prompt, model)
        self.response_cache.set(key, response)

    def get_embedding(self, text: str) -> Optional[list[float]]:
        key = self._make_key("embedding", text)
        return self.embedding_cache.get(key)

    def set_embedding(self, text: str, embedding: list[float]):
        key = self._make_key("embedding", text)
        self.embedding_cache.set(key, embedding)

    def get_tool_result(self, tool_name: str, params: str) -> Optional[str]:
        key = self._make_key("tool", tool_name, params)
        return self.tool_cache.get(key)

    def set_tool_result(self, tool_name: str, params: str, result: str):
        key = self._make_key("tool", tool_name, params)
        self.tool_cache.set(key, result)

    def get_plan(self, task: str) -> Optional[dict]:
        key = self._make_key("plan", task)
        return self.plan_cache.get(key)

    def set_plan(self, task: str, plan: dict):
        key = self._make_key("plan", task)
        self.plan_cache.set(key, plan)

    def should_use_cache(self, task: str, model: str) -> bool:
        key = self._make_key("response", task, model)
        return self.response_cache.get(key) is not None

    def get_stats(self) -> dict:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": f"{hit_rate:.1%}",
            "response_cache_size": self.response_cache.size(),
            "embedding_cache_size": self.embedding_cache.size(),
            "tool_cache_size": self.tool_cache.size(),
        }

    def clear_all(self):
        self.response_cache.clear()
        self.embedding_cache.clear()
        self.tool_cache.clear()
        self.plan_cache.clear()
        self._stats = {"hits": 0, "misses": 0}

    def _make_key(self, prefix: str, *parts: str) -> str:
        content = f"{prefix}:{':'.join(parts)}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]


cache_system = CacheSystem()
