"""Cache System - Response, embedding, and tool-result caching."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger("swarm.cache")


class LRU:
    """LRU cache with TTL."""

    def __init__(self, max_size: int = 500, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._data: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        if key in self._data:
            val, ts = self._data[key]
            if time.time() - ts < self.ttl:
                self._data.move_to_end(key)
                self.hits += 1
                return val
            del self._data[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any):
        if key in self._data:
            del self._data[key]
        self._data[key] = (value, time.time())
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def size(self) -> int:
        return len(self._data)

    def hit_rate(self) -> str:
        total = self.hits + self.misses
        return f"{self.hits / total:.1%}" if total > 0 else "0%"

    def clear(self):
        self._data.clear()
        self.hits = 0
        self.misses = 0


class Cache:
    """Multi-layer cache."""

    def __init__(self):
        self.responses = LRU(max_size=500, ttl=7200)
        self.embeddings = LRU(max_size=2000, ttl=86400)
        self.plans = LRU(max_size=100, ttl=3600)

    def get(self, layer: str, *parts: str) -> Optional[Any]:
        key = self._key(layer, *parts)
        lru = getattr(self, layer, None)
        if lru:
            return lru.get(key)
        return None

    def set(self, layer: str, value: Any, *parts: str):
        key = self._key(layer, *parts)
        lru = getattr(self, layer, None)
        if lru:
            lru.set(key, value)

    def _key(self, layer: str, *parts: str) -> str:
        content = f"{layer}:{':'.join(str(p) for p in parts)}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get_stats(self) -> dict:
        return {
            "responses": {"size": self.responses.size(), "hit_rate": self.responses.hit_rate()},
            "embeddings": {"size": self.embeddings.size(), "hit_rate": self.embeddings.hit_rate()},
            "plans": {"size": self.plans.size(), "hit_rate": self.plans.hit_rate()},
        }

    def clear(self):
        self.responses.clear()
        self.embeddings.clear()
        self.plans.clear()


cache = Cache()
