"""Retry Queue - Exponential backoff for failed API calls."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("swarm.retry")


@dataclass
class RetryTask:
    task_id: str
    agent_id: str
    prompt: str
    agent: Any  # Agent object
    attempt: int = 0
    max_attempts: int = 3
    created_at: float = field(default_factory=time.time)
    next_retry_at: float = 0.0
    error: Optional[str] = None


class RetryQueue:
    """Queue for retrying failed API calls with exponential backoff."""

    def __init__(self):
        self._queue: deque[RetryTask] = deque()
        self._completed: list[RetryTask] = []
        self._failed: list[RetryTask] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callback = None

    def add(self, agent_id: str, prompt: str, agent: Any, error: str = ""):
        task = RetryTask(
            task_id=f"{agent_id}_{int(time.time())}",
            agent_id=agent_id,
            prompt=prompt,
            agent=agent,
            error=error,
        )
        self._queue.append(task)
        logger.info(f"[RETRY] Added task {task.task_id} for {agent.agent_name if hasattr(agent, 'agent_name') else agent_id}")

    def set_callback(self, callback):
        self._callback = callback

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _process_loop(self):
        while self._running:
            if self._queue:
                task = self._queue[0]
                if time.time() >= task.next_retry_at:
                    self._queue.popleft()
                    await self._retry_task(task)
            await asyncio.sleep(1.0)

    async def _retry_task(self, task: RetryTask):
        task.attempt += 1
        if task.attempt > task.max_attempts:
            logger.warning(f"[RETRY] Task {task.task_id} exceeded max attempts")
            self._failed.append(task)
            return

        backoff = min(2 ** task.attempt, 60)
        task.next_retry_at = time.time() + backoff

        logger.info(f"[RETRY] Retrying {task.task_id} (attempt {task.attempt}/{task.max_attempts}, backoff {backoff}s)")

        try:
            from .providers import create_provider
            provider = create_provider(task.agent)
            result = await provider.generate_with_retry(prompt=task.prompt, max_retries=1)
            await provider.close()

            if self._callback:
                await self._callback(task.agent_id, result)

            self._completed.append(task)
            logger.info(f"[RETRY] Task {task.task_id} succeeded")
        except Exception as e:
            task.error = str(e)
            task.next_retry_at = time.time() + min(2 ** task.attempt, 60)
            self._queue.append(task)
            logger.error(f"[RETRY] Task {task.task_id} failed: {e}")

    def get_stats(self) -> dict:
        return {
            "pending": len(self._queue),
            "completed": len(self._completed),
            "failed": len(self._failed),
        }

    def clear(self):
        self._queue.clear()
        self._completed.clear()
        self._failed.clear()


retry_queue = RetryQueue()
