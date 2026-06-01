"""Parallel Execution Engine - asyncio orchestration for parallel model calls."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .models import SwarmParticipant
from .providers import OpenSourceProvider, create_provider

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    task_id: str
    agent_id: str
    agent_name: str
    result: str
    success: bool
    latency_ms: float
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class ParallelEngine:
    """Execute multiple LLM calls in parallel with streaming aggregation."""

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._providers: dict[str, OpenSourceProvider] = {}

    def _get_provider(self, participant: SwarmParticipant) -> OpenSourceProvider:
        if participant.id not in self._providers:
            self._providers[participant.id] = create_provider(participant)
        return self._providers[participant.id]

    async def run_parallel(
        self,
        task_fn: Callable,
        participants: list[SwarmParticipant],
        **kwargs,
    ) -> list[TaskResult]:
        async def _wrapped(participant: SwarmParticipant) -> TaskResult:
            async with self._semaphore:
                start = time.time()
                try:
                    provider = self._get_provider(participant)
                    result = await task_fn(provider, participant, **kwargs)
                    latency = (time.time() - start) * 1000
                    return TaskResult(
                        task_id=f"{participant.id}_{int(start)}",
                        agent_id=participant.id,
                        agent_name=participant.name,
                        result=result,
                        success=True,
                        latency_ms=latency,
                    )
                except Exception as e:
                    latency = (time.time() - start) * 1000
                    logger.error(f"Agent {participant.name} failed: {e}")
                    return TaskResult(
                        task_id=f"{participant.id}_{int(start)}",
                        agent_id=participant.id,
                        agent_name=participant.name,
                        result="",
                        success=False,
                        latency_ms=latency,
                        error=str(e),
                    )

        tasks = [_wrapped(p) for p in participants]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    async def run_with_timeout(
        self,
        task_fn: Callable,
        participants: list[SwarmParticipant],
        timeout: float = 120.0,
        **kwargs,
    ) -> list[TaskResult]:
        async def _timed():
            return await self.run_parallel(task_fn, participants, **kwargs)

        try:
            return await asyncio.wait_for(_timed(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Parallel execution timed out after {timeout}s")
            return []

    async def run_sequential_with_early_stop(
        self,
        task_fn: Callable,
        participants: list[SwarmParticipant],
        stop_condition: Callable[[TaskResult], bool] = lambda r: False,
        **kwargs,
    ) -> list[TaskResult]:
        results = []
        for participant in participants:
            async with self._semaphore:
                start = time.time()
                try:
                    provider = self._get_provider(participant)
                    result = await task_fn(provider, participant, **kwargs)
                    latency = (time.time() - start) * 1000
                    task_result = TaskResult(
                        task_id=f"{participant.id}_{int(start)}",
                        agent_id=participant.id,
                        agent_name=participant.name,
                        result=result,
                        success=True,
                        latency_ms=latency,
                    )
                    results.append(task_result)
                    if stop_condition(task_result):
                        logger.info(f"Early stop triggered after {participant.name}")
                        break
                except Exception as e:
                    latency = (time.time() - start) * 1000
                    results.append(TaskResult(
                        task_id=f"{participant.id}_{int(start)}",
                        agent_id=participant.id,
                        agent_name=participant.name,
                        result="",
                        success=False,
                        latency_ms=latency,
                        error=str(e),
                    ))
        return results

    async def stream_results(
        self,
        task_fn: Callable,
        participants: list[SwarmParticipant],
        callback: Optional[Callable] = None,
        **kwargs,
    ) -> list[TaskResult]:
        queue: asyncio.Queue[TaskResult] = asyncio.Queue()

        async def _run_one(participant: SwarmParticipant):
            async with self._semaphore:
                start = time.time()
                try:
                    provider = self._get_provider(participant)
                    result = await task_fn(provider, participant, **kwargs)
                    latency = (time.time() - start) * 1000
                    task_result = TaskResult(
                        task_id=f"{participant.id}_{int(start)}",
                        agent_id=participant.id,
                        agent_name=participant.name,
                        result=result,
                        success=True,
                        latency_ms=latency,
                    )
                except Exception as e:
                    latency = (time.time() - start) * 1000
                    task_result = TaskResult(
                        task_id=f"{participant.id}_{int(start)}",
                        agent_id=participant.id,
                        agent_name=participant.name,
                        result="",
                        success=False,
                        latency_ms=latency,
                        error=str(e),
                    )
                await queue.put(task_result)
                if callback:
                    callback(task_result)

        tasks = [_run_one(p) for p in participants]
        asyncio.create_task(asyncio.gather(*tasks))

        results = []
        while len(results) < len(participants):
            try:
                result = await asyncio.wait_for(queue.get(), timeout=0.5)
                results.append(result)
            except asyncio.TimeoutError:
                continue

        return results

    async def close(self):
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()


parallel_engine = ParallelEngine()
