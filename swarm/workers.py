"""Worker Pool - Simple process-based parallel execution."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import uuid4

logger = logging.getLogger("swarm.workers")


@dataclass
class WorkerTask:
    task_id: str
    func_name: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class WorkerPool:
    """Simple worker pool for CPU and IO bound tasks."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(4, os.cpu_count() or 4)
        self._process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers * 2)
        self._tasks: dict[str, WorkerTask] = {}
        self._running = True

    async def submit(
        self,
        func: Callable,
        *args,
        use_process: bool = False,
        **kwargs,
    ) -> Any:
        task_id = uuid4().hex[:8]
        worker_task = WorkerTask(
            task_id=task_id,
            func_name=func.__name__,
            args=args,
            kwargs=kwargs,
        )
        self._tasks[task_id] = worker_task

        try:
            worker_task.status = "running"
            worker_task.started_at = time.time()

            loop = asyncio.get_event_loop()
            pool = self._process_pool if use_process else self._thread_pool

            future = loop.run_in_executor(pool, lambda: func(*args, **kwargs))
            result = await asyncio.wait_for(future, timeout=300)

            worker_task.status = "completed"
            worker_task.result = result
            worker_task.completed_at = time.time()
            return result

        except asyncio.TimeoutError:
            worker_task.status = "timeout"
            worker_task.error = "Task timed out after 300s"
            worker_task.completed_at = time.time()
            logger.error(f"Worker {task_id} timed out")
            raise

        except Exception as e:
            worker_task.status = "failed"
            worker_task.error = str(e)
            worker_task.completed_at = time.time()
            logger.error(f"Worker {task_id} failed: {e}")
            raise

    async def submit_many(
        self,
        func: Callable,
        args_list: list[tuple],
        kwargs_list: Optional[list[dict]] = None,
        use_process: bool = False,
    ) -> list[Any]:
        if kwargs_list is None:
            kwargs_list = [{} for _ in args_list]

        tasks = [
            self.submit(func, *args, use_process=use_process, **kwargs)
            for args, kwargs in zip(args_list, kwargs_list)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_stats(self) -> dict:
        completed = sum(1 for t in self._tasks.values() if t.status == "completed")
        failed = sum(1 for t in self._tasks.values() if t.status == "failed")
        running = sum(1 for t in self._tasks.values() if t.status == "running")
        return {
            "total": len(self._tasks),
            "completed": completed,
            "failed": failed,
            "running": running,
            "max_workers": self.max_workers,
        }

    def shutdown(self):
        self._running = False
        self._process_pool.shutdown(wait=False)
        self._thread_pool.shutdown(wait=False)


worker_pool = WorkerPool()
