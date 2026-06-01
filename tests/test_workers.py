"""Tests for Worker Pool."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.workers import WorkerPool


@pytest.fixture
def pool():
    wp = WorkerPool(max_workers=2)
    yield wp
    wp.shutdown()


def add(a, b):
    return a + b


def slow_task(n):
    import time
    time.sleep(0.1)
    return n * 2


def failing_task():
    raise ValueError("Intentional error")


@pytest.mark.asyncio
async def test_submit_task(pool):
    result = await pool.submit(add, 2, 3)
    assert result == 5


@pytest.mark.asyncio
async def test_submit_many(pool):
    args = [(1, 2), (3, 4), (5, 6)]
    results = await pool.submit_many(add, args)
    assert results == [3, 7, 11]


@pytest.mark.asyncio
async def test_task_failure(pool):
    with pytest.raises(ValueError, match="Intentional error"):
        await pool.submit(failing_task)


@pytest.mark.asyncio
async def test_concurrent_execution(pool):
    tasks = [pool.submit(slow_task, i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    assert results == [0, 2, 4, 6, 8]


def test_get_stats(pool):
    stats = pool.get_stats()
    assert "total" in stats
    assert "max_workers" in stats
    assert stats["max_workers"] == 2
