import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from task4_router.ratelimiter import SlidingWindowTokenLimiter


class FakeClock:
    def __init__(self, t: float = 1_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def limiter(tmp_path, clock):
    return SlidingWindowTokenLimiter(tmp_path / "rl.sqlite3", limit_tokens=50_000, window_seconds=60, clock=clock)


def rows(limiter, tenant=None) -> int:
    with sqlite3.connect(limiter.db_path) as conn:
        if tenant:
            return conn.execute("SELECT COUNT(*) FROM token_usage WHERE tenant = ?", (tenant,)).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]


def test_admits_up_to_the_limit_then_blocks(limiter):
    assert limiter.acquire("t1", 30_000).allowed
    assert limiter.acquire("t1", 20_000).allowed
    blocked = limiter.acquire("t1", 1)
    assert not blocked.allowed
    assert blocked.used == 50_000
    assert blocked.remaining == 0


def test_tenants_are_isolated(limiter):
    assert limiter.acquire("t1", 50_000).allowed
    assert limiter.acquire("t2", 50_000).allowed
    assert not limiter.acquire("t1", 1).allowed


def test_window_slides_rather_than_resetting(limiter, clock):
    limiter.acquire("t1", 30_000)          # t=0
    clock.t += 30
    limiter.acquire("t1", 20_000)          # t=30
    clock.t += 29
    assert not limiter.acquire("t1", 1).allowed          # t=59: both still in window
    clock.t += 1.01
    assert limiter.acquire("t1", 30_000).allowed         # t=60.01: first 30k expired, 20k remain
    assert not limiter.acquire("t1", 1).allowed


def test_retry_after_points_at_when_enough_tokens_expire(limiter, clock):
    limiter.acquire("t1", 10_000)          # t=0
    clock.t += 20
    limiter.acquire("t1", 40_000)          # t=20
    clock.t += 5                           # t=25
    assert limiter.acquire("t1", 5_000).retry_after == pytest.approx(35)   # needs the t=0 row gone (at 60)
    assert limiter.acquire("t1", 20_000).retry_after == pytest.approx(55)  # needs both rows gone (at 80)


def test_expired_rows_are_evicted_from_disk(limiter, clock):
    for _ in range(5):
        limiter.acquire("t1", 1_000)
    assert rows(limiter, "t1") == 5
    clock.t += 61
    limiter.acquire("t1", 1_000)
    assert rows(limiter, "t1") == 1


def test_purge_expired_sweeps_idle_tenants(limiter, clock):
    limiter.acquire("idle", 1_000)
    clock.t += 61
    assert limiter.purge_expired() == 1
    assert rows(limiter) == 0


def test_reconcile_replaces_estimate_with_actual_usage(limiter):
    d = limiter.acquire("t1", 40_000)
    limiter.reconcile(d.reservation_id, 1_000)
    assert limiter.acquire("t1", 49_000).allowed


def test_release_returns_tokens(limiter):
    d = limiter.acquire("t1", 50_000)
    limiter.release(d.reservation_id)
    assert limiter.acquire("t1", 50_000).allowed


def test_request_larger_than_limit_is_never_admitted(limiter):
    d = limiter.acquire("t1", 50_001)
    assert not d.allowed and d.retry_after == float("inf")


def test_state_persists_across_instances(tmp_path, clock):
    path = tmp_path / "rl.sqlite3"
    SlidingWindowTokenLimiter(path, clock=clock).acquire("t1", 50_000)
    assert not SlidingWindowTokenLimiter(path, clock=clock).acquire("t1", 1).allowed


def test_concurrent_requests_cannot_overshoot_the_limit(tmp_path):
    limiter = SlidingWindowTokenLimiter(tmp_path / "rl.sqlite3", limit_tokens=50_000)
    with ThreadPoolExecutor(max_workers=20) as pool:
        decisions = list(pool.map(lambda _: limiter.acquire("t1", 5_000), range(40)))
    assert sum(d.allowed for d in decisions) == 10
    assert rows(limiter, "t1") == 10
