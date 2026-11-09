"""Token-aware sliding-window rate limiter persisted in an on-disk SQLite database.

Each admitted request is a row (tenant, ts, tokens). A tenant's usage is the sum of its rows
inside the last `window_seconds`. Check-and-insert runs inside `BEGIN IMMEDIATE`, which takes
SQLite's write lock up front, so concurrent requests (threads or processes) cannot both read
the same total and both slip under the limit.
"""

import math
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant  TEXT    NOT NULL,
    ts      REAL    NOT NULL,
    tokens  INTEGER NOT NULL CHECK (tokens >= 0)
);
CREATE INDEX IF NOT EXISTS idx_token_usage_tenant_ts ON token_usage (tenant, ts);
"""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    used: int
    limit: int
    retry_after: float = 0.0
    reservation_id: int | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


class SlidingWindowTokenLimiter:
    def __init__(
        self,
        db_path: str | Path,
        limit_tokens: int = 50_000,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db_path = str(db_path)
        self.limit = limit_tokens
        self.window = window_seconds
        self.clock = clock
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # One short-lived connection per operation: safe to call from any worker thread.
        conn = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def acquire(self, tenant: str, tokens: int) -> Decision:
        """Atomically admit `tokens` for `tenant` if the sliding window has room."""
        if tokens > self.limit:
            return Decision(allowed=False, used=0, limit=self.limit, retry_after=math.inf)

        now = self.clock()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM token_usage WHERE tenant = ? AND ts <= ?", (tenant, now - self.window))
                (used,) = conn.execute(
                    "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ?", (tenant,)
                ).fetchone()

                if used + tokens > self.limit:
                    retry_after = self._retry_after(conn, tenant, used, tokens, now)
                    conn.execute("COMMIT")
                    return Decision(allowed=False, used=used, limit=self.limit, retry_after=retry_after)

                cursor = conn.execute(
                    "INSERT INTO token_usage (tenant, ts, tokens) VALUES (?, ?, ?)", (tenant, now, tokens)
                )
                conn.execute("COMMIT")
                return Decision(allowed=True, used=used + tokens, limit=self.limit, reservation_id=cursor.lastrowid)
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def _retry_after(self, conn: sqlite3.Connection, tenant: str, used: int, tokens: int, now: float) -> float:
        """Seconds until enough of the oldest usage expires for this request to fit."""
        freed = 0
        rows = conn.execute("SELECT ts, tokens FROM token_usage WHERE tenant = ? ORDER BY ts", (tenant,))
        for ts, row_tokens in rows:
            freed += row_tokens
            if used - freed + tokens <= self.limit:
                return max(0.0, ts + self.window - now)
        return self.window

    def reconcile(self, reservation_id: int, actual_tokens: int) -> None:
        """Replace the pre-call estimate with the provider-reported usage."""
        with closing(self._connect()) as conn:
            conn.execute("UPDATE token_usage SET tokens = ? WHERE id = ?", (max(0, actual_tokens), reservation_id))

    def release(self, reservation_id: int) -> None:
        """Give tokens back when the request never reached a model."""
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM token_usage WHERE id = ?", (reservation_id,))

    def purge_expired(self) -> int:
        """Global sweep for tenants that went quiet (their rows aren't evicted by acquire)."""
        with closing(self._connect()) as conn:
            return conn.execute("DELETE FROM token_usage WHERE ts <= ?", (self.clock() - self.window,)).rowcount
