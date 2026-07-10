"""Persistence layer for Ploidy.

Stores debate history, session contexts, and convergence results
using aiosqlite for async SQLite access. All debate data is persisted
so that future sessions can reference past decisions -- this is how
Session A (the experienced session) accumulates context over time.

Tables:
    debates      -- Debate metadata (id, prompt, status, timestamps)
    sessions     -- Session contexts and roles within a debate
    messages     -- Individual debate messages with phase information
    convergence  -- Convergence results and synthesis outputs
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Literal, TypeVar

import aiosqlite

from ploidy.exceptions import PloidyError  # noqa: I001

_T = TypeVar("_T")


def default_db_path() -> Path:
    """Resolve the database path from the current environment."""
    return Path(os.environ.get("PLOIDY_DB_PATH", str(Path.home() / ".ploidy" / "ploidy.db")))


_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS debates (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    debate_id TEXT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    base_prompt TEXT NOT NULL,
    context_documents TEXT NOT NULL DEFAULT '[]',
    delivery_mode TEXT NOT NULL DEFAULT 'none',
    compressed_summary TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id TEXT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    phase TEXT NOT NULL,
    content TEXT NOT NULL,
    action TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS convergence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id TEXT NOT NULL UNIQUE REFERENCES debates(id) ON DELETE CASCADE,
    synthesis TEXT NOT NULL,
    confidence REAL NOT NULL,
    points_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- OAuth 2.0 Authorization Server tables.
-- See ``planning/oauth-integration.md`` for the design. Co-located in
-- the existing SQLite DB so operators get one file to back up.

CREATE TABLE IF NOT EXISTS oauth_clients (
    client_id TEXT PRIMARY KEY,
    client_secret_hash TEXT,
    redirect_uris TEXT NOT NULL,
    grant_types TEXT NOT NULL,
    token_endpoint_auth_method TEXT NOT NULL DEFAULT 'none',
    client_name TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS oauth_codes (
    code TEXT PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES oauth_clients(client_id) ON DELETE CASCADE,
    redirect_uri TEXT NOT NULL,
    scopes TEXT NOT NULL,
    code_challenge TEXT NOT NULL,
    code_challenge_method TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS oauth_tokens (
    token TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    client_id TEXT NOT NULL REFERENCES oauth_clients(client_id) ON DELETE CASCADE,
    scopes TEXT NOT NULL,
    expires_at TEXT,
    revoked INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_SESSION_MIGRATIONS = (
    (
        "context_documents",
        "ALTER TABLE sessions ADD COLUMN context_documents TEXT NOT NULL DEFAULT '[]'",
    ),
    ("delivery_mode", "ALTER TABLE sessions ADD COLUMN delivery_mode TEXT NOT NULL DEFAULT 'none'"),
    ("compressed_summary", "ALTER TABLE sessions ADD COLUMN compressed_summary TEXT"),
    ("metadata_json", "ALTER TABLE sessions ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"),
    ("model", "ALTER TABLE sessions ADD COLUMN model TEXT"),
    ("effort", "ALTER TABLE sessions ADD COLUMN effort TEXT NOT NULL DEFAULT 'high'"),
)

_DEBATE_MIGRATIONS = (
    (
        "paused_context",
        "ALTER TABLE debates ADD COLUMN paused_context TEXT",
    ),
    (
        "config_json",
        "ALTER TABLE debates ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}'",
    ),
    # Multitenancy: NULL == unscoped legacy row. Filter on owner_id at the
    # service layer; the column alone does not enforce isolation.
    ("owner_id", "ALTER TABLE debates ADD COLUMN owner_id TEXT"),
)

_MESSAGE_MIGRATIONS = (
    ("round", "ALTER TABLE messages ADD COLUMN round INTEGER NOT NULL DEFAULT 1"),
)

_CONVERGENCE_MIGRATIONS = (
    ("meta_analysis", "ALTER TABLE convergence ADD COLUMN meta_analysis TEXT"),
)

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_debates_status ON debates(status);
CREATE INDEX IF NOT EXISTS idx_debates_owner ON debates(owner_id);
CREATE INDEX IF NOT EXISTS idx_sessions_debate_id ON sessions(debate_id);
CREATE INDEX IF NOT EXISTS idx_messages_debate_id ON messages(debate_id);
CREATE INDEX IF NOT EXISTS idx_oauth_tokens_client ON oauth_tokens(client_id);
CREATE INDEX IF NOT EXISTS idx_oauth_tokens_expires ON oauth_tokens(expires_at);
CREATE INDEX IF NOT EXISTS idx_oauth_codes_expires ON oauth_codes(expires_at);
"""


def _require_db(db: aiosqlite.Connection | None) -> aiosqlite.Connection:
    """Validate that the database connection is initialized."""
    if db is None:
        raise PloidyError("Store not initialized — call initialize() first")
    return db


class DebateStore:
    """Async SQLite store for debate data.

    Provides CRUD operations for debates, sessions, messages,
    and convergence results. Supports async context manager usage.

    Usage::

        async with DebateStore() as store:
            await store.save_debate("d1", "Should we use Rust?")
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the store.

        Args:
            db_path: Path to the SQLite database file.
                     Defaults to ``~/.ploidy/ploidy.db``.
        """
        self.db_path = db_path or default_db_path()
        self._db: aiosqlite.Connection | None = None
        self._initialize_lock = asyncio.Lock()
        # One aiosqlite connection is shared by every request using this store.
        # Gate all mutations *before* their first execute so another task cannot
        # accidentally join an explicit transaction and be rolled back with it.
        self._write_gate = asyncio.Lock()
        self._transaction_owner: asyncio.Task[object] | None = None
        self._transaction_depth: ContextVar[int] = ContextVar(
            f"ploidy_transaction_depth_{id(self)}",
            default=0,
        )

    async def __aenter__(self) -> DebateStore:
        """Enter the async context manager -- open DB and create tables."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the async context manager -- close DB."""
        await self.close()

    async def initialize(self) -> None:
        """Create database tables if they don't exist.

        Enables WAL mode for concurrent read/write access and sets up
        the schema for debates, sessions, messages, and convergence results.

        Idempotent: calling more than once returns immediately so callers
        that share a store across lazy-init sites (e.g. the OAuth provider
        alongside the debate service) can safely re-enter without leaking
        connections.
        """
        async with self._initialize_lock:
            if self._db is not None:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            db = await aiosqlite.connect(self.db_path)
            self._db = db
            try:
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA busy_timeout=5000")
                await db.execute("PRAGMA foreign_keys=ON")
                await db.executescript(_CREATE_TABLES)
                await self._migrate_schema()
                await db.executescript(_CREATE_INDEXES)
                await self._finish_boundary(db.commit())
                self._secure_database_files()
            except BaseException:
                try:
                    await self._finish_boundary(db.close())
                except BaseException:
                    # Preserve the boot failure/cancellation that triggered
                    # cleanup; the connection is invalidated below either way.
                    pass
                self._db = None
                raise

    def _secure_database_files(self) -> None:
        """Restrict the SQLite database and live sidecars to the owner."""
        if str(self.db_path) == ":memory:":
            return

        for path in (
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        ):
            try:
                path.chmod(0o600)
            except FileNotFoundError:
                # SQLite may remove a WAL/SHM sidecar between transactions.
                continue

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Batch every commit inside the block into a single SQLite transaction.

        Mutations made by the owning task join the transaction. Mutations from
        any other task wait at the write gate until the outer transaction has
        committed or rolled back, then run and commit independently. Nested
        ``transaction()`` calls from the owner reuse the outer transaction.
        """
        db = _require_db(self._db)
        task = asyncio.current_task()
        if task is None:
            raise PloidyError("Transactions require a running asyncio task")

        if self._owns_transaction():
            # Nested call: defer commit/rollback to the outer transaction.
            token = self._transaction_depth.set(self._transaction_depth.get() + 1)
            try:
                yield
            finally:
                self._transaction_depth.reset(token)
            return

        async with self._write_gate:
            self._transaction_owner = task
            token = self._transaction_depth.set(1)
            try:
                try:
                    await self._finish_boundary(db.execute("BEGIN"))
                    yield
                except BaseException:
                    await self._finish_boundary(db.rollback())
                    raise
                else:
                    await self._finish_boundary(db.commit())
                    self._secure_database_files()
            finally:
                self._transaction_depth.reset(token)
                self._transaction_owner = None

    @staticmethod
    async def _finish_boundary(operation: Awaitable[_T]) -> _T:
        """Finish a queued SQLite boundary operation before propagating cancel.

        Cancelling an await on aiosqlite does not remove the operation from its
        worker queue.  Letting the write gate go at that point can expose a live
        ``BEGIN`` or an unfinished ``COMMIT``/``ROLLBACK`` to the next task.
        Shield the worker operation, remember cancellation requests, and raise
        only after the connection has reached the requested boundary.
        """
        boundary = asyncio.ensure_future(operation)
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                result = await asyncio.shield(boundary)
                break
            except asyncio.CancelledError as exc:
                cancellation = exc
                if boundary.done():
                    result = boundary.result()
                    break

        if cancellation is not None:
            raise cancellation
        return result

    def _owns_transaction(self) -> bool:
        """Return whether the current task owns this store's transaction."""
        task = asyncio.current_task()
        return (
            task is not None
            and self._transaction_owner is task
            and self._transaction_depth.get() > 0
        )

    @asynccontextmanager
    async def _mutation(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield the connection only after this task may safely mutate it.

        The transaction owner is already protected by the outer write gate, so
        its nested store calls execute immediately and defer commit/rollback.
        Every other task acquires the gate before executing any write and owns
        the resulting commit or rollback.
        """
        db = _require_db(self._db)
        if self._owns_transaction():
            yield db
            return

        async with self._write_gate:
            try:
                yield db
            except BaseException:
                await self._finish_boundary(db.rollback())
                raise
            else:
                await self._finish_boundary(db.commit())
                self._secure_database_files()

    @asynccontextmanager
    async def _read(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield a connection that cannot observe another task's transaction.

        SQLite exposes uncommitted writes to every coroutine sharing the same
        connection.  The transaction owner must be able to read its own writes,
        but every other task waits at the same gate used by mutations until the
        transaction commits or rolls back.
        """
        db = _require_db(self._db)
        if self._owns_transaction():
            yield db
            return

        async with self._write_gate:
            yield db

    async def _migrate_schema(self) -> None:
        """Apply additive schema migrations for older databases."""
        db = _require_db(self._db)

        async def _apply_migrations(table: str, migrations: tuple) -> None:
            cursor = await db.execute(f"PRAGMA table_info({table})")
            columns = {row["name"] for row in await cursor.fetchall()}
            for column, statement in migrations:
                if column not in columns:
                    await db.execute(statement)

        await _apply_migrations("sessions", _SESSION_MIGRATIONS)
        await _apply_migrations("debates", _DEBATE_MIGRATIONS)
        await _apply_migrations("messages", _MESSAGE_MIGRATIONS)
        await _apply_migrations("convergence", _CONVERGENCE_MIGRATIONS)

    # ------------------------------------------------------------------
    # Debates
    # ------------------------------------------------------------------

    async def save_debate(
        self,
        debate_id: str,
        prompt: str,
        config: dict | None = None,
        owner_id: str | None = None,
    ) -> None:
        """Persist a new debate record.

        Args:
            debate_id: Unique identifier for the debate.
            prompt: The decision prompt for the debate.
            config: Optional debate configuration dict (ploidy, injection, etc.).
            owner_id: Tenant/owner identifier for multitenant filtering.
                NULL treats the debate as unscoped.
        """
        async with self._mutation() as db:
            await db.execute(
                "INSERT INTO debates (id, prompt, config_json, owner_id) VALUES (?, ?, ?, ?)",
                (debate_id, prompt, json.dumps(config or {}), owner_id),
            )

    async def get_debate(self, debate_id: str, owner_id: str | None = None) -> dict | None:
        """Retrieve a debate by its ID.

        Args:
            debate_id: The debate to look up.
            owner_id: When set, enforce that the debate belongs to this owner.

        Returns:
            Debate record as a dict, or None if not found (or owned by someone else).
        """
        async with self._read() as db:
            if owner_id is None:
                cursor = await db.execute(
                    "SELECT id, prompt, status, owner_id, config_json, created_at, updated_at "
                    "FROM debates WHERE id = ?",
                    (debate_id,),
                )
            else:
                cursor = await db.execute(
                    "SELECT id, prompt, status, owner_id, config_json, created_at, updated_at "
                    "FROM debates WHERE id = ? AND owner_id = ?",
                    (debate_id, owner_id),
                )
            row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def list_debates(self, limit: int = 50, owner_id: str | None = None) -> list[dict]:
        """List recent debates.

        Args:
            limit: Maximum number of debates to return.
            owner_id: When set, restrict to debates owned by this id.

        Returns:
            List of debate records, most recent first.
        """
        async with self._read() as db:
            if owner_id is None:
                cursor = await db.execute(
                    "SELECT id, prompt, status, owner_id, config_json, created_at, updated_at "
                    "FROM debates ORDER BY created_at DESC LIMIT ?",
                    (min(limit, 200),),
                )
            else:
                cursor = await db.execute(
                    "SELECT id, prompt, status, owner_id, config_json, created_at, updated_at "
                    "FROM debates WHERE owner_id = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (owner_id, min(limit, 200)),
                )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def list_active_debates(self) -> list[dict]:
        """List active debates for state recovery.

        Returns:
            List of active debate records (all owners — recovery is global).
        """
        async with self._read() as db:
            cursor = await db.execute(
                "SELECT id, prompt, status, owner_id, config_json, created_at, updated_at "
                "FROM debates WHERE status = 'active' ORDER BY created_at",
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def update_debate_status(self, debate_id: str, status: str) -> None:
        """Update a debate's status.

        Args:
            debate_id: The debate to update.
            status: New status value.
        """
        async with self._mutation() as db:
            await db.execute(
                "UPDATE debates SET status = ?, updated_at = datetime('now') WHERE id = ?",
                (status, debate_id),
            )

    async def save_paused_context(self, debate_id: str, context: dict) -> None:
        """Persist paused auto-debate context alongside the debate record.

        This ensures HITL paused state survives server restarts.

        Args:
            debate_id: The paused debate.
            context: The auto-debate context dict to serialize.
        """
        async with self._mutation() as db:
            await db.execute(
                "UPDATE debates SET paused_context = ?, updated_at = datetime('now') WHERE id = ?",
                (json.dumps(context), debate_id),
            )

    async def load_paused_context(self, debate_id: str) -> dict | None:
        """Load persisted paused context for a debate.

        Args:
            debate_id: The debate to look up.

        Returns:
            The paused context dict, or None if not found.
        """
        async with self._read() as db:
            cursor = await db.execute(
                "SELECT paused_context FROM debates WHERE id = ?",
                (debate_id,),
            )
            row = await cursor.fetchone()
        if row is None or row["paused_context"] is None:
            return None
        return json.loads(row["paused_context"])

    async def clear_paused_context(self, debate_id: str) -> None:
        """Clear persisted paused context (e.g., after resume or cancel).

        Args:
            debate_id: The debate to clear paused context for.
        """
        async with self._mutation() as db:
            await db.execute(
                "UPDATE debates SET paused_context = NULL, "
                "updated_at = datetime('now') WHERE id = ?",
                (debate_id,),
            )

    async def list_paused_debates(self) -> list[dict]:
        """List paused debates for state recovery.

        Returns:
            List of paused debate records.
        """
        async with self._read() as db:
            cursor = await db.execute(
                "SELECT id, prompt, status, paused_context, created_at, updated_at "
                "FROM debates WHERE status = 'paused' ORDER BY created_at",
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def save_session(
        self,
        session_id: str,
        debate_id: str,
        role: str,
        base_prompt: str,
        context_documents: list[str] | None = None,
        delivery_mode: str = "none",
        compressed_summary: str | None = None,
        metadata: dict | None = None,
        model: str | None = None,
        effort: str = "high",
    ) -> None:
        """Persist a new session record.

        Args:
            session_id: Unique identifier for the session.
            debate_id: The debate this session belongs to.
            role: Session role ('deep', 'semi_fresh', or 'fresh').
            base_prompt: The decision prompt for this session.
            model: Model identifier used for this session.
            effort: Effort level for this session.
        """
        async with self._mutation() as db:
            await db.execute(
                "INSERT INTO sessions "
                "(id, debate_id, role, base_prompt, context_documents, delivery_mode, "
                "compressed_summary, metadata_json, model, effort) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    debate_id,
                    role,
                    base_prompt,
                    json.dumps(context_documents or []),
                    delivery_mode,
                    compressed_summary,
                    json.dumps(metadata or {}),
                    model,
                    effort,
                ),
            )

    async def get_sessions(self, debate_id: str) -> list[dict]:
        """Retrieve all sessions for a debate.

        Args:
            debate_id: The debate to look up sessions for.

        Returns:
            List of session records.
        """
        async with self._read() as db:
            cursor = await db.execute(
                "SELECT id, debate_id, role, base_prompt, context_documents, delivery_mode, "
                "compressed_summary, metadata_json, model, effort, created_at "
                "FROM sessions WHERE debate_id = ?",
                (debate_id,),
            )
            rows = await cursor.fetchall()
        sessions = []
        for row in rows:
            session = dict(row)
            session["context_documents"] = json.loads(session.get("context_documents") or "[]")
            session["metadata"] = json.loads(session.pop("metadata_json", "{}") or "{}")
            sessions.append(session)
        return sessions

    async def update_session_context(
        self,
        session_id: str,
        *,
        context_documents: list[str] | None = None,
        delivery_mode: str | None = None,
        compressed_summary: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Update persisted session context fields."""
        async with self._mutation() as db:
            await db.execute(
                "UPDATE sessions SET context_documents = ?, delivery_mode = ?, "
                "compressed_summary = ?, metadata_json = ? WHERE id = ?",
                (
                    json.dumps(context_documents or []),
                    delivery_mode or "none",
                    compressed_summary,
                    json.dumps(metadata or {}),
                    session_id,
                ),
            )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def save_message(
        self,
        debate_id: str,
        session_id: str,
        phase: str,
        content: str,
        action: str | None = None,
    ) -> None:
        """Persist a debate message.

        Args:
            debate_id: The debate this message belongs to.
            session_id: The session that authored this message.
            phase: The debate phase when this message was created.
            content: The message content.
            action: Optional semantic action classifying the message.
        """
        async with self._mutation() as db:
            await db.execute(
                "INSERT INTO messages (debate_id, session_id, phase, content, action) "
                "VALUES (?, ?, ?, ?, ?)",
                (debate_id, session_id, phase, content, action),
            )

    async def get_messages(self, debate_id: str) -> list[dict]:
        """Retrieve all messages for a debate, ordered by creation time.

        Args:
            debate_id: The debate to look up messages for.

        Returns:
            List of message records.
        """
        async with self._read() as db:
            cursor = await db.execute(
                "SELECT id, debate_id, session_id, phase, content, action, timestamp "
                "FROM messages WHERE debate_id = ? ORDER BY id",
                (debate_id,),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------

    async def save_convergence(
        self,
        debate_id: str,
        synthesis: str,
        confidence: float,
        points_json: str,
        meta_analysis: str | None = None,
    ) -> None:
        """Persist a convergence result.

        Args:
            debate_id: The debate this result belongs to.
            synthesis: The synthesized recommendation.
            confidence: Confidence score (0.0 to 1.0).
            points_json: JSON string of convergence points.
            meta_analysis: Optional LLM meta-analysis text.
        """
        async with self._mutation() as db:
            await db.execute(
                "INSERT INTO convergence "
                "(debate_id, synthesis, confidence, points_json, meta_analysis) "
                "VALUES (?, ?, ?, ?, ?)",
                (debate_id, synthesis, confidence, points_json, meta_analysis),
            )

    async def get_convergence(self, debate_id: str) -> dict | None:
        """Retrieve the convergence result for a debate.

        Args:
            debate_id: The debate to look up.

        Returns:
            Convergence record as a dict, or None if not found.
        """
        async with self._read() as db:
            cursor = await db.execute(
                "SELECT debate_id, synthesis, confidence, points_json, meta_analysis, created_at "
                "FROM convergence WHERE debate_id = ?",
                (debate_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    async def save_convergence_and_complete(
        self,
        debate_id: str,
        synthesis: str,
        confidence: float,
        points_json: str,
        meta_analysis: str | None = None,
    ) -> None:
        """Atomically save convergence result and mark debate as complete.

        Args:
            debate_id: The debate to finalize.
            synthesis: The synthesized recommendation.
            confidence: Confidence score (0.0 to 1.0).
            points_json: JSON string of convergence points.
            meta_analysis: Optional LLM meta-analysis text.
        """
        async with self._mutation() as db:
            await db.execute(
                "INSERT INTO convergence "
                "(debate_id, synthesis, confidence, points_json, meta_analysis) "
                "VALUES (?, ?, ?, ?, ?)",
                (debate_id, synthesis, confidence, points_json, meta_analysis),
            )
            await db.execute(
                "UPDATE debates SET status = 'complete', updated_at = datetime('now') WHERE id = ?",
                (debate_id,),
            )

    async def delete_debate(self, debate_id: str) -> None:
        """Permanently delete a debate and all associated data.

        Removes messages, sessions, convergence results, and the debate itself.

        Args:
            debate_id: The debate to delete.
        """
        async with self._mutation() as db:
            await db.execute("DELETE FROM messages WHERE debate_id = ?", (debate_id,))
            await db.execute("DELETE FROM convergence WHERE debate_id = ?", (debate_id,))
            await db.execute("DELETE FROM sessions WHERE debate_id = ?", (debate_id,))
            await db.execute("DELETE FROM debates WHERE id = ?", (debate_id,))

    async def purge_terminal_before(self, cutoff_iso: str) -> int:
        """Delete completed/cancelled debates whose updated_at < cutoff.

        Active and paused debates are preserved regardless of age so long
        running sessions are never reaped. Related messages, sessions, and
        convergence rows are cleaned up by the ON DELETE CASCADE FK wired
        into the schema.

        Args:
            cutoff_iso: ISO-8601 UTC timestamp. Rows older than this are
                eligible for deletion.

        Returns:
            Number of debate rows removed.
        """
        async with self._mutation() as db:
            cursor = await db.execute(
                "DELETE FROM debates WHERE status IN ('complete', 'cancelled') AND updated_at < ?",
                (cutoff_iso,),
            )
            return cursor.rowcount or 0

    async def vacuum(self) -> None:
        """Run ``VACUUM`` to reclaim space after a bulk delete.

        SQLite disallows VACUUM inside a transaction, so callers must
        ensure no ``transaction()`` context is active when this runs.
        """
        async with self._mutation() as db:
            await db.execute("VACUUM")

    async def close(self) -> None:
        """Close the database connection."""
        async with self._initialize_lock:
            if self._db is not None:
                db = self._db
                try:
                    await self._finish_boundary(db.close())
                finally:
                    self._db = None

    # ------------------------------------------------------------------
    # OAuth 2.0 storage (see planning/oauth-integration.md)
    # ------------------------------------------------------------------

    async def save_oauth_client(
        self,
        client_id: str,
        *,
        redirect_uris: list[str],
        grant_types: list[str],
        client_secret_hash: str | None = None,
        token_endpoint_auth_method: str = "none",
        client_name: str | None = None,
    ) -> None:
        """Register an OAuth client (RFC 7591 DCR or admin-provisioned)."""
        async with self._mutation() as db:
            await db.execute(
                "INSERT INTO oauth_clients "
                "(client_id, client_secret_hash, redirect_uris, grant_types, "
                "token_endpoint_auth_method, client_name) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    client_id,
                    client_secret_hash,
                    json.dumps(redirect_uris),
                    json.dumps(grant_types),
                    token_endpoint_auth_method,
                    client_name,
                ),
            )

    async def get_oauth_client(self, client_id: str) -> dict | None:
        """Retrieve a registered client by id, or None when unknown."""
        async with self._read() as db:
            cursor = await db.execute(
                "SELECT client_id, client_secret_hash, redirect_uris, grant_types, "
                "token_endpoint_auth_method, client_name, created_at "
                "FROM oauth_clients WHERE client_id = ?",
                (client_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        record = dict(row)
        record["redirect_uris"] = json.loads(record["redirect_uris"])
        record["grant_types"] = json.loads(record["grant_types"])
        return record

    async def save_oauth_code(
        self,
        code: str,
        *,
        client_id: str,
        redirect_uri: str,
        scopes: list[str],
        code_challenge: str,
        code_challenge_method: str,
        expires_at: str,
    ) -> None:
        """Persist a freshly-issued authorization code pending /token exchange."""
        async with self._mutation() as db:
            await db.execute(
                "INSERT INTO oauth_codes "
                "(code, client_id, redirect_uri, scopes, code_challenge, "
                "code_challenge_method, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    code,
                    client_id,
                    redirect_uri,
                    json.dumps(scopes),
                    code_challenge,
                    code_challenge_method,
                    expires_at,
                ),
            )

    # Single-source-of-truth SELECT for live OAuth codes. The ``used = 0
    # AND expires_at >= datetime('now')`` clause filters consumed /
    # expired rows at the DB layer, which collapses the previous
    # three-query dance (fetch row → compare ``used`` in Python → fetch
    # ``datetime('now')`` → compare expiry in Python) into a single
    # round trip. SQLite compares the ISO timestamps as strings, which
    # is correct because ``datetime('now')`` and our stored
    # ``YYYY-MM-DD HH:MM:SS`` values share lexical ordering.
    _OAUTH_CODE_LIVE_SELECT = (
        "SELECT code, client_id, redirect_uri, scopes, code_challenge, "
        "code_challenge_method, expires_at, used, created_at "
        "FROM oauth_codes "
        "WHERE code = ? AND used = 0 AND expires_at >= datetime('now')"
    )

    async def get_oauth_code_for_load(self, code: str) -> dict | None:
        """Read-only code lookup used by ``load_authorization_code``.

        Unlike ``consume_oauth_code`` this does not flip ``used``; the
        MCP SDK calls this method first to gather pre-exchange metadata
        (PKCE challenge, redirect URI) before invoking the consuming
        exchange call. A row is returned only when the code is known,
        unused, and not yet expired.
        """
        async with self._read() as db:
            cursor = await db.execute(self._OAUTH_CODE_LIVE_SELECT, (code,))
            row = await cursor.fetchone()
        if row is None:
            return None
        record = dict(row)
        record["scopes"] = json.loads(record["scopes"])
        return record

    async def consume_oauth_code(self, code: str) -> dict | None:
        """Atomically look up and mark a code as used.

        Returns the full code record on first use, ``None`` if the code
        is unknown, already-used, or has expired. The single-use guard is
        critical: an attacker who captures a code must not be able to
        redeem it twice.
        """
        async with self.transaction():
            async with self._mutation() as db:
                cursor = await db.execute(self._OAUTH_CODE_LIVE_SELECT, (code,))
                row = await cursor.fetchone()
                if row is None:
                    return None
                record = dict(row)
                await db.execute("UPDATE oauth_codes SET used = 1 WHERE code = ?", (code,))
                record["scopes"] = json.loads(record["scopes"])
                return record

    async def save_oauth_token(
        self,
        token: str,
        *,
        kind: Literal["access", "refresh"],
        client_id: str,
        scopes: list[str],
        expires_at: str | None = None,
    ) -> None:
        """Persist an access or refresh token."""
        async with self._mutation() as db:
            await db.execute(
                "INSERT INTO oauth_tokens (token, kind, client_id, scopes, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (token, kind, client_id, json.dumps(scopes), expires_at),
            )

    async def get_oauth_token(self, token: str) -> dict | None:
        """Resolve a token to its record, or ``None`` when revoked / expired / unknown.

        Liveness (``revoked = 0``) and expiry (``expires_at IS NULL OR
        >= datetime('now')``) are both filtered at the SQL layer to save
        a round trip per lookup — this is on the auth hot path.
        """
        async with self._read() as db:
            cursor = await db.execute(
                "SELECT token, kind, client_id, scopes, expires_at, revoked, created_at "
                "FROM oauth_tokens "
                "WHERE token = ? AND revoked = 0 "
                "AND (expires_at IS NULL OR expires_at >= datetime('now'))",
                (token,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        record = dict(row)
        record["scopes"] = json.loads(record["scopes"])
        return record

    async def revoke_oauth_token(self, token: str) -> None:
        """Mark a token as revoked. Idempotent — unknown tokens no-op silently."""
        async with self._mutation() as db:
            await db.execute("UPDATE oauth_tokens SET revoked = 1 WHERE token = ?", (token,))

    async def purge_oauth_expired(self) -> int:
        """Delete expired/used codes and expired/revoked tokens.

        Returns the combined row count removed. Intended to be called
        from the retention loop alongside ``purge_terminal_before``.
        """
        async with self.transaction():
            async with self._mutation() as db:
                # Remove codes that are past their expiry or already consumed.
                c1 = await db.execute(
                    "DELETE FROM oauth_codes WHERE used = 1 OR expires_at < datetime('now')"
                )
                # Tokens: revoked OR (expires_at set AND past).
                c2 = await db.execute(
                    "DELETE FROM oauth_tokens WHERE revoked = 1 "
                    "OR (expires_at IS NOT NULL AND expires_at < datetime('now'))"
                )
                return (c1.rowcount or 0) + (c2.rowcount or 0)
