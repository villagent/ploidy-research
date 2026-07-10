"""Task-isolation regressions for :class:`ploidy.store.DebateStore`."""

import asyncio
import stat
import threading

import pytest

from ploidy.store import DebateStore


@pytest.fixture
async def store(tmp_path):
    debate_store = DebateStore(db_path=tmp_path / "transactions.db")
    await debate_store.initialize()
    yield debate_store
    await debate_store.close()


async def test_other_task_write_waits_for_owner_rollback_then_persists(store):
    """A successful concurrent write must not join another task's rollback."""
    owner_ready = asyncio.Event()
    allow_rollback = asyncio.Event()
    writer_started = asyncio.Event()
    writer_returned = asyncio.Event()

    async def transaction_owner() -> None:
        try:
            async with store.transaction():
                await store.save_debate("owner", "must roll back")
                owner_ready.set()
                await allow_rollback.wait()
                raise RuntimeError("intentional rollback")
        except RuntimeError:
            pass

    async def concurrent_writer() -> None:
        await owner_ready.wait()
        writer_started.set()
        await store.save_debate("writer", "must persist")
        writer_returned.set()

    owner_task = asyncio.create_task(transaction_owner())
    writer_task = asyncio.create_task(concurrent_writer())

    await writer_started.wait()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(writer_returned.wait(), timeout=0.05)

    allow_rollback.set()
    await asyncio.gather(owner_task, writer_task)

    assert await store.get_debate("owner") is None
    writer = await store.get_debate("writer")
    assert writer is not None
    assert writer["prompt"] == "must persist"


async def test_child_task_cannot_inherit_transaction_ownership(store):
    """ContextVar inheritance alone must not let a child bypass the gate."""
    child_started = asyncio.Event()

    async def child_write() -> None:
        child_started.set()
        await store.save_debate("child", "separate commit")

    async with store.transaction():
        await store.save_debate("parent", "outer commit")
        child_task = asyncio.create_task(child_write())
        await child_started.wait()
        assert not child_task.done()

    await child_task
    assert await store.get_debate("parent") is not None
    assert await store.get_debate("child") is not None


async def test_nested_transactions_reuse_owner_and_rollback_together(store):
    """Nested transactions remain reentrant and share the outer outcome."""
    with pytest.raises(RuntimeError, match="roll back both"):
        async with store.transaction():
            await store.save_debate("outer", "outer")
            async with store.transaction():
                await store.save_debate("inner", "inner")
            raise RuntimeError("roll back both")

    assert await store.get_debate("outer") is None
    assert await store.get_debate("inner") is None


async def test_owner_read_is_immediate_but_other_task_waits_for_rollback(store):
    """Only the owner may observe writes that its transaction may roll back."""
    owner_ready = asyncio.Event()
    allow_rollback = asyncio.Event()
    reader_started = asyncio.Event()
    reader_returned = asyncio.Event()
    reader_result = None

    async def transaction_owner() -> None:
        try:
            async with store.transaction():
                await store.save_debate("uncommitted", "owner can see this")
                # ``wait_for(coroutine)`` creates a child Task on Python 3.11,
                # which correctly must not inherit transaction ownership. Use
                # the timeout context so this read stays in the owner Task.
                async with asyncio.timeout(0.1):
                    own_view = await store.get_debate("uncommitted")
                assert own_view is not None
                owner_ready.set()
                await allow_rollback.wait()
                raise RuntimeError("roll back")
        except RuntimeError:
            pass

    async def concurrent_reader() -> None:
        nonlocal reader_result
        await owner_ready.wait()
        reader_started.set()
        reader_result = await store.get_debate("uncommitted")
        reader_returned.set()

    owner_task = asyncio.create_task(transaction_owner())
    reader_task = asyncio.create_task(concurrent_reader())

    await reader_started.wait()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(reader_returned.wait(), timeout=0.05)

    allow_rollback.set()
    await asyncio.gather(owner_task, reader_task)

    assert reader_result is None


async def test_other_task_read_waits_for_commit_then_sees_row(store):
    """A blocked reader resumes against committed state, not a dirty snapshot."""
    owner_ready = asyncio.Event()
    allow_commit = asyncio.Event()
    reader_started = asyncio.Event()
    reader_returned = asyncio.Event()
    reader_result = None

    async def transaction_owner() -> None:
        async with store.transaction():
            await store.save_debate("committed", "visible after commit")
            owner_ready.set()
            await allow_commit.wait()

    async def concurrent_reader() -> None:
        nonlocal reader_result
        await owner_ready.wait()
        reader_started.set()
        reader_result = await store.get_debate("committed")
        reader_returned.set()

    owner_task = asyncio.create_task(transaction_owner())
    reader_task = asyncio.create_task(concurrent_reader())

    await reader_started.wait()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(reader_returned.wait(), timeout=0.05)

    allow_commit.set()
    await asyncio.gather(owner_task, reader_task)

    assert reader_result is not None
    assert reader_result["prompt"] == "visible after commit"


async def test_cancel_during_queued_begin_rolls_back_before_releasing_gate(store, monkeypatch):
    """A cancelled aiosqlite ``BEGIN`` cannot execute later without an owner."""
    db = store._db
    assert db is not None

    worker_started = threading.Event()
    release_worker = threading.Event()

    def block_worker() -> int:
        worker_started.set()
        release_worker.wait(timeout=2)
        return 1

    await db.create_function("block_worker", 0, block_worker)
    blocker = asyncio.create_task(db.execute("SELECT block_worker()"))
    assert await asyncio.to_thread(worker_started.wait, 1)

    original_execute = db.execute
    begin_queued = asyncio.Event()

    async def observe_begin(sql, *args, **kwargs):
        if sql == "BEGIN":
            begin_queued.set()
        return await original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(db, "execute", observe_begin)
    entered_body = asyncio.Event()

    async def cancelled_transaction() -> None:
        async with store.transaction():
            entered_body.set()

    transaction_task = asyncio.create_task(cancelled_transaction())
    await asyncio.wait_for(begin_queued.wait(), timeout=1)
    transaction_task.cancel()
    await asyncio.sleep(0)
    assert not entered_body.is_set()

    release_worker.set()
    cursor = await blocker
    await cursor.close()
    with pytest.raises(asyncio.CancelledError):
        await transaction_task

    assert not db.in_transaction
    await store.save_debate("after-begin-cancel", "connection remains usable")
    assert await store.get_debate("after-begin-cancel") is not None


async def test_cancel_during_commit_finishes_commit_before_releasing_gate(store, monkeypatch):
    """Commit ambiguity may cancel the caller, but never leaves a live transaction."""
    db = store._db
    assert db is not None
    commit_started = asyncio.Event()
    allow_commit = asyncio.Event()
    original_commit = db.commit

    async def gated_commit() -> None:
        commit_started.set()
        await allow_commit.wait()
        await original_commit()

    monkeypatch.setattr(db, "commit", gated_commit)

    async def cancelled_transaction() -> None:
        async with store.transaction():
            await store.save_debate("commit-boundary", "must commit atomically")

    transaction_task = asyncio.create_task(cancelled_transaction())
    await asyncio.wait_for(commit_started.wait(), timeout=1)
    transaction_task.cancel()
    await asyncio.sleep(0)
    assert not transaction_task.done()

    allow_commit.set()
    with pytest.raises(asyncio.CancelledError):
        await transaction_task

    assert not db.in_transaction
    committed = await store.get_debate("commit-boundary")
    assert committed is not None


async def test_repeated_cancel_during_rollback_finishes_cleanup(store, monkeypatch):
    """A second cancellation cannot interrupt rollback and leak transaction state."""
    db = store._db
    assert db is not None
    rollback_started = asyncio.Event()
    allow_rollback = asyncio.Event()
    original_rollback = db.rollback

    async def gated_rollback() -> None:
        rollback_started.set()
        await allow_rollback.wait()
        await original_rollback()

    monkeypatch.setattr(db, "rollback", gated_rollback)

    async def failing_transaction() -> None:
        async with store.transaction():
            await store.save_debate("rollback-boundary", "must disappear")
            raise RuntimeError("trigger rollback")

    transaction_task = asyncio.create_task(failing_transaction())
    await asyncio.wait_for(rollback_started.wait(), timeout=1)
    transaction_task.cancel()
    await asyncio.sleep(0)
    assert not transaction_task.done()

    allow_rollback.set()
    with pytest.raises(asyncio.CancelledError):
        await transaction_task

    assert not db.in_transaction
    assert await store.get_debate("rollback-boundary") is None


async def test_cancel_during_single_mutation_commit_finishes_boundary(store, monkeypatch):
    """Non-batched writes keep the gate until their queued commit completes."""
    db = store._db
    assert db is not None
    commit_started = asyncio.Event()
    allow_commit = asyncio.Event()
    original_commit = db.commit

    async def gated_commit() -> None:
        commit_started.set()
        await allow_commit.wait()
        await original_commit()

    monkeypatch.setattr(db, "commit", gated_commit)
    mutation_task = asyncio.create_task(
        store.save_debate("mutation-commit", "commit before cancel escapes")
    )
    await asyncio.wait_for(commit_started.wait(), timeout=1)
    mutation_task.cancel()
    await asyncio.sleep(0)
    assert not mutation_task.done()

    allow_commit.set()
    with pytest.raises(asyncio.CancelledError):
        await mutation_task

    assert not db.in_transaction
    assert await store.get_debate("mutation-commit") is not None


async def test_repeated_cancel_during_single_mutation_rollback_finishes_cleanup(store, monkeypatch):
    """Non-batched rollback also survives a cancellation during cleanup."""
    db = store._db
    assert db is not None
    rollback_started = asyncio.Event()
    allow_rollback = asyncio.Event()
    original_rollback = db.rollback

    async def gated_rollback() -> None:
        rollback_started.set()
        await allow_rollback.wait()
        await original_rollback()

    monkeypatch.setattr(db, "rollback", gated_rollback)

    async def failing_mutation() -> None:
        async with store._mutation() as connection:
            await connection.execute(
                "INSERT INTO debates (id, prompt) VALUES (?, ?)",
                ("mutation-rollback", "must disappear"),
            )
            raise RuntimeError("trigger mutation rollback")

    mutation_task = asyncio.create_task(failing_mutation())
    await asyncio.wait_for(rollback_started.wait(), timeout=1)
    mutation_task.cancel()
    await asyncio.sleep(0)
    assert not mutation_task.done()

    allow_rollback.set()
    with pytest.raises(asyncio.CancelledError):
        await mutation_task

    assert not db.in_transaction
    assert await store.get_debate("mutation-rollback") is None


async def test_database_and_wal_files_are_owner_only(tmp_path):
    """SQLite persistence files must not inherit a permissive process umask."""
    db_path = tmp_path / "private.db"
    debate_store = DebateStore(db_path=db_path)
    await debate_store.initialize()
    try:
        await debate_store.save_debate("permissions", "owner only")

        assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
        for suffix in ("-wal", "-shm"):
            sidecar = tmp_path / f"private.db{suffix}"
            if sidecar.exists():
                assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
    finally:
        await debate_store.close()


async def test_initialize_tightens_existing_database_permissions(tmp_path):
    """Reopening an older database upgrades its file permissions."""
    db_path = tmp_path / "legacy.db"
    db_path.touch(mode=0o644)
    db_path.chmod(0o644)

    debate_store = DebateStore(db_path=db_path)
    await debate_store.initialize()
    try:
        assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    finally:
        await debate_store.close()


async def test_cancelled_initialize_resets_connection_and_can_retry(tmp_path, monkeypatch):
    """Cancellation after connect cannot masquerade as a ready store."""
    debate_store = DebateStore(db_path=tmp_path / "cancelled-init.db")
    migrate_started = asyncio.Event()
    original_migrate = debate_store._migrate_schema

    async def blocked_migrate() -> None:
        migrate_started.set()
        await asyncio.Future()

    monkeypatch.setattr(debate_store, "_migrate_schema", blocked_migrate)
    initialize_task = asyncio.create_task(debate_store.initialize())
    await asyncio.wait_for(migrate_started.wait(), timeout=1)
    initialize_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await initialize_task

    assert debate_store._db is None
    monkeypatch.setattr(debate_store, "_migrate_schema", original_migrate)
    await debate_store.initialize()
    try:
        await debate_store.save_debate("after-init-cancel", "retry succeeded")
        assert await debate_store.get_debate("after-init-cancel") is not None
    finally:
        await debate_store.close()


async def test_failed_initialize_resets_connection_and_can_retry(tmp_path, monkeypatch):
    """Schema boot errors close the partial connection instead of poisoning retries."""
    debate_store = DebateStore(db_path=tmp_path / "failed-init.db")
    original_migrate = debate_store._migrate_schema

    async def failed_migrate() -> None:
        raise RuntimeError("migration failed")

    monkeypatch.setattr(debate_store, "_migrate_schema", failed_migrate)
    with pytest.raises(RuntimeError, match="migration failed"):
        await debate_store.initialize()

    assert debate_store._db is None
    monkeypatch.setattr(debate_store, "_migrate_schema", original_migrate)
    await debate_store.initialize()
    try:
        await debate_store.save_debate("after-init-error", "retry succeeded")
        assert await debate_store.get_debate("after-init-error") is not None
    finally:
        await debate_store.close()
