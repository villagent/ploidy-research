"""Multitenant isolation tests at the service layer.

Two tenants create debates. Each tenant must see only its own; cross-tenant
lookups surface as 'not found' (the same error a non-existent debate would
emit), not as a permission-denied leak.
"""

import pytest

from ploidy import server
from ploidy.service import DebateService
from ploidy.store import DebateStore


@pytest.fixture
async def svc(tmp_path):
    store = DebateStore(db_path=tmp_path / "mt.db")
    service = DebateService(store=store)
    await service.initialize()
    yield service
    await service.shutdown()


async def test_owner_sees_own_debates_only(svc):
    a_res = await svc.start_debate("tenant A prompt", owner_id="tenant-a")
    b_res = await svc.start_debate("tenant B prompt", owner_id="tenant-b")

    a_history = await svc.history(owner_id="tenant-a")
    b_history = await svc.history(owner_id="tenant-b")

    a_ids = {d["id"] for d in a_history["debates"]}
    b_ids = {d["id"] for d in b_history["debates"]}

    assert a_res["debate_id"] in a_ids
    assert a_res["debate_id"] not in b_ids
    assert b_res["debate_id"] in b_ids
    assert b_res["debate_id"] not in a_ids


async def test_cross_tenant_status_is_not_found(svc):
    res = await svc.start_debate("A's debate", owner_id="tenant-a")

    with pytest.raises(Exception, match="not found"):
        await svc.status(res["debate_id"], owner_id="tenant-b")


async def test_cross_tenant_cancel_is_rejected(svc):
    res = await svc.start_debate("A's debate", owner_id="tenant-a")

    with pytest.raises(Exception, match="not found"):
        await svc.cancel(res["debate_id"], owner_id="tenant-b")

    # A can still cancel its own.
    cancelled = await svc.cancel(res["debate_id"], owner_id="tenant-a")
    assert cancelled["status"] == "cancelled"


async def test_cross_tenant_delete_is_rejected(svc):
    res = await svc.start_debate("A's debate", owner_id="tenant-a")

    with pytest.raises(Exception, match="not found"):
        await svc.delete(res["debate_id"], owner_id="tenant-b")

    deleted = await svc.delete(res["debate_id"], owner_id="tenant-a")
    assert deleted["status"] == "deleted"


async def test_cross_tenant_cannot_submit_position(svc):
    start = await svc.start_debate("A's", owner_id="tenant-a")
    deep_sid = start["session_id"]

    with pytest.raises(Exception, match="not found"):
        await svc.submit_position(deep_sid, "stolen input", owner_id="tenant-b")


async def test_cross_tenant_cannot_join(svc):
    start = await svc.start_debate("A's", owner_id="tenant-a")

    with pytest.raises(Exception, match="not found"):
        await svc.join_debate(start["debate_id"], owner_id="tenant-b")


async def test_unscoped_legacy_debate_is_local_only(svc):
    """Authenticated tenants cannot inherit access to legacy unscoped rows."""
    start = await svc.start_debate("legacy debate", owner_id=None)
    with pytest.raises(Exception, match="not found"):
        await svc.status(start["debate_id"], owner_id="tenant-x")

    status = await svc.status(start["debate_id"], owner_id=None)
    assert status["prompt"] == "legacy debate"


async def test_unscoped_caller_cannot_list_or_delete_tenant_debate(svc):
    """The legacy local surface cannot become a backdoor into tenant rows."""
    tenant = await svc.start_debate("tenant private", owner_id="tenant-a")
    local = await svc.start_debate("local only", owner_id=None)

    local_history = await svc.history(owner_id=None)
    local_ids = {debate["id"] for debate in local_history["debates"]}
    assert local["debate_id"] in local_ids
    assert tenant["debate_id"] not in local_ids

    with pytest.raises(Exception, match="not found"):
        await svc.delete(tenant["debate_id"], owner_id=None)


async def test_owner_recovered_after_restart(tmp_path):
    db = tmp_path / "recover.db"
    svc = DebateService(store=DebateStore(db_path=db))
    await svc.initialize()
    start = await svc.start_debate("persist me", owner_id="tenant-a")
    debate_id = start["debate_id"]
    await svc.shutdown()

    svc2 = DebateService(store=DebateStore(db_path=db))
    await svc2.initialize()
    try:
        # Other tenant still gets not-found
        with pytest.raises(Exception, match="not found"):
            await svc2.status(debate_id, owner_id="tenant-b")
        # Original owner sees it
        status = await svc2.status(debate_id, owner_id="tenant-a")
        assert status["phase"] == "independent"
    finally:
        await svc2.shutdown()


async def test_server_tool_extracts_owner_from_auth(monkeypatch):
    """debate_start reads the owner id from the MCP auth context."""
    # Reset module state so _init() builds a fresh service.
    if server._service is not None:
        await server._service.shutdown()
    server._service = None

    # Arrange: pretend we have a token map and an active access token.
    monkeypatch.setattr(server, "_TOKEN_MAP", {"tok-a": "tenant-a"})

    class _FakeAccessToken:
        client_id = "tenant-a"

    monkeypatch.setattr(server, "get_access_token", lambda: _FakeAccessToken())

    result = await server.debate_start(prompt="hello")
    debate_id = result["debate_id"]

    # The service must have recorded the resolved tenant as the owner.
    assert server._service.debate_owners[debate_id] == "tenant-a"

    await server._service.shutdown()
    server._service = None


async def test_server_tool_extracts_oauth_owner_without_static_tokens(monkeypatch):
    """OAuth-only tool calls persist ``AccessToken.client_id`` as owner_id."""
    if server._service is not None:
        await server._service.shutdown()
    server._service = None

    monkeypatch.setattr(server, "_AUTH_MODE", "oauth")
    monkeypatch.setattr(server, "_TOKEN_MAP", {})

    class _FakeAccessToken:
        client_id = "oauth-tenant"

    monkeypatch.setattr(server, "get_access_token", lambda: _FakeAccessToken())

    result = await server.debate_start(prompt="oauth-owned")
    debate_id = result["debate_id"]
    assert server._service.debate_owners[debate_id] == "oauth-tenant"

    await server._service.shutdown()
    server._service = None
