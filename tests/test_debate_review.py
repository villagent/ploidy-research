"""HITL (human-in-the-loop) tests for debate_review.

Verifies that debate_auto + pause_at correctly pauses state, persists the
paused context, and that debate_review's three actions — approve, override,
reject — resume the debate as expected.
"""

import pytest

import ploidy.api_client as api_client
from ploidy import server


@pytest.fixture(autouse=True)
async def _reset_state():
    if server._service is not None:
        await server._service.shutdown()
    server._service = None
    yield
    if server._service is not None:
        await server._service.shutdown()
    server._service = None


@pytest.fixture
def _fake_api(monkeypatch):
    """Stub the whole api_client so debate_auto runs without a real LLM."""
    monkeypatch.setattr(api_client, "is_api_available", lambda: True)

    async def fake_deep(prompt, context_documents=None, effort="high", model=None):
        return "deep position v1"

    async def fake_fresh(prompt, effort="high", model=None):
        return "fresh position v1"

    async def fake_challenge(
        own_position,
        other_position,
        own_role="fresh",
        other_role="deep",
        effort="high",
        model=None,
    ):
        return f"{own_role} CHALLENGE vs {other_role}: reasons..."

    monkeypatch.setattr(api_client, "generate_experienced_position", fake_deep)
    monkeypatch.setattr(api_client, "generate_fresh_position", fake_fresh)
    monkeypatch.setattr(api_client, "generate_challenge", fake_challenge)


async def test_pause_at_challenge_persists_context(_fake_api):
    """pause_at='challenge' stops after positions and persists paused state."""
    result = await server.debate_auto(
        prompt="Should we rewrite in Rust?",
        context_documents=["project context"],
        pause_at="challenge",
    )

    assert result["phase"] == "paused"
    assert result["paused_before"] == "challenge"
    assert "positions" in result

    debate_id = result["debate_id"]
    assert debate_id in server._service.paused_debates

    # Paused context must also live in the DB so a restart can recover it.
    loaded = await server._service.store.load_paused_context(debate_id)
    assert loaded is not None
    assert loaded["paused_phase"] == "challenge"


async def test_approve_resumes_and_completes(_fake_api):
    """approve runs the challenge + convergence phases to completion."""
    paused = await server.debate_auto(
        prompt="Approve flow", context_documents=["project context"], pause_at="challenge"
    )
    debate_id = paused["debate_id"]

    result = await server.debate_review(debate_id, action="approve")

    assert result["phase"] == "complete"
    assert result["mode"] == "auto_hitl"
    assert result["reviewer_action"] == "approve"
    assert debate_id not in server._service.paused_debates

    history = await server.debate_history()
    assert any(d["id"] == debate_id and d["status"] == "complete" for d in history["debates"])


async def test_override_at_challenge_replaces_fresh_position(_fake_api):
    """override replaces the fresh position before challenges run."""
    paused = await server.debate_auto(
        prompt="Override flow", context_documents=["project context"], pause_at="challenge"
    )
    debate_id = paused["debate_id"]

    result = await server.debate_review(
        debate_id,
        action="override",
        override_content="reviewer-supplied fresh position",
    )

    assert result["phase"] == "complete"
    # Verify the override landed in the DB
    messages = await server._service.store.get_messages(debate_id)
    position_msgs = [m for m in messages if m["phase"] == "position"]
    contents = {m["content"] for m in position_msgs}
    assert "reviewer-supplied fresh position" in contents


async def test_override_requires_content(_fake_api):
    paused = await server.debate_auto(
        prompt="Missing content", context_documents=["project context"], pause_at="challenge"
    )
    debate_id = paused["debate_id"]

    with pytest.raises(Exception, match="override_content is required"):
        await server.debate_review(debate_id, action="override")


async def test_reject_cancels_debate(_fake_api):
    paused = await server.debate_auto(
        prompt="Reject flow", context_documents=["project context"], pause_at="challenge"
    )
    debate_id = paused["debate_id"]

    result = await server.debate_review(debate_id, action="reject")

    assert result["phase"] == "cancelled"
    assert debate_id not in server._service.paused_debates
    assert debate_id not in server._service.protocols


async def test_review_unknown_debate_raises(_fake_api):
    with pytest.raises(Exception, match="not paused"):
        await server.debate_review("does-not-exist", action="approve")


async def test_review_rejects_invalid_action(_fake_api):
    paused = await server.debate_auto(
        prompt="Bad action", context_documents=["project context"], pause_at="challenge"
    )
    with pytest.raises(Exception, match="Invalid action"):
        await server.debate_review(paused["debate_id"], action="maybe")


async def test_pause_at_convergence_then_approve(_fake_api):
    """pause_at='convergence' captures challenges; approve should converge."""
    paused = await server.debate_auto(
        prompt="Pause before convergence",
        context_documents=["project context"],
        pause_at="convergence",
    )
    debate_id = paused["debate_id"]
    assert paused["paused_before"] == "convergence"
    assert "challenges" in paused

    result = await server.debate_review(debate_id, action="approve")
    assert result["phase"] == "complete"
    assert debate_id not in server._service.paused_debates


async def test_paused_state_recovers_after_shutdown(_fake_api):
    """A paused debate survives a service restart and stays reviewable."""
    paused = await server.debate_auto(
        prompt="Recover after restart",
        context_documents=["project context"],
        pause_at="challenge",
    )
    debate_id = paused["debate_id"]

    await server.shutdown()
    await server._init()

    assert debate_id in server._service.paused_debates

    result = await server.debate_review(debate_id, action="approve")
    assert result["phase"] == "complete"
