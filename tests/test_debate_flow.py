"""End-to-end test for the full debate flow.

Simulates the two-terminal workflow:
1. Deep session starts a debate
2. Fresh session joins
3. Both submit positions
4. Both submit challenges
5. Convergence produces a structured result
"""

import pytest

import ploidy.api_client as api_client
from ploidy import server


@pytest.fixture(autouse=True)
async def _reset_state():
    """Drop the shared DebateService between tests."""
    if server._service is not None:
        await server._service.shutdown()
    server._service = None
    yield
    if server._service is not None:
        await server._service.shutdown()
    server._service = None


async def test_full_debate_flow():
    """Complete debate lifecycle: start → join → positions → challenges → converge."""
    # 1. Deep session starts debate
    start = await server.debate_start(
        prompt="Should we use monorepo or polyrepo?",
        context_documents=["We have 3 teams, 12 microservices"],
    )
    assert start["role"] == "deep"
    assert start["phase"] == "independent"
    debate_id = start["debate_id"]
    exp_id = start["session_id"]

    # 2. Fresh session joins
    join = await server.debate_join(debate_id)
    assert join["role"] == "fresh"
    assert join["prompt"] == "Should we use monorepo or polyrepo?"
    fresh_id = join["session_id"]

    # 3. Both submit positions
    pos_exp = await server.debate_position(
        session_id=exp_id,
        content="Monorepo. We have shared libs and cross-team dependencies.",
    )
    assert pos_exp["status"] == "recorded"
    assert pos_exp["all_positions_in"] is False
    assert pos_exp["phase"] == "position"

    pos_fresh = await server.debate_position(
        session_id=fresh_id,
        content="Polyrepo. Each service should be independently deployable.",
    )
    assert pos_fresh["status"] == "recorded"
    assert pos_fresh["all_positions_in"] is True
    assert pos_fresh["phase"] == "challenge"  # auto-advanced

    # 4. Check status — both positions visible
    status = await server.debate_status(debate_id)
    assert status["phase"] == "challenge"
    assert status["message_count"] == 2
    assert "position" in status["messages"]
    assert len(status["messages"]["position"]) == 2

    # 5. Both submit challenges
    ch_exp = await server.debate_challenge(
        session_id=exp_id,
        content="Polyrepo ignores our shared auth library. Duplication across 12 repos.",
        action="challenge",
    )
    assert ch_exp["status"] == "recorded"

    ch_fresh = await server.debate_challenge(
        session_id=fresh_id,
        content="Monorepo creates coupling. Independent deployment is more valuable.",
        action="challenge",
    )
    assert ch_fresh["status"] == "recorded"

    # 6. Converge
    result = await server.debate_converge(debate_id)
    assert result["phase"] == "complete"
    assert result["confidence"] is not None
    assert isinstance(result["points"], list)
    assert len(result["points"]) == 2
    assert result["synthesis"]  # non-empty


async def test_start_and_status():
    """Status shows correct info after start."""
    start = await server.debate_start(prompt="Test prompt")
    status = await server.debate_status(start["debate_id"])
    assert status["prompt"] == "Test prompt"
    assert status["phase"] == "independent"
    assert len(status["sessions"]) == 1


async def test_position_before_join_is_rejected_without_freezing_roster():
    """A position cannot freeze the roster before a Fresh session joins."""
    start = await server.debate_start(prompt="Test")
    exp_id = start["session_id"]

    with pytest.raises(Exception, match="At least two sessions"):
        await server.debate_position(exp_id, "My position")

    join = await server.debate_join(start["debate_id"])
    assert join["phase"] == "independent"


async def test_join_is_rejected_after_position_submission_begins():
    """The participant roster stays frozen once any position is recorded."""
    start = await server.debate_start(prompt="Roster freeze")
    await server.debate_join(start["debate_id"])
    await server.debate_position(start["session_id"], "Deep position")

    with pytest.raises(Exception, match="participant roster is frozen"):
        await server.debate_join(start["debate_id"])


async def test_each_session_can_submit_only_one_position():
    """A seat cannot revise its sealed position before the barrier opens."""
    start = await server.debate_start(prompt="One position per seat")
    await server.debate_join(start["debate_id"])
    await server.debate_position(start["session_id"], "first position")

    with pytest.raises(Exception, match="already submitted a position"):
        await server.debate_position(start["session_id"], "revised position")

    status = await server.debate_status(start["debate_id"])
    assert status["message_count"] == 1
    assert status["positions_released"] is False


async def test_challenge_in_wrong_phase_fails():
    """Cannot submit challenge during position phase."""
    start = await server.debate_start(prompt="Test")
    exp_id = start["session_id"]

    with pytest.raises(Exception, match="Cannot submit challenge"):
        await server.debate_challenge(exp_id, "Challenge!", "challenge")


async def test_converge_in_wrong_phase_fails():
    """Cannot converge before challenge phase."""
    start = await server.debate_start(prompt="Test")

    with pytest.raises(Exception, match="Cannot converge"):
        await server.debate_converge(start["debate_id"])


async def test_converge_requires_one_challenge_from_every_session():
    """The challenge barrier cannot complete with a missing or duplicate seat."""
    start = await server.debate_start(prompt="Challenge barrier")
    joined = await server.debate_join(start["debate_id"])
    await server.debate_position(start["session_id"], "deep position")
    await server.debate_position(joined["session_id"], "fresh position")
    await server.debate_challenge(start["session_id"], "deep challenge")

    with pytest.raises(Exception, match="every participating session"):
        await server.debate_converge(start["debate_id"])
    with pytest.raises(Exception, match="already submitted a challenge"):
        await server.debate_challenge(start["session_id"], "duplicate deep challenge")

    await server.debate_challenge(joined["session_id"], "fresh challenge")
    result = await server.debate_converge(start["debate_id"])
    assert result["phase"] == "complete"


async def test_agree_action_increases_confidence():
    """Agree actions produce higher confidence than challenges."""
    start = await server.debate_start(prompt="Agree test")
    debate_id = start["debate_id"]
    exp_id = start["session_id"]

    join = await server.debate_join(debate_id)
    fresh_id = join["session_id"]

    await server.debate_position(exp_id, "Position A")
    await server.debate_position(fresh_id, "Position B")

    await server.debate_challenge(exp_id, "I agree with Fresh", "agree")
    await server.debate_challenge(fresh_id, "I agree with Experienced", "agree")

    result = await server.debate_converge(debate_id)
    assert result["confidence"] == 1.0


async def test_cancel_debate():
    """Cancel removes debate from active state."""
    start = await server.debate_start(prompt="Cancel test")
    debate_id = start["debate_id"]

    result = await server.debate_cancel(debate_id)
    assert result["status"] == "cancelled"

    with pytest.raises(Exception, match="not found"):
        await server.debate_status(debate_id)


async def test_delete_debate():
    """Delete removes debate from both memory and database."""
    start = await server.debate_start(prompt="Delete test")
    debate_id = start["debate_id"]

    result = await server.debate_delete(debate_id)
    assert result["status"] == "deleted"

    history = await server.debate_history()
    assert not any(d["id"] == debate_id for d in history["debates"])


async def test_input_validation():
    """Oversized inputs are rejected."""
    with pytest.raises(Exception, match="exceeds maximum length"):
        await server.debate_start(prompt="x" * 20000)


async def test_session_cap():
    """Cannot exceed max sessions per debate."""
    start = await server.debate_start(prompt="Cap test")
    debate_id = start["debate_id"]

    for _ in range(4):
        await server.debate_join(debate_id)

    with pytest.raises(Exception, match="max"):
        await server.debate_join(debate_id)


async def test_persistence():
    """Debates are persisted and appear in history."""
    start = await server.debate_start(prompt="Persist test")
    debate_id = start["debate_id"]
    exp_id = start["session_id"]

    join = await server.debate_join(debate_id)
    fresh_id = join["session_id"]

    await server.debate_position(exp_id, "A")
    await server.debate_position(fresh_id, "B")
    await server.debate_challenge(exp_id, "Challenge A", "challenge")
    await server.debate_challenge(fresh_id, "Challenge B", "challenge")
    await server.debate_converge(debate_id)

    history = await server.debate_history()
    assert history["total"] >= 1
    assert any(d["id"] == debate_id for d in history["debates"])


async def test_cancelled_debate_is_not_recovered():
    """Cancelled debates stay cancelled across restart."""
    start = await server.debate_start(prompt="Recovery test")
    debate_id = start["debate_id"]

    await server.debate_cancel(debate_id)
    await server.shutdown()

    await server._init()
    history = await server.debate_history()

    assert debate_id not in server._service.protocols
    assert any(d["id"] == debate_id and d["status"] == "cancelled" for d in history["debates"])


async def test_session_context_is_recovered():
    """Recovered sessions preserve stored context metadata."""
    start = await server.debate_start(
        prompt="Context recovery",
        context_documents=["doc-a", "doc-b"],
    )
    debate_id = start["debate_id"]

    join = await server.debate_join(debate_id, role="semi_fresh", delivery_mode="active")
    semi_fresh_id = join["session_id"]
    server._service.sessions[semi_fresh_id].compressed_summary = "compressed"
    await server._service.store.update_session_context(
        semi_fresh_id,
        context_documents=[],
        delivery_mode="active",
        compressed_summary="compressed",
        metadata={"source": "test"},
    )

    await server.shutdown()
    await server._init()

    sessions = {
        sid: ctx for sid, ctx in server._service.sessions.items() if sid.startswith(debate_id)
    }
    exp = next(ctx for ctx in sessions.values() if ctx.role.value == "deep")
    sf = next(ctx for ctx in sessions.values() if ctx.role.value == "semi_fresh")

    assert exp.context_documents == ["doc-a", "doc-b"]
    assert sf.delivery_mode.value == "active"
    assert sf.compressed_summary == "compressed"
    assert sf.metadata == {"source": "test"}


async def test_recovered_session_preserves_effort_and_model():
    """Effort and model fields survive a server restart cycle.

    Regression: get_sessions() previously omitted these columns from
    its SELECT, so recovery silently dropped them.
    """
    svc = await server._init()
    store = svc.store
    debate_id = "rec-em-001"
    await store.save_debate(debate_id, "Recovery prompt")

    sid = f"{debate_id}-deep-aaa"
    await store.save_session(
        sid,
        debate_id,
        "deep",
        "Recovery prompt",
        delivery_mode="passive",
        model="gpt-test-mini",
        effort="max",
    )

    await server.shutdown()
    await server._init()

    ctx = server._service.sessions.get(sid)
    assert ctx is not None, "session not recovered"
    assert ctx.model == "gpt-test-mini"
    assert ctx.effort.value == "max"


async def test_debate_solo_runs_without_api():
    """debate_solo accepts caller-supplied positions and converges in one call."""
    result = await server.debate_solo(
        prompt="Should we adopt async I/O for the request handlers?",
        deep_position=(
            "Async will pay off because we've already taken on aiosqlite and httpx. "
            "Migrating the handlers consolidates the loop ownership."
        ),
        fresh_position=(
            "Threads are simpler and our throughput target is modest. "
            "Async adds debugging overhead with no measured benefit."
        ),
        deep_challenge="CHALLENGE: simplicity is rationalization given our existing async deps.",
        fresh_challenge="CHALLENGE: deep session is over-fitting to historical choices.",
        context_documents=["aiosqlite + httpx already in dependencies"],
    )

    assert result["phase"] == "complete"
    assert result["mode"] == "solo"
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["points"], list)
    assert len(result["points"]) >= 1
    assert result["synthesis"]
    debate_id = result["debate_id"]
    # _cleanup_debate must drop every map it touches; missing one would
    # leak server state across runs
    assert debate_id not in server._service.protocols
    assert debate_id not in server._service.debate_sessions
    assert not any(sid.startswith(debate_id) for sid in server._service.sessions)
    assert not any(sid.startswith(debate_id) for sid in server._service.session_to_debate)

    history = await server.debate_history()
    assert any(d["id"] == debate_id and d["status"] == "complete" for d in history["debates"])


async def test_debate_solo_without_challenges():
    """debate_solo works when no challenges are supplied (positions only)."""
    result = await server.debate_solo(
        prompt="Test",
        deep_position="A",
        fresh_position="B",
    )
    assert result["phase"] == "complete"
    # No challenges → convergence engine emits an informational
    # no_challenges marker (not a disagreement).
    assert any(p["category"] == "no_challenges" for p in result["points"])


async def test_debate_solo_validates_input_lengths():
    """Oversized positions are rejected."""
    with pytest.raises(Exception, match="exceeds maximum length"):
        await server.debate_solo(
            prompt="Test",
            deep_position="x" * 100000,
            fresh_position="ok",
        )


async def test_debate_auto_generates_both_sides(monkeypatch):
    """Auto debate should generate positions and challenges for both sessions."""
    monkeypatch.setattr(api_client, "is_api_available", lambda: True)

    async def fake_exp(prompt, context_documents=None, effort="high", model=None):
        # With injection_mode="raw", context is embedded in prompt
        assert "Auto prompt" in prompt
        return "deep position"

    async def fake_fresh(prompt, effort="high", model=None):
        assert "Auto prompt" in prompt
        return "fresh position"

    async def fake_challenge(
        own_position,
        other_position,
        own_role="fresh",
        other_role="deep",
        effort="high",
        model=None,
    ):
        return f"{own_role} CHALLENGE vs {other_role}: {own_position} / {other_position}"

    monkeypatch.setattr(api_client, "generate_experienced_position", fake_exp)
    monkeypatch.setattr(api_client, "generate_fresh_position", fake_fresh)
    monkeypatch.setattr(api_client, "generate_challenge", fake_challenge)

    result = await server.debate_auto(prompt="Auto prompt", context_documents=["ctx"])

    assert result["phase"] == "complete"
    assert "config" in result

    history = await server.debate_history()
    debate_id = result["debate_id"]
    assert any(d["id"] == debate_id and d["status"] == "complete" for d in history["debates"])

    messages = await server._service.store.get_messages(debate_id)
    position_messages = [m for m in messages if m["phase"] == "position"]
    challenge_messages = [m for m in messages if m["phase"] == "challenge"]

    assert len(position_messages) == 2
    assert len(challenge_messages) == 2
    assert {m["content"] for m in position_messages} == {"deep position", "fresh position"}


async def test_debate_auto_cleans_up_failed_runs(monkeypatch):
    """Failed auto debates should not leave partial active records behind."""
    monkeypatch.setattr(api_client, "is_api_available", lambda: True)

    async def fake_exp(prompt, context_documents=None, effort="high", model=None):
        return "deep position"

    async def fake_fresh(prompt, effort="high", model=None):
        raise RuntimeError("boom")

    async def fake_challenge(*args, **kwargs):
        return "challenge"

    monkeypatch.setattr(api_client, "generate_experienced_position", fake_exp)
    monkeypatch.setattr(api_client, "generate_fresh_position", fake_fresh)
    monkeypatch.setattr(api_client, "generate_challenge", fake_challenge)

    with pytest.raises(RuntimeError, match="boom"):
        await server.debate_auto(prompt="Auto prompt", context_documents=["project context"])

    history = await server.debate_history()
    assert history["debates"] == []
