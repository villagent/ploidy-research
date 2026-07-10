"""Regression tests for context-asymmetric service invariants."""

from __future__ import annotations

import asyncio
import json

import pytest

import ploidy.api_client as api_client
from ploidy import server
from ploidy.exceptions import PloidyError, ProtocolError


@pytest.fixture(autouse=True)
async def _reset_state():
    if server._service is not None:
        await server._service.shutdown()
    server._service = None
    yield
    if server._service is not None:
        await server._service.shutdown()
    server._service = None


def _stub_auto_api(monkeypatch, *, deep, fresh, challenge) -> None:
    monkeypatch.setattr(api_client, "is_api_available", lambda: True)
    monkeypatch.setattr(api_client, "generate_experienced_position", deep)
    monkeypatch.setattr(api_client, "generate_fresh_position", fresh)
    monkeypatch.setattr(api_client, "generate_challenge", challenge)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({}, "requires non-empty context_documents"),
        ({"context_documents": ["  \n"]}, "requires non-empty context_documents"),
        (
            {"context_documents": ["actual context"], "context_pct": 0},
            "context_pct must be 1..100",
        ),
        (
            {"context_documents": ["x"], "context_pct": 1},
            "truncates the supplied context to empty",
        ),
    ],
)
async def test_auto_fails_closed_without_actual_deep_context(kwargs, error):
    """Role labels and zero-percent context cannot masquerade as Ploidy."""
    svc = await server._init()

    with pytest.raises(ProtocolError, match=error):
        await svc.run_auto(prompt="Review this decision", **kwargs)

    assert (await svc.history())["debates"] == []


async def test_auto_rejects_different_models_before_creating_state():
    """Auto mode may vary context depth, never the underlying model."""
    svc = await server._init()

    with pytest.raises(ProtocolError, match="must match"):
        await svc.run_auto(
            prompt="model isolation",
            context_documents=["deep-only context"],
            deep_model="model-a",
            fresh_model="model-b",
        )

    assert (await svc.history())["debates"] == []


async def test_auto_applies_one_model_override_to_every_seat_and_phase(monkeypatch):
    """A single override resolves to one model for both sides and challenges."""
    seen_models: list[str | None] = []

    async def fake_deep(prompt, context_documents=None, effort="high", model=None):
        seen_models.append(model)
        return "deep position"

    async def fake_fresh(prompt, effort="high", model=None):
        seen_models.append(model)
        return "fresh position"

    async def fake_challenge(**kwargs):
        seen_models.append(kwargs["model"])
        return "CHALLENGE: independent"

    _stub_auto_api(
        monkeypatch,
        deep=fake_deep,
        fresh=fake_fresh,
        challenge=fake_challenge,
    )
    svc = await server._init()
    result = await svc.run_auto(
        prompt="same model",
        context_documents=["deep-only context"],
        deep_model="one-model",
    )

    assert seen_models == ["one-model"] * 4
    assert result["config"]["deep_model"] == "one-model"
    assert result["config"]["fresh_model"] == "one-model"


async def test_auto_delivers_system_context_once_and_raw_context_once(monkeypatch):
    """Injection mode selects one real channel without duplicating raw context."""
    deep_calls: list[dict] = []

    async def fake_deep(
        prompt,
        context_documents=None,
        effort="high",
        model=None,
        system_prompt=None,
    ):
        deep_calls.append(
            {
                "prompt": prompt,
                "context_documents": context_documents,
                "system_prompt": system_prompt,
            }
        )
        return "deep position"

    async def fake_fresh(prompt, effort="high", model=None):
        return "fresh position"

    async def fake_challenge(**kwargs):
        return f"CHALLENGE: {kwargs['own_role']} independent response"

    _stub_auto_api(
        monkeypatch,
        deep=fake_deep,
        fresh=fake_fresh,
        challenge=fake_challenge,
    )
    svc = await server._init()

    raw_context = "RAW_CONTEXT_SENTINEL"
    await svc.run_auto(
        prompt="raw question",
        context_documents=[raw_context],
        injection_mode="raw",
    )
    system_context = "SYSTEM_CONTEXT_SENTINEL"
    await svc.run_auto(
        prompt="system question",
        context_documents=[system_context],
        injection_mode="system_prompt",
    )

    raw_call, system_call = deep_calls
    assert raw_call["prompt"].count(raw_context) == 1
    assert raw_call["context_documents"] is None
    assert raw_call["system_prompt"] is None
    assert system_context not in system_call["prompt"]
    assert system_context in system_call["system_prompt"]
    assert system_call["context_documents"] is None


async def test_manual_positions_stay_sealed_until_every_session_submits():
    """Status and history cannot leak the first manual position to a peer."""
    svc = await server._init()
    started = await svc.start_debate("Choose a datastore")
    joined = await svc.join_debate(started["debate_id"])

    secret_position = "DEEP_SECRET_POSITION"
    await svc.submit_position(started["session_id"], secret_position)

    waiting = await svc.status(started["debate_id"])
    history = await svc.history()
    assert waiting["positions_released"] is False
    assert "position" not in waiting["messages"]
    assert secret_position not in json.dumps(history)

    await svc.submit_position(joined["session_id"], "fresh independent position")
    released = await svc.status(started["debate_id"])
    assert released["positions_released"] is True
    assert {message["content"] for message in released["messages"]["position"]} == {
        secret_position,
        "fresh independent position",
    }


async def test_join_racing_first_position_cannot_reopen_frozen_roster(monkeypatch):
    """Join and the INDEPENDENT-to-POSITION transition share one lock."""
    svc = await server._init()
    started = await svc.start_debate("Freeze the roster")
    await svc.join_debate(started["debate_id"])

    save_started = asyncio.Event()
    release_save = asyncio.Event()
    original_save_message = svc.store.save_message

    async def blocking_save_message(*args, **kwargs):
        save_started.set()
        await release_save.wait()
        await original_save_message(*args, **kwargs)

    monkeypatch.setattr(svc.store, "save_message", blocking_save_message)
    position_task = asyncio.create_task(
        svc.submit_position(started["session_id"], "first position")
    )
    await asyncio.wait_for(save_started.wait(), timeout=1)
    late_join = asyncio.create_task(svc.join_debate(started["debate_id"]))
    await asyncio.sleep(0)
    assert late_join.done() is False

    release_save.set()
    await position_task
    with pytest.raises(ProtocolError, match="participant roster is frozen"):
        await late_join

    assert len(svc.debate_sessions[started["debate_id"]]) == 2


async def test_two_n_generates_and_persists_one_challenge_per_session(monkeypatch):
    """A 2n run makes four challenge calls and never clones a role response."""
    deep_counter = 0
    fresh_counter = 0
    challenge_calls: list[dict] = []

    async def fake_deep(prompt, context_documents=None, effort="high", model=None):
        nonlocal deep_counter
        deep_counter += 1
        return f"deep position {deep_counter}"

    async def fake_fresh(prompt, effort="high", model=None):
        nonlocal fresh_counter
        fresh_counter += 1
        return f"fresh position {fresh_counter}"

    async def fake_challenge(**kwargs):
        challenge_calls.append(kwargs)
        return f"CHALLENGE: independent call {len(challenge_calls)}"

    _stub_auto_api(
        monkeypatch,
        deep=fake_deep,
        fresh=fake_fresh,
        challenge=fake_challenge,
    )
    svc = await server._init()
    result = await svc.run_auto(
        prompt="2n decision",
        context_documents=["deep-only project context"],
        deep_n=2,
        fresh_n=2,
    )

    messages = await svc.store.get_messages(result["debate_id"])
    challenges = [message for message in messages if message["phase"] == "challenge"]
    assert len(challenge_calls) == 4
    assert [call["own_position"] for call in challenge_calls] == [
        "deep position 1",
        "deep position 2",
        "fresh position 1",
        "fresh position 2",
    ]
    assert len(challenges) == 4
    assert len({message["session_id"] for message in challenges}) == 4
    assert len({message["content"] for message in challenges}) == 4


async def test_two_n_hitl_resume_also_generates_one_challenge_per_session(monkeypatch):
    """Resuming before challenge preserves the same per-seat 2n contract."""
    challenge_calls: list[dict] = []

    async def fake_deep(prompt, context_documents=None, effort="high", model=None):
        return f"deep position {prompt[-1]}"

    async def fake_fresh(prompt, effort="high", model=None):
        return "fresh position"

    async def fake_challenge(**kwargs):
        challenge_calls.append(kwargs)
        return f"CHALLENGE: resumed call {len(challenge_calls)}"

    _stub_auto_api(
        monkeypatch,
        deep=fake_deep,
        fresh=fake_fresh,
        challenge=fake_challenge,
    )
    svc = await server._init()
    paused = await svc.run_auto(
        prompt="2n HITL",
        context_documents=["deep-only context"],
        deep_n=2,
        fresh_n=2,
        pause_at="challenge",
    )
    assert challenge_calls == []

    result = await svc.review(paused["debate_id"], action="approve")
    messages = await svc.store.get_messages(result["debate_id"])
    challenges = [message for message in messages if message["phase"] == "challenge"]
    assert len(challenge_calls) == 4
    assert len(challenges) == 4
    assert len({message["content"] for message in challenges}) == 4


async def test_cancelled_auto_run_removes_persistent_and_in_memory_state(monkeypatch):
    """SSE worker cancellation cannot leave a recoverable active debate row."""
    deep_started = asyncio.Event()

    async def blocked_deep(prompt, context_documents=None, effort="high", model=None):
        deep_started.set()
        await asyncio.Future()

    async def fake_fresh(prompt, effort="high", model=None):
        return "fresh position"

    async def fake_challenge(**kwargs):
        return "CHALLENGE: unreachable"

    _stub_auto_api(
        monkeypatch,
        deep=blocked_deep,
        fresh=fake_fresh,
        challenge=fake_challenge,
    )
    svc = await server._init()
    task = asyncio.create_task(
        svc.run_auto(
            prompt="cancel me",
            context_documents=["deep-only project context"],
        )
    )
    await asyncio.wait_for(deep_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (await svc.history())["debates"] == []
    assert svc.protocols == {}
    assert svc.sessions == {}
    assert svc.debate_sessions == {}
    assert svc.session_to_debate == {}
    assert svc.paused_debates == {}


async def test_cancelled_solo_run_removes_persistent_and_in_memory_state(monkeypatch):
    """Cancelling convergence in solo mode cannot leak its local roster."""
    from ploidy.convergence import ConvergenceEngine

    analyze_started = asyncio.Event()

    async def blocked_analyze(self, protocol, session_roles):
        analyze_started.set()
        await asyncio.Future()

    monkeypatch.setattr(ConvergenceEngine, "analyze", blocked_analyze)
    svc = await server._init()
    task = asyncio.create_task(
        svc.run_solo(
            prompt="cancel solo",
            deep_position="deep",
            fresh_position="fresh",
        )
    )
    await asyncio.wait_for(analyze_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (await svc.history())["debates"] == []
    assert svc.protocols == {}
    assert svc.sessions == {}
    assert svc.debate_sessions == {}
    assert svc.session_to_debate == {}
    assert svc.paused_debates == {}


async def test_authenticated_tenant_cannot_access_unscoped_debate():
    """Legacy unscoped rows stay local instead of becoming shared tenant data."""
    svc = await server._init()
    started = await svc.start_debate("legacy local debate", owner_id=None)

    with pytest.raises(PloidyError, match="not found"):
        await svc.status(started["debate_id"], owner_id="tenant-a")

    assert (await svc.status(started["debate_id"], owner_id=None))["debate_id"] == started[
        "debate_id"
    ]
