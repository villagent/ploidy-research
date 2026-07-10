"""Tests for the unified v0.4 ``debate`` tool.

The tool dispatches to ``DebateService.run_solo`` / ``run_auto`` based on
``mode``. These tests verify dispatch, validation, and parity with the
legacy tools' behaviour.
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


async def test_solo_dispatch_completes_with_convergence():
    """mode='solo' runs the full flow end to end, no API needed."""
    result = await server.debate(
        prompt="monorepo vs polyrepo",
        mode="solo",
        deep_position="Monorepo — shared libs, atomic refactors.",
        fresh_position="Polyrepo — independent deploys.",
        deep_challenge="CHALLENGE: polyrepo ignores shared-auth duplication.",
        fresh_challenge="CHALLENGE: monorepo couples release cadence.",
    )
    assert result["mode"] == "solo"
    assert result["phase"] == "complete"
    assert isinstance(result["confidence"], float)
    assert result["synthesis"]


async def test_solo_requires_both_positions():
    with pytest.raises(ValueError, match="requires a non-empty fresh_position"):
        await server.debate(prompt="x", mode="solo", deep_position="only one")


async def test_solo_rejects_blank_position():
    """Whitespace-only positions should be caught at the tool boundary."""
    with pytest.raises(ValueError, match="requires a non-empty deep_position"):
        await server.debate(
            prompt="x",
            mode="solo",
            deep_position="   \n\t  ",
            fresh_position="actual content",
        )


async def test_auto_rejects_solo_only_params():
    """Catching a common paste mistake — deep_position in auto mode."""
    with pytest.raises(ValueError, match="does not accept deep_position"):
        await server.debate(prompt="x", mode="auto", deep_position="stray input")


async def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Invalid mode"):
        await server.debate(prompt="x", mode="three_way")


async def test_auto_dispatch_invokes_api_client(monkeypatch):
    """mode='auto' should drive the api_client path exactly like debate_auto."""
    monkeypatch.setattr(api_client, "is_api_available", lambda: True)

    async def fake_deep(prompt, context_documents=None, effort="high", model=None):
        return "deep pos"

    async def fake_fresh(prompt, effort="high", model=None):
        return "fresh pos"

    async def fake_challenge(
        own_position,
        other_position,
        own_role="fresh",
        other_role="deep",
        effort="high",
        model=None,
    ):
        return f"{own_role} CHALLENGE vs {other_role}"

    monkeypatch.setattr(api_client, "generate_experienced_position", fake_deep)
    monkeypatch.setattr(api_client, "generate_fresh_position", fake_fresh)
    monkeypatch.setattr(api_client, "generate_challenge", fake_challenge)

    result = await server.debate(
        prompt="Auto flow", mode="auto", context_documents=["project context"]
    )

    assert result["mode"] == "auto"
    assert result["phase"] == "complete"
    assert result["synthesis"]


async def test_auto_pause_at_returns_paused_state(monkeypatch):
    """HITL pause_at still works through the unified tool."""
    monkeypatch.setattr(api_client, "is_api_available", lambda: True)

    async def fake_deep(prompt, context_documents=None, effort="high", model=None):
        return "deep"

    async def fake_fresh(prompt, effort="high", model=None):
        return "fresh"

    monkeypatch.setattr(api_client, "generate_experienced_position", fake_deep)
    monkeypatch.setattr(api_client, "generate_fresh_position", fake_fresh)

    result = await server.debate(
        prompt="paused",
        mode="auto",
        context_documents=["project context"],
        pause_at="challenge",
    )
    assert result["phase"] == "paused"
    assert result["paused_before"] == "challenge"


async def test_solo_result_equivalent_to_legacy_tool():
    """The solo result shape should match what debate_solo used to return."""
    args = dict(
        prompt="same prompt",
        deep_position="A",
        fresh_position="B",
        deep_challenge="CHALLENGE: a",
        fresh_challenge="CHALLENGE: b",
    )
    unified = await server.debate(mode="solo", **args)
    # Fresh service instance for the legacy call so they don't collide.
    await server._service.shutdown()
    server._service = None
    legacy = await server.debate_solo(**args)

    assert unified.keys() == legacy.keys()
    assert unified["mode"] == legacy["mode"] == "solo"
    assert unified["phase"] == legacy["phase"] == "complete"
