"""Tests for the progress-callback plumbing and SSE framing.

The service-level progress callback is unit-testable in isolation — just
collect every event emitted during a ``run_auto`` call and assert on the
ordered type sequence. The HTTP SSE route is exercised via the HTTP e2e
infrastructure in a separate slow test; this file covers the in-process
contract.
"""

import asyncio

import pytest

import ploidy.api_client as api_client
from ploidy import server
from ploidy.stream import ProgressEvent, sse_format


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
        return f"{own_role} CHALLENGE"

    monkeypatch.setattr(api_client, "generate_experienced_position", fake_deep)
    monkeypatch.setattr(api_client, "generate_fresh_position", fake_fresh)
    monkeypatch.setattr(api_client, "generate_challenge", fake_challenge)


class TestSSEFraming:
    def test_sse_format_includes_event_type_and_data(self):
        ev = ProgressEvent(type="phase_started", data={"phase": "position"})
        frame = sse_format(ev)
        assert frame.startswith("event: phase_started\n")
        assert 'data: {"type": "phase_started"' in frame
        assert frame.endswith("\n\n")

    def test_sse_format_handles_unicode_payloads(self):
        ev = ProgressEvent(type="completed", data={"note": "수렴 완료"})
        frame = sse_format(ev)
        assert "수렴 완료" in frame


class TestProgressCallback:
    async def test_run_auto_emits_phase_sequence(self, _fake_api):
        events: list[ProgressEvent] = []

        async def collect(event):
            events.append(event)

        svc = await server._init()
        await svc.run_auto(prompt="test", context_documents=["project context"], progress=collect)

        types = [e.type for e in events]
        assert types[0] == "phase_started"
        assert types[-1] == "completed"
        assert "phase_started" in types
        assert "positions_generated" in types
        assert "challenges_generated" in types
        phase_starts = [e for e in events if e.type == "phase_started"]
        ordered_phases = [e.data["phase"] for e in phase_starts]
        assert ordered_phases == ["position", "challenge", "convergence"]

    async def test_callback_exceptions_do_not_break_the_debate(self, _fake_api):
        """Progress listeners are best-effort; a broken one must not crash run_auto."""
        call_count = 0

        async def broken(event):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("client disconnected")

        svc = await server._init()
        result = await svc.run_auto(
            prompt="test", context_documents=["project context"], progress=broken
        )
        assert result["phase"] == "complete"
        # The callback was still called despite each invocation raising.
        assert call_count >= 1

    async def test_positions_generated_payload_carries_previews(self, _fake_api):
        events: list[ProgressEvent] = []

        async def collect(event):
            events.append(event)

        svc = await server._init()
        await svc.run_auto(
            prompt="test",
            context_documents=["project context"],
            deep_n=2,
            fresh_n=1,
            progress=collect,
        )

        deep_events = [
            e for e in events if e.type == "positions_generated" and e.data.get("side") == "deep"
        ]
        assert len(deep_events) == 1
        assert deep_events[0].data["count"] == 2
        assert len(deep_events[0].data["previews"]) == 2

    async def test_run_auto_without_callback_still_works(self, _fake_api):
        """Passing progress=None is the default path — no regression."""
        svc = await server._init()
        result = await svc.run_auto(prompt="no listener", context_documents=["project context"])
        assert result["phase"] == "complete"


class TestClientDisconnect:
    """If the consumer drops the SSE connection, the worker task must not leak."""

    async def test_cancelling_consumer_cancels_worker(self, _fake_api):
        # Simulate the SSE endpoint's queue+worker pattern.
        queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()

        async def on_progress(event):
            await queue.put(event)

        svc = await server._init()

        async def run():
            try:
                await svc.run_auto(
                    prompt="test",
                    context_documents=["project context"],
                    progress=on_progress,
                )
            finally:
                await queue.put(None)

        worker = asyncio.create_task(run())
        # Consumer reads one frame then "disconnects".
        first = await queue.get()
        assert first is not None
        worker.cancel()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await worker

    async def test_sse_stream_close_waits_for_service_cleanup(self, monkeypatch):
        """Closing the response iterator returns only after run_auto is cleaned."""
        deep_started = asyncio.Event()

        async def blocked_deep(prompt, context_documents=None, effort="high", model=None):
            deep_started.set()
            await asyncio.Future()

        async def fake_fresh(prompt, effort="high", model=None):
            return "fresh pos"

        async def fake_challenge(**kwargs):
            return "CHALLENGE: unreachable"

        monkeypatch.setattr(api_client, "is_api_available", lambda: True)
        monkeypatch.setattr(api_client, "generate_experienced_position", blocked_deep)
        monkeypatch.setattr(api_client, "generate_fresh_position", fake_fresh)
        monkeypatch.setattr(api_client, "generate_challenge", fake_challenge)
        monkeypatch.setattr(server, "_AUTH_MODE", "bearer")
        monkeypatch.setattr(server, "_TOKEN_MAP", {})

        class Request:
            scope: dict = {}

            async def json(self):
                return {
                    "prompt": "disconnect",
                    "context_documents": ["deep-only project context"],
                }

        svc = await server._init()
        response = await server._stream_debate(Request())
        stream = response.body_iterator
        first_frame = await anext(stream)
        assert "phase_started" in first_frame
        await asyncio.wait_for(deep_started.wait(), timeout=1)

        await stream.aclose()

        assert (await svc.history())["debates"] == []
        assert svc.protocols == {}
        assert svc.sessions == {}
        assert svc.debate_sessions == {}
        assert svc.session_to_debate == {}
