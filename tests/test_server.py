"""Tests for the MCP server module.

Covers the helper functions and custom ASGI routes that ``server.py``
exposes. The MCP tool decorators themselves (``@mcp.tool``) delegate
to ``DebateService``, which has its own dedicated test suite — so we
focus on the auth, route, and lifecycle glue that lives in this module.
"""

from __future__ import annotations

import json as _json
from unittest.mock import MagicMock


class TestLoadTokenMap:
    """``_load_token_map`` converts env strings into token→tenant dicts."""

    def _reload(self, monkeypatch, *, tokens: str | None = None, auth: str | None = None):
        # Module-level globals cache the env at import time; exercise the
        # function after clearing the caches so each scenario is fresh.
        from ploidy import server as srv

        monkeypatch.setattr(srv, "_TOKEN_MAP_RAW", tokens or "")
        monkeypatch.setattr(srv, "_AUTH_TOKEN", auth)
        return srv._load_token_map()

    def test_json_map_parsed_into_dict(self, monkeypatch):
        mapping = self._reload(
            monkeypatch,
            tokens=_json.dumps({"tok-a": "tenant-a", "tok-b": "tenant-b"}),
        )
        assert mapping == {"tok-a": "tenant-a", "tok-b": "tenant-b"}

    def test_malformed_json_is_logged_and_returns_empty(self, monkeypatch, caplog):
        import logging

        caplog.set_level(logging.ERROR, logger="ploidy")
        mapping = self._reload(monkeypatch, tokens="{not json")
        assert mapping == {}
        assert any("not valid JSON" in r.message for r in caplog.records)

    def test_non_object_json_is_rejected(self, monkeypatch, caplog):
        import logging

        caplog.set_level(logging.ERROR, logger="ploidy")
        mapping = self._reload(monkeypatch, tokens=_json.dumps(["a", "b"]))
        assert mapping == {}
        assert any("object of token" in r.message for r in caplog.records)

    def test_single_auth_token_falls_back_to_ploidy_client(self, monkeypatch):
        mapping = self._reload(monkeypatch, auth="legacy-token")
        assert mapping == {"legacy-token": "ploidy-client"}

    def test_neither_configured_returns_empty(self, monkeypatch):
        mapping = self._reload(monkeypatch)
        assert mapping == {}

    def test_values_coerced_to_strings(self, monkeypatch):
        # JSON allows any scalar in values; the server normalises to str
        # so tenant labels are always safe to pass into Prometheus.
        mapping = self._reload(monkeypatch, tokens=_json.dumps({"t": 42}))
        assert mapping == {"t": "42"}


class TestTokenVerifier:
    """``_PloidyTokenVerifier.verify_token`` drives the bearer auth path."""

    async def test_valid_token_returns_access_token_with_tenant(self, monkeypatch):
        from ploidy import server as srv

        monkeypatch.setattr(srv, "_TOKEN_MAP", {"tok-valid": "tenant-42"})
        verifier = srv._PloidyTokenVerifier()
        access = await verifier.verify_token("tok-valid")
        assert access is not None
        assert access.client_id == "tenant-42"
        assert "debate" in access.scopes

    async def test_invalid_token_returns_none(self, monkeypatch):
        from ploidy import server as srv

        monkeypatch.setattr(srv, "_TOKEN_MAP", {"tok-valid": "tenant-42"})
        verifier = srv._PloidyTokenVerifier()
        assert await verifier.verify_token("wrong") is None

    async def test_empty_token_map_always_returns_none(self, monkeypatch):
        from ploidy import server as srv

        monkeypatch.setattr(srv, "_TOKEN_MAP", {})
        verifier = srv._PloidyTokenVerifier()
        assert await verifier.verify_token("anything") is None

    async def test_verifier_runs_full_scan_for_constant_time(self, monkeypatch):
        """Matching is constant-time — every entry is compared even after a hit."""
        from ploidy import server as srv

        call_count = {"n": 0}
        real_compare = srv.hmac.compare_digest

        def counting_compare(a, b):
            call_count["n"] += 1
            return real_compare(a, b)

        monkeypatch.setattr(srv.hmac, "compare_digest", counting_compare)
        monkeypatch.setattr(
            srv,
            "_TOKEN_MAP",
            {"a": "ta", "b": "tb", "c": "tc"},
        )
        verifier = srv._PloidyTokenVerifier()
        await verifier.verify_token("a")
        assert call_count["n"] == 3  # did not short-circuit on the first match


class TestCurrentOwner:
    """``_current_owner`` resolves the tenant from the MCP auth context."""

    def test_no_token_map_returns_none_without_consulting_context(self, monkeypatch):
        from ploidy import server as srv

        monkeypatch.setattr(srv, "_TOKEN_MAP", {})
        # ``get_access_token`` should not even be called — guarded by the
        # token-map check so tests can run without FastMCP's context set.
        sentinel = object()
        monkeypatch.setattr(srv, "get_access_token", lambda: sentinel)
        assert srv._current_owner() is None

    def test_access_token_client_id_is_returned(self, monkeypatch):
        from ploidy import server as srv

        monkeypatch.setattr(srv, "_TOKEN_MAP", {"t": "tenant"})
        fake_token = MagicMock()
        fake_token.client_id = "tenant-7"
        monkeypatch.setattr(srv, "get_access_token", lambda: fake_token)
        assert srv._current_owner() == "tenant-7"

    def test_oauth_access_token_client_id_is_returned_without_static_map(self, monkeypatch):
        from ploidy import server as srv

        monkeypatch.setattr(srv, "_AUTH_MODE", "oauth")
        monkeypatch.setattr(srv, "_TOKEN_MAP", {})
        fake_token = MagicMock()
        fake_token.client_id = "oauth-client-7"
        monkeypatch.setattr(srv, "get_access_token", lambda: fake_token)
        assert srv._current_owner() == "oauth-client-7"

    def test_missing_access_token_returns_none(self, monkeypatch):
        from ploidy import server as srv

        monkeypatch.setattr(srv, "_TOKEN_MAP", {"t": "tenant"})
        monkeypatch.setattr(srv, "get_access_token", lambda: None)
        assert srv._current_owner() is None


class TestResolveStreamOwner:
    """``_resolve_stream_owner`` reads FastMCP's authenticated ASGI user."""

    def _request(self, access_token=None):
        req = MagicMock()
        req.scope = {}
        if access_token is not None:
            user = MagicMock()
            user.access_token = access_token
            req.scope["user"] = user
        return req

    def test_missing_authenticated_user_returns_none(self):
        from ploidy import server as srv

        assert srv._resolve_stream_owner(self._request()) is None

    def test_authenticated_user_returns_tenant(self):
        from ploidy import server as srv

        token = MagicMock(client_id="tenant-42", scopes=["debate"])
        assert srv._resolve_stream_owner(self._request(token)) == "tenant-42"

    def test_token_without_required_scope_returns_none(self):
        from ploidy import server as srv

        token = MagicMock(client_id="tenant-42", scopes=[])
        assert srv._resolve_stream_owner(self._request(token)) is None


class TestBuildLockProvider:
    """``_build_lock_provider`` picks the Redis backend when PLOIDY_REDIS_URL is set."""

    def test_no_redis_url_returns_async_lock_provider(self, monkeypatch):
        from ploidy import server as srv
        from ploidy.lockprovider import AsyncLockProvider

        # The module caches the URL at import time; patch the cached value.
        monkeypatch.setattr(srv, "_REDIS_URL", "")
        provider = srv._build_lock_provider()
        assert isinstance(provider, AsyncLockProvider)

    def test_redis_url_returns_redis_lock_provider(self, monkeypatch):
        # The factory only constructs the client — it does not open a
        # connection until first use, so pointing at a fake URL is fine.
        from ploidy import server as srv
        from ploidy.lockprovider import RedisLockProvider

        monkeypatch.setattr(srv, "_REDIS_URL", "redis://fake:6379/0")
        provider = srv._build_lock_provider()
        assert isinstance(provider, RedisLockProvider)


class TestHealthzRoute:
    async def test_ok_when_init_succeeds(self, monkeypatch):
        from ploidy import server as srv

        # Stub _init so the handler doesn't touch the real DB.
        async def fake_init():
            return MagicMock()

        monkeypatch.setattr(srv, "_init", fake_init)
        resp = await srv._healthz(MagicMock())
        assert resp.status_code == 200
        body = _json.loads(resp.body)
        assert body["status"] == "ok"

    async def test_503_when_init_raises(self, monkeypatch):
        from ploidy import server as srv

        async def broken_init():
            raise RuntimeError("db down")

        monkeypatch.setattr(srv, "_init", broken_init)
        resp = await srv._healthz(MagicMock())
        assert resp.status_code == 503
        body = _json.loads(resp.body)
        assert body["status"] == "error"
        assert "db down" in body["detail"]


class TestMetricsRoute:
    async def test_returns_prometheus_body_with_correct_media_type(self):
        from ploidy import server as srv

        resp = await srv._metrics_endpoint(MagicMock())
        # Prometheus exposition body is text/plain-ish; media_type must match
        # whatever ``metrics.content_type()`` returns so scrapers parse it.
        from ploidy.metrics import content_type

        assert resp.media_type == content_type()
        # Body is bytes; decoding it should not raise.
        resp.body.decode("utf-8")


class TestWebappRoute:
    async def test_returns_html_index(self):
        from ploidy import server as srv

        resp = await srv._webapp(MagicMock())
        assert resp.status_code == 200
        text = resp.body.decode("utf-8")
        # Webapp docstring promises static HTML — verify the shape rather
        # than the content so this test is resilient to UI tweaks.
        assert "<!DOCTYPE html>" in text or "<html" in text.lower()


class TestBuildAuthKwargs:
    """``_build_auth_kwargs`` decides which auth primitives FastMCP gets."""

    def test_bearer_mode_with_tokens_sets_only_token_verifier(self, monkeypatch):
        from mcp.server.fastmcp import FastMCP

        from ploidy import server as srv

        monkeypatch.setattr(srv, "_AUTH_MODE", "bearer")
        monkeypatch.setattr(srv, "_TOKEN_MAP", {"tok": "tenant"})
        kwargs = srv._build_auth_kwargs()
        assert "token_verifier" in kwargs
        assert isinstance(kwargs["token_verifier"], srv._PloidyTokenVerifier)
        assert "auth_server_provider" not in kwargs
        assert kwargs["auth"].required_scopes == ["debate"]
        # Regression: FastMCP requires AuthSettings whenever a verifier is set.
        FastMCP("bearer-boot", **kwargs)

    def test_bearer_mode_without_tokens_returns_empty(self, monkeypatch):
        # PLOIDY_AUTH_MODE=bearer with no tokens is a valid
        # "unauthenticated local dev" config.
        from ploidy import server as srv

        monkeypatch.setattr(srv, "_AUTH_MODE", "bearer")
        monkeypatch.setattr(srv, "_TOKEN_MAP", {})
        kwargs = srv._build_auth_kwargs()
        assert kwargs == {}

    def test_oauth_mode_sets_provider_and_auth_settings(self, monkeypatch):
        from mcp.server.fastmcp import FastMCP

        from ploidy import server as srv
        from ploidy.oauth import PloidyOAuthProvider

        monkeypatch.setattr(srv, "_AUTH_MODE", "oauth")
        monkeypatch.setattr(srv, "_TOKEN_MAP", {})
        monkeypatch.setattr(srv, "_OAUTH_ISSUER", "https://example.com")
        kwargs = srv._build_auth_kwargs()
        assert isinstance(kwargs["auth_server_provider"], PloidyOAuthProvider)
        assert kwargs["auth"] is not None
        # Discovery must advertise the configured issuer.
        assert str(kwargs["auth"].issuer_url).startswith("https://example.com")
        # Registration enabled so Claude.ai's DCR flow succeeds.
        assert kwargs["auth"].client_registration_options.enabled is True
        # Revocation enabled so clients can log out properly.
        assert kwargs["auth"].revocation_options.enabled is True
        # Bearer path must NOT be active in pure OAuth mode even if a
        # token map exists — that would create two concurrent auth
        # surfaces, which is what ``both`` mode is for.
        assert "token_verifier" not in kwargs
        FastMCP("oauth-boot", **kwargs)

    async def test_both_mode_combines_bearer_and_oauth_in_one_provider(self, monkeypatch):
        from mcp.server.fastmcp import FastMCP

        from ploidy import server as srv
        from ploidy.oauth import PloidyOAuthProvider

        monkeypatch.setattr(srv, "_AUTH_MODE", "both")
        monkeypatch.setattr(srv, "_TOKEN_MAP", {"tok": "tenant"})
        kwargs = srv._build_auth_kwargs()
        assert isinstance(kwargs["auth_server_provider"], PloidyOAuthProvider)
        assert "token_verifier" not in kwargs
        assert kwargs["auth"] is not None
        # Regression: provider+verifier is forbidden by FastMCP. The provider
        # owns the static fallback instead, so construction and verification work.
        FastMCP("both-boot", **kwargs)
        provider = kwargs["auth_server_provider"]
        access = await provider.load_access_token("tok")
        assert access is not None
        assert access.client_id == "tenant"
        await provider._store.close()

    def test_oauth_mode_without_tokens_still_works(self, monkeypatch):
        # OAuth does not rely on PLOIDY_TOKENS — a fresh deployment with
        # mode=oauth and no bearer map is a supported configuration.
        from ploidy import server as srv
        from ploidy.oauth import PloidyOAuthProvider

        monkeypatch.setattr(srv, "_AUTH_MODE", "oauth")
        monkeypatch.setattr(srv, "_TOKEN_MAP", {})
        kwargs = srv._build_auth_kwargs()
        assert isinstance(kwargs["auth_server_provider"], PloidyOAuthProvider)
        assert "token_verifier" not in kwargs


class _FakeRequest:
    """Minimal async request stand-in for the SSE handler."""

    def __init__(self, body, *, scope: dict | None = None):
        self._body = body
        self.scope = scope or {}

    async def json(self):
        return self._body


async def _drain_stream(resp, *, max_frames: int = 20) -> list[str]:
    """Collect SSE frames from a StreamingResponse, bounded to avoid hangs."""
    frames = []
    async for chunk in resp.body_iterator:
        text = chunk if isinstance(chunk, str) else chunk.decode("utf-8")
        frames.append(text)
        if len(frames) >= max_frames:
            break
    return frames


class _FakeService:
    """Stand-in for DebateService that emits a scripted progress stream."""

    def __init__(
        self,
        *,
        progress_events=(),
        run_result: dict | None = None,
        run_error: Exception | None = None,
    ):
        self._progress_events = list(progress_events)
        self._run_result = run_result or {"debate_id": "x", "phase": "complete"}
        self._run_error = run_error
        self.last_kwargs = None

    async def run_auto(self, *, progress=None, **kwargs):
        self.last_kwargs = kwargs
        if progress is not None:
            for ev in self._progress_events:
                await progress(ev)
        if self._run_error is not None:
            raise self._run_error
        return self._run_result


class TestStreamDebateRoute:
    """``_stream_debate`` wraps ``DebateService.run_auto`` with SSE framing.

    The service layer is stubbed so we can validate the SSE mechanics
    (event ordering, error framing, bad-request handling) without
    driving a real model call.
    """

    async def test_bad_body_returns_400(self, monkeypatch):
        from ploidy import server as srv

        async def fake_init():
            return _FakeService()

        monkeypatch.setattr(srv, "_init", fake_init)
        resp = await srv._stream_debate(_FakeRequest(body=[1, 2, 3]))
        assert resp.status_code == 400

    async def test_auth_enabled_rejects_missing_bearer_before_init(self, monkeypatch):
        from ploidy import server as srv

        monkeypatch.setattr(srv, "_AUTH_MODE", "bearer")
        monkeypatch.setattr(srv, "_TOKEN_MAP", {"tok": "tenant"})

        async def should_not_init():
            raise AssertionError("unauthorized request reached service initialization")

        monkeypatch.setattr(srv, "_init", should_not_init)
        resp = await srv._stream_debate(_FakeRequest(body={"prompt": "q"}))
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"

    async def test_authenticated_scope_sets_stream_tenant(self, monkeypatch):
        from ploidy import server as srv

        fake_svc = _FakeService()

        async def fake_init():
            return fake_svc

        token = MagicMock(client_id="tenant-stream", scopes=["debate"])
        user = MagicMock(access_token=token)
        monkeypatch.setattr(srv, "_AUTH_MODE", "oauth")
        monkeypatch.setattr(srv, "_TOKEN_MAP", {})
        monkeypatch.setattr(srv, "_init", fake_init)

        resp = await srv._stream_debate(_FakeRequest(body={"prompt": "q"}, scope={"user": user}))
        await _drain_stream(resp)
        assert fake_svc.last_kwargs["owner_id"] == "tenant-stream"
        assert fake_svc.last_kwargs["tenant"] == "tenant-stream"

    async def test_progress_events_and_result_reach_the_client(self, monkeypatch):
        from ploidy import server as srv
        from ploidy.stream import ProgressEvent

        fake_svc = _FakeService(
            progress_events=[
                ProgressEvent(type="phase_started", data={"phase": "position"}),
                ProgressEvent(type="positions_generated", data={"side": "deep", "count": 1}),
            ],
            run_result={"debate_id": "abc", "phase": "complete", "confidence": 0.9},
        )

        async def fake_init():
            return fake_svc

        monkeypatch.setattr(srv, "_init", fake_init)

        resp = await srv._stream_debate(_FakeRequest(body={"prompt": "q"}))
        frames = await _drain_stream(resp)
        combined = "".join(frames)
        assert "phase_started" in combined
        assert "positions_generated" in combined
        # ``result`` frame is emitted after the service call returns.
        assert "result" in combined
        assert "abc" in combined

    async def test_service_exception_emits_error_frame(self, monkeypatch):
        from ploidy import server as srv

        fake_svc = _FakeService(run_error=RuntimeError("model blew up"))

        async def fake_init():
            return fake_svc

        monkeypatch.setattr(srv, "_init", fake_init)

        resp = await srv._stream_debate(_FakeRequest(body={"prompt": "q"}))
        combined = "".join(await _drain_stream(resp))
        assert "error" in combined
        assert "model blew up" in combined
        # ``_drain_stream`` bounds iteration — if the error frame did not
        # terminate the stream cleanly, we'd hit the frame cap with
        # timeout-or-empty content, not a useful assertion failure. The
        # ``in`` checks above cover the termination guard indirectly.
