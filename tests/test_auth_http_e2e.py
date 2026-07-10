"""Auth boot and custom SSE route tests through the real FastMCP ASGI stack."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from mcp.server.fastmcp import FastMCP

from ploidy import server


class _CaptureService:
    """Record the tenant arguments received by the streaming service path."""

    def __init__(self) -> None:
        self.owners: list[tuple[str | None, str]] = []

    async def run_auto(self, *, owner_id=None, tenant="global", **_kwargs):
        self.owners.append((owner_id, tenant))
        return {"debate_id": "auth-e2e", "phase": "complete"}


def _stream_app(auth_kwargs: dict) -> object:
    """Build the same authenticated custom-route shape used by Ploidy."""
    mcp = FastMCP("PloidyAuthE2E", **auth_kwargs)
    mcp.custom_route("/v1/debate/stream", methods=["POST"])(server._stream_debate)
    return mcp.streamable_http_app()


@pytest.mark.parametrize("mode", ["bearer", "oauth", "both"])
def test_server_module_imports_with_real_auth_environment(tmp_path, mode):
    """Every supported auth mode must survive the module-level FastMCP boot."""
    env = os.environ.copy()
    env.update(
        {
            "PLOIDY_AUTH_MODE": mode,
            "PLOIDY_DB_PATH": str(tmp_path / f"{mode}.db"),
            "PLOIDY_OAUTH_ISSUER": "http://localhost:8765",
        }
    )
    if mode in ("bearer", "both"):
        env["PLOIDY_TOKENS"] = '{"static-token":"static-tenant"}'
    else:
        env.pop("PLOIDY_TOKENS", None)
        env.pop("PLOIDY_AUTH_TOKEN", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ploidy.server import mcp; "
            "assert mcp.settings.auth is not None; print('auth-ready')",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "auth-ready" in result.stdout


async def test_bearer_sse_rejects_missing_and_invalid_tokens_and_scopes_valid_owner(
    monkeypatch,
):
    """Static bearer auth is mandatory on SSE and its tenant reaches the service."""
    monkeypatch.setattr(server, "_AUTH_MODE", "bearer")
    monkeypatch.setattr(server, "_TOKEN_MAP", {"static-token": "static-tenant"})
    capture = _CaptureService()

    async def fake_init():
        return capture

    monkeypatch.setattr(server, "_init", fake_init)
    app = _stream_app(server._build_auth_kwargs())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        missing = await client.post("/v1/debate/stream", json={"prompt": "q"})
        invalid = await client.post(
            "/v1/debate/stream",
            headers={"Authorization": "Bearer wrong"},
            json={"prompt": "q"},
        )
        valid = await client.post(
            "/v1/debate/stream",
            headers={"Authorization": "Bearer static-token"},
            json={"prompt": "q"},
        )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert valid.status_code == 200
    assert capture.owners == [("static-tenant", "static-tenant")]


async def test_oauth_only_sse_uses_authenticated_client_id_as_owner(monkeypatch):
    """An OAuth access token scopes the custom route to its registered client."""
    monkeypatch.setattr(server, "_AUTH_MODE", "oauth")
    monkeypatch.setattr(server, "_TOKEN_MAP", {})
    kwargs = server._build_auth_kwargs()
    provider = kwargs["auth_server_provider"]
    await provider._store.initialize()
    await provider._store.save_oauth_client(
        "oauth-tenant",
        redirect_uris=["https://example.com/callback"],
        grant_types=["authorization_code", "refresh_token"],
    )
    await provider._store.save_oauth_token(
        "oauth-access",
        kind="access",
        client_id="oauth-tenant",
        scopes=["debate"],
    )
    capture = _CaptureService()

    async def fake_init():
        return capture

    monkeypatch.setattr(server, "_init", fake_init)
    app = _stream_app(kwargs)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/debate/stream",
                headers={"Authorization": "Bearer oauth-access"},
                json={"prompt": "q"},
            )
        assert response.status_code == 200
        assert capture.owners == [("oauth-tenant", "oauth-tenant")]
    finally:
        await provider._store.close()


async def test_both_mode_accepts_static_and_oauth_tokens_through_one_provider(monkeypatch):
    """Transition mode supports both credential families without dual FastMCP verifiers."""
    monkeypatch.setattr(server, "_AUTH_MODE", "both")
    monkeypatch.setattr(server, "_TOKEN_MAP", {"static-access": "static-tenant"})
    kwargs = server._build_auth_kwargs()
    assert "token_verifier" not in kwargs
    provider = kwargs["auth_server_provider"]
    await provider._store.initialize()
    await provider._store.save_oauth_client(
        "oauth-tenant",
        redirect_uris=["https://example.com/callback"],
        grant_types=["authorization_code", "refresh_token"],
    )
    await provider._store.save_oauth_token(
        "oauth-access",
        kind="access",
        client_id="oauth-tenant",
        scopes=["debate"],
    )
    capture = _CaptureService()

    async def fake_init():
        return capture

    monkeypatch.setattr(server, "_init", fake_init)
    app = _stream_app(kwargs)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            static_response = await client.post(
                "/v1/debate/stream",
                headers={"Authorization": "Bearer static-access"},
                json={"prompt": "q"},
            )
            oauth_response = await client.post(
                "/v1/debate/stream",
                headers={"Authorization": "Bearer oauth-access"},
                json={"prompt": "q"},
            )
        assert static_response.status_code == 200
        assert oauth_response.status_code == 200
        assert capture.owners == [
            ("static-tenant", "static-tenant"),
            ("oauth-tenant", "oauth-tenant"),
        ]
    finally:
        await provider._store.close()
