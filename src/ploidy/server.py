"""MCP server entry point for Ploidy.

Thin tool layer over ``DebateService``. Tools validate MCP-specific
concerns (annotations, docstrings surfaced to clients) and forward to
the service. All state lives on the service instance.

Primary tool (v0.4):
- debate(prompt, mode="auto"|"solo", ...): unified entry point.
  Pick auto for API-generated debate or solo for caller-supplied
  positions. Handles HITL via pause_at + debate_review.

Legacy tools (deprecated, kept for two-terminal and HITL workflows):
- debate_start / debate_join / debate_position / debate_challenge
- debate_converge / debate_cancel / debate_delete
- debate_status / debate_history
- debate_auto / debate_solo / debate_review
"""

import asyncio
import hmac
import json
import logging
import os

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ploidy.lockprovider import AsyncLockProvider, LockProvider, RedisLockProvider
from ploidy.logctx import deprecated, traced
from ploidy.logctx import install as install_logctx
from ploidy.ratelimit import TokenBucketLimiter
from ploidy.service import DebateService
from ploidy.store import DebateStore

logger = logging.getLogger("ploidy")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PORT = int(os.environ.get("PLOIDY_PORT", "8765"))
_MAX_PROMPT_LEN = int(os.environ.get("PLOIDY_MAX_PROMPT_LEN", "10000"))
_MAX_CONTENT_LEN = int(os.environ.get("PLOIDY_MAX_CONTENT_LEN", "50000"))
_MAX_CONTEXT_DOCS = int(os.environ.get("PLOIDY_MAX_CONTEXT_DOCS", "10"))
_MAX_SESSIONS_PER_DEBATE = int(os.environ.get("PLOIDY_MAX_SESSIONS", "5"))
# Combined context_documents token ceiling (approx ~4 chars/token).
# 0 disables the cap for research behaviour; production deployments
# should set this to something sane (e.g. 20000) to avoid runaway
# input-token spend.
_MAX_CONTEXT_TOKENS = int(os.environ.get("PLOIDY_MAX_CONTEXT_TOKENS", "0"))
_AUTH_TOKEN = os.environ.get("PLOIDY_AUTH_TOKEN")
# Multi-tenant: a JSON map {token: tenant_id}. When set, overrides
# PLOIDY_AUTH_TOKEN. Each accepted token resolves to a distinct
# AccessToken.client_id, which the service uses as owner_id.
_TOKEN_MAP_RAW = os.environ.get("PLOIDY_TOKENS", "")
# Auth mode selector. ``bearer`` (default, legacy) keeps today's
# PLOIDY_TOKENS behaviour. ``oauth`` swaps in the full OAuth 2.0
# Authorization Server (required for Claude.ai Connectors Directory).
# ``both`` runs OAuth alongside legacy bearer for the transition window.
_AUTH_MODE = os.environ.get("PLOIDY_AUTH_MODE", "bearer").lower()
if _AUTH_MODE not in ("bearer", "oauth", "both"):
    logger.warning(
        "Unknown PLOIDY_AUTH_MODE=%r; falling back to 'bearer'. "
        "Valid values: bearer / oauth / both.",
        _AUTH_MODE,
    )
    _AUTH_MODE = "bearer"
# Issuer URL advertised in the OAuth discovery document. Defaults to
# ``http://localhost:$PLOIDY_PORT`` for local dev; production must set
# this to the publicly reachable HTTPS base URL.
_OAUTH_ISSUER = os.environ.get("PLOIDY_OAUTH_ISSUER", f"http://localhost:{_PORT}")


def _load_token_map() -> dict[str, str]:
    if _TOKEN_MAP_RAW:
        try:
            parsed = json.loads(_TOKEN_MAP_RAW)
        except json.JSONDecodeError as exc:
            logger.error("PLOIDY_TOKENS is not valid JSON: %s", exc)
            return {}
        if not isinstance(parsed, dict):
            logger.error("PLOIDY_TOKENS must decode to an object of token→tenant")
            return {}
        return {str(k): str(v) for k, v in parsed.items()}
    if _AUTH_TOKEN:
        return {_AUTH_TOKEN: "ploidy-client"}
    return {}


_TOKEN_MAP: dict[str, str] = _load_token_map()

_USE_LLM_CONVERGENCE = os.environ.get("PLOIDY_LLM_CONVERGENCE", "").lower() in (
    "1",
    "true",
    "yes",
)
# 0 disables the limiter. Capacity is the burst allowance; rate is sustained.
_RATE_CAPACITY = float(os.environ.get("PLOIDY_RATE_CAPACITY", "0"))
_RATE_PER_SEC = float(os.environ.get("PLOIDY_RATE_PER_SEC", "0"))
# Retention: purge completed/cancelled debates older than N days. 0 disables.
_RETENTION_DAYS = int(os.environ.get("PLOIDY_RETENTION_DAYS", "0"))
_RETENTION_INTERVAL_SEC = int(os.environ.get("PLOIDY_RETENTION_INTERVAL_SEC", "3600"))
_RETENTION_VACUUM = os.environ.get("PLOIDY_RETENTION_VACUUM", "1").lower() in ("1", "true", "yes")
# Optional distributed lock backend for multi-replica deployments. When
# unset, single-node asyncio.Lock continues to be used.
_REDIS_URL = os.environ.get("PLOIDY_REDIS_URL")
_REDIS_LOCK_TTL_MS = int(os.environ.get("PLOIDY_REDIS_LOCK_TTL_MS", "30000"))


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class _PloidyTokenVerifier:
    """Bearer token verifier with constant-time compare over a token map.

    The resolved ``AccessToken.client_id`` becomes the ``owner_id`` every
    tool call carries, so the token the caller presents effectively picks
    which tenant's data they can see.
    """

    async def verify_token(self, token: str) -> AccessToken | None:
        if not _TOKEN_MAP:
            return None
        presented = token.encode("utf-8")
        # Iterate all entries so the comparison time is independent of
        # which token (if any) matches, instead of short-circuiting.
        matched_tenant: str | None = None
        for candidate, tenant in _TOKEN_MAP.items():
            if hmac.compare_digest(presented, candidate.encode("utf-8")):
                matched_tenant = tenant
        if matched_tenant is None:
            return None
        return AccessToken(
            token=token,
            client_id=matched_tenant,
            scopes=["debate"],
        )


def _build_auth_kwargs() -> dict:
    """Assemble the auth kwargs passed to FastMCP based on ``PLOIDY_AUTH_MODE``."""
    from mcp.server.auth.settings import (
        AuthSettings,
        ClientRegistrationOptions,
        RevocationOptions,
    )

    bearer_enabled = _AUTH_MODE in ("bearer", "both") and bool(_TOKEN_MAP)
    oauth_enabled = _AUTH_MODE in ("oauth", "both")
    if not bearer_enabled and not oauth_enabled:
        return {}

    kwargs: dict = {
        "auth": AuthSettings(
            issuer_url=_OAUTH_ISSUER,
            # Ploidy is a self-contained AS/RS in OAuth modes. In static
            # bearer mode this URL still identifies the protected resource
            # in the MCP WWW-Authenticate challenge.
            resource_server_url=_OAUTH_ISSUER,
            client_registration_options=(
                ClientRegistrationOptions(
                    enabled=True,
                    valid_scopes=["debate"],
                    default_scopes=["debate"],
                )
                if oauth_enabled
                else None
            ),
            revocation_options=(RevocationOptions(enabled=True) if oauth_enabled else None),
            required_scopes=["debate"],
        )
    }

    static_verifier = _PloidyTokenVerifier() if bearer_enabled else None
    if oauth_enabled:
        from ploidy.oauth import PloidyOAuthProvider

        # FastMCP permits exactly one of provider/verifier. In ``both``
        # mode the OAuth provider delegates unknown tokens to the static
        # verifier, preserving a single verifier surface while mounting
        # the OAuth endpoints.
        oauth_store = DebateStore()
        kwargs["auth_server_provider"] = PloidyOAuthProvider(
            oauth_store,
            fallback_token_verifier=static_verifier,
        )
        logger.info("OAuth 2.0 auth server enabled (issuer=%s)", _OAUTH_ISSUER)
    elif static_verifier is not None:
        kwargs["token_verifier"] = static_verifier

    if bearer_enabled:
        logger.info(
            "Bearer token auth enabled; %d tenant token(s) loaded",
            len(_TOKEN_MAP),
        )
    return kwargs


_auth_kwargs: dict = _build_auth_kwargs()


def _auth_is_enabled() -> bool:
    """Return whether requests must carry an authenticated bearer token."""
    if _AUTH_MODE in ("oauth", "both"):
        return True
    return _AUTH_MODE == "bearer" and bool(_TOKEN_MAP)


def _current_owner() -> str | None:
    """Return the caller's tenant id from the MCP auth context, if any."""
    if not _auth_is_enabled():
        return None
    tok = get_access_token()
    return tok.client_id if tok is not None else None


mcp = FastMCP(
    "Ploidy",
    instructions="Cross-session multi-agent debate MCP server. "
    "Same model, different context depths, better decisions.",
    port=_PORT,
    **_auth_kwargs,
)


@mcp.custom_route("/", methods=["GET"])
async def _webapp(_request):
    """Single-page debate UI served from the root path.

    Unauthenticated on purpose — the form itself collects a bearer
    token when the server has ``PLOIDY_TOKENS`` configured. The page
    is pure static HTML (no server state leaked) so public exposure
    of the root path is safe.
    """
    from starlette.responses import HTMLResponse

    from ploidy.webapp import index_html

    return HTMLResponse(index_html())


@mcp.custom_route("/healthz", methods=["GET"])
async def _healthz(_request):
    """Liveness probe. Succeeds once the DB connection is initialised."""
    from starlette.responses import JSONResponse

    try:
        await _init()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=503)
    return JSONResponse({"status": "ok"})


@mcp.custom_route("/metrics", methods=["GET"])
async def _metrics_endpoint(_request):
    """Prometheus scrape endpoint. Unauthenticated by design (infra-private)."""
    from starlette.responses import Response

    from ploidy.metrics import content_type, metrics

    body = metrics().render()
    return Response(body, media_type=content_type())


@mcp.custom_route("/v1/debate/stream", methods=["POST"])
async def _stream_debate(request):
    """Server-Sent Events stream for ``debate(mode='auto')``.

    The MCP tool call returns once the whole debate has converged, which
    hides 20-60 seconds of intermediate phases. This endpoint runs the
    same service path but emits ``phase_started`` / ``positions_generated``
    / ``challenges_generated`` / ``completed`` frames as they happen, so
    web / Discord clients can render a live progress UI.

    Auth mirrors the MCP tool surface. When bearer or OAuth auth is
    configured, missing/invalid credentials are rejected before parsing
    the body or initialising the service. With auth disabled, requests
    retain the unscoped local-development behaviour.
    """
    import asyncio as _asyncio

    from starlette.responses import JSONResponse, StreamingResponse

    from ploidy.stream import ProgressEvent, sse_format

    owner = _resolve_stream_owner(request)
    if _auth_is_enabled() and owner is None:
        return JSONResponse(
            {"error": "unauthorized"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )

    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

    svc = await _init()

    queue: _asyncio.Queue[ProgressEvent | None] = _asyncio.Queue()

    async def on_progress(event: ProgressEvent) -> None:
        await queue.put(event)

    async def run() -> None:
        try:
            result = await svc.run_auto(
                prompt=body.get("prompt", ""),
                context_documents=body.get("context_documents"),
                context_sources=body.get("context_sources"),
                blocked_sources=body.get("blocked_sources"),
                target_lease=body.get("target_lease"),
                allowed_sources=body.get("allowed_sources"),
                fresh_role=body.get("fresh_role", "fresh"),
                delivery_mode=body.get("delivery_mode", "none"),
                pause_at=body.get("pause_at"),
                deep_n=int(body.get("deep_n", 1)),
                fresh_n=int(body.get("fresh_n", 1)),
                effort=body.get("effort", "high"),
                injection_mode=body.get("injection_mode", "raw"),
                context_pct=int(body.get("context_pct", 100)),
                language=body.get("language", "en"),
                deep_model=body.get("deep_model"),
                fresh_model=body.get("fresh_model"),
                tenant=owner or "global",
                owner_id=owner,
                progress=on_progress,
            )
            await queue.put(ProgressEvent(type="result", data=result))
        except Exception as exc:  # noqa: BLE001
            await queue.put(
                ProgressEvent(type="error", data={"message": str(exc), "kind": type(exc).__name__})
            )
        finally:
            await queue.put(None)

    worker = _asyncio.create_task(run())

    async def event_stream():
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield sse_format(event)
        finally:
            # If the client disconnects mid-stream, cancel the worker
            # so no token is wasted on output nobody is consuming. Await it
            # before closing the generator so service cancellation cleanup is
            # complete when the disconnect boundary returns.
            if not worker.done():
                worker.cancel()
            try:
                await worker
            except _asyncio.CancelledError:
                pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _resolve_stream_owner(request) -> str | None:
    """Resolve the authenticated tenant attached by FastMCP middleware.

    FastMCP custom routes are not wrapped in its ``RequireAuthMiddleware``,
    but the app-level bearer authentication middleware still records the
    verified ``AccessToken`` on the ASGI scope. The route checks that token
    itself so static and OAuth credentials follow the exact same verifier.
    """
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        return None
    user = scope.get("user")
    access_token = getattr(user, "access_token", None)
    if access_token is None or "debate" not in access_token.scopes:
        return None
    return access_token.client_id


# ---------------------------------------------------------------------------
# Service instance
# ---------------------------------------------------------------------------

_service: DebateService | None = None
_init_lock = asyncio.Lock()


def _build_lock_provider() -> LockProvider:
    if _REDIS_URL:
        try:
            from redis.asyncio import Redis
        except ImportError:
            logger.error(
                "PLOIDY_REDIS_URL is set but the 'redis' package is missing; "
                "falling back to single-node locks"
            )
            return AsyncLockProvider()
        client = Redis.from_url(_REDIS_URL, decode_responses=True)
        logger.info("RedisLockProvider enabled (url=%s, ttl_ms=%d)", _REDIS_URL, _REDIS_LOCK_TTL_MS)
        return RedisLockProvider(client, ttl_ms=_REDIS_LOCK_TTL_MS)
    return AsyncLockProvider()


async def _init() -> DebateService:
    """Lazily construct and initialise the shared DebateService."""
    global _service
    if _service is not None and _service._initialized:
        return _service
    async with _init_lock:
        if _service is None:
            _service = DebateService(
                store=DebateStore(),
                use_llm_convergence=_USE_LLM_CONVERGENCE,
                max_prompt_len=_MAX_PROMPT_LEN,
                max_content_len=_MAX_CONTENT_LEN,
                max_context_docs=_MAX_CONTEXT_DOCS,
                max_sessions_per_debate=_MAX_SESSIONS_PER_DEBATE,
                max_context_tokens=_MAX_CONTEXT_TOKENS or None,
                rate_limiter=TokenBucketLimiter(
                    capacity=_RATE_CAPACITY, rate_per_sec=_RATE_PER_SEC
                ),
                lock_provider=_build_lock_provider(),
                retention_days=_RETENTION_DAYS,
                retention_interval_seconds=_RETENTION_INTERVAL_SEC,
                retention_vacuum=_RETENTION_VACUUM,
            )
        await _service.initialize()
        return _service


async def shutdown() -> None:
    """Close the database connection and drop runtime state."""
    global _service
    if _service is not None:
        await _service.shutdown()
        _service = None
    logger.info("Database connection closed and runtime state cleared")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(
    annotations=ToolAnnotations(destructiveHint=True, readOnlyHint=False, idempotentHint=False),
)
@traced
async def debate(
    prompt: str,
    mode: str = "auto",
    # solo-mode inputs
    deep_position: str | None = None,
    fresh_position: str | None = None,
    deep_challenge: str | None = None,
    fresh_challenge: str | None = None,
    deep_label: str = "Deep",
    fresh_label: str = "Fresh",
    # auto-mode inputs
    fresh_role: str = "fresh",
    delivery_mode: str = "none",
    pause_at: str | None = None,
    deep_n: int = 1,
    fresh_n: int = 1,
    effort: str = "high",
    injection_mode: str = "raw",
    context_pct: int = 100,
    language: str = "en",
    deep_model: str | None = None,
    fresh_model: str | None = None,
    # shared
    context_documents: list[str] | None = None,
    context_sources: list[str] | None = None,
    blocked_sources: list[str] | None = None,
    target_lease: str | None = None,
    allowed_sources: list[str] | None = None,
) -> dict:
    """Run a context-asymmetric debate in a single call.

    One canonical entry point. Pick a mode:

    - ``auto`` (default): Ploidy generates both sides via the configured
      OpenAI-compatible API endpoint and returns the convergence result.
      Requires non-empty ``context_documents`` plus a configured API
      endpoint. HITL pause/resume is available via ``pause_at``; resume
      with the legacy ``debate_review`` tool.
    - ``solo``: you supply both positions (and optionally both challenges)
      and Ploidy persists + converges them. No external API key needed —
      the recommended single-terminal flow when the caller (e.g. Claude
      Code itself) writes both sides locally.

    The older 12-tool surface (``debate_start`` / ``debate_join`` /
    ``debate_position`` / ...) remains available for two-terminal and
    HITL workflows but is now deprecated; new integrations should use
    this tool.

    Args:
        prompt: The decision question to debate.
        mode: ``"auto"`` or ``"solo"``.
        deep_position: (solo only) Deep side's stance.
        fresh_position: (solo only) Fresh side's stance.
        deep_challenge: (solo only) Optional deep-side critique.
        fresh_challenge: (solo only) Optional fresh-side critique.
        deep_label / fresh_label: (solo) display labels for roles.
        fresh_role: (auto) ``"fresh"`` or ``"semi_fresh"``.
        delivery_mode: (auto + semi_fresh) ``"passive"``, ``"active"``,
            or ``"selective"``.
        pause_at: (auto) Optional ``"challenge"`` or ``"convergence"``
            HITL pause point.
        deep_n / fresh_n: (auto) Ploidy level per side.
        effort: (auto) ``"low"``, ``"medium"``, ``"high"``, ``"max"``.
        injection_mode: (auto) Context formatting mode.
        context_pct: (auto) Percentage of context to retain.
        language: (auto) Output language code.
        deep_model / fresh_model: (auto) Model override. Supplying either
            applies it to both sides; if both are supplied they must match.
        context_documents: Documents attached only to the Deep side. Required
            and non-empty in auto mode; optional provenance in solo mode.
        context_sources: Optional provenance labels, one per context document.
        blocked_sources: Optional strings that must not appear in source labels
            or context documents.
        target_lease: Optional target identifier that pins this debate's scope.
        allowed_sources: Optional source-label allowlist; defaults to target_lease
            when a target lease is set.

    Returns:
        Convergence result dict, or paused state (auto + pause_at).
    """
    svc = await _init()
    owner = _current_owner()

    if mode == "solo":
        # Treat None and blank strings symmetrically — a whitespace-only
        # position is almost certainly a caller mistake and would fail
        # convergence later with a less informative error.
        if not (deep_position and deep_position.strip()):
            raise ValueError("debate(mode='solo') requires a non-empty deep_position")
        if not (fresh_position and fresh_position.strip()):
            raise ValueError("debate(mode='solo') requires a non-empty fresh_position")
        return await svc.run_solo(
            prompt=prompt,
            deep_position=deep_position,
            fresh_position=fresh_position,
            deep_challenge=deep_challenge,
            fresh_challenge=fresh_challenge,
            context_documents=context_documents,
            context_sources=context_sources,
            blocked_sources=blocked_sources,
            target_lease=target_lease,
            allowed_sources=allowed_sources,
            deep_label=deep_label,
            fresh_label=fresh_label,
            tenant=owner or "global",
            owner_id=owner,
        )

    if mode == "auto":
        for name, value in (
            ("deep_position", deep_position),
            ("fresh_position", fresh_position),
            ("deep_challenge", deep_challenge),
            ("fresh_challenge", fresh_challenge),
        ):
            if value is not None:
                raise ValueError(
                    f"debate(mode='auto') does not accept {name}; pass these only when mode='solo'"
                )
        return await svc.run_auto(
            prompt=prompt,
            context_documents=context_documents,
            context_sources=context_sources,
            blocked_sources=blocked_sources,
            target_lease=target_lease,
            allowed_sources=allowed_sources,
            fresh_role=fresh_role,
            delivery_mode=delivery_mode,
            pause_at=pause_at,
            deep_n=deep_n,
            fresh_n=fresh_n,
            effort=effort,
            injection_mode=injection_mode,
            context_pct=context_pct,
            language=language,
            deep_model=deep_model,
            fresh_model=fresh_model,
            tenant=owner or "global",
            owner_id=owner,
        )

    raise ValueError(f"Invalid mode '{mode}'. Must be 'auto' or 'solo'")


@mcp.tool(
    annotations=ToolAnnotations(destructiveHint=True, readOnlyHint=False, idempotentHint=False),
)
@deprecated(version="0.4", prefer="``debate(prompt, mode=...)``")
@traced
async def debate_start(
    prompt: str,
    context_documents: list[str] | None = None,
    context_sources: list[str] | None = None,
    blocked_sources: list[str] | None = None,
    target_lease: str | None = None,
    allowed_sources: list[str] | None = None,
) -> dict:
    """Begin a new debate session with a decision prompt.

    Creates a debate and a Deep (full-context) session.
    Share the returned debate_id with the fresh session so it can join.
    """
    svc = await _init()
    owner = _current_owner()
    return await svc.start_debate(
        prompt,
        context_documents,
        context_sources=context_sources,
        blocked_sources=blocked_sources,
        target_lease=target_lease,
        allowed_sources=allowed_sources,
        tenant=owner or "global",
        owner_id=owner,
    )


@mcp.tool(
    annotations=ToolAnnotations(destructiveHint=False, readOnlyHint=False, idempotentHint=False),
)
@deprecated(version="0.4", prefer="``debate(prompt, mode=...)``")
@traced
async def debate_join(
    debate_id: str,
    role: str = "fresh",
    delivery_mode: str = "none",
) -> dict:
    """Join an existing debate as a fresh or semi-fresh session."""
    svc = await _init()
    return await svc.join_debate(debate_id, role, delivery_mode, owner_id=_current_owner())


@mcp.tool(
    annotations=ToolAnnotations(destructiveHint=False, readOnlyHint=False, idempotentHint=False),
)
@deprecated(version="0.4", prefer="``debate(prompt, mode=...)``")
@traced
async def debate_position(session_id: str, content: str) -> dict:
    """Submit a position from a session."""
    svc = await _init()
    return await svc.submit_position(session_id, content, owner_id=_current_owner())


@mcp.tool(
    annotations=ToolAnnotations(destructiveHint=False, readOnlyHint=False, idempotentHint=False),
)
@deprecated(version="0.4", prefer="``debate(prompt, mode=...)``")
@traced
async def debate_challenge(session_id: str, content: str, action: str = "challenge") -> dict:
    """Submit a challenge to another session's position."""
    svc = await _init()
    return await svc.submit_challenge(session_id, content, action, owner_id=_current_owner())


@mcp.tool(
    annotations=ToolAnnotations(
        destructiveHint=False,
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
@deprecated(version="0.4", prefer="``debate(prompt, mode=...)``")
@traced
async def debate_converge(debate_id: str) -> dict:
    """Trigger convergence analysis for a debate."""
    svc = await _init()
    return await svc.converge(debate_id, owner_id=_current_owner())


@mcp.tool(
    annotations=ToolAnnotations(destructiveHint=True, readOnlyHint=False, idempotentHint=True),
)
@deprecated(version="0.4", prefer="``debate(prompt, mode=...)``")
@traced
async def debate_cancel(debate_id: str) -> dict:
    """Cancel a debate in progress."""
    svc = await _init()
    return await svc.cancel(debate_id, owner_id=_current_owner())


@mcp.tool(
    annotations=ToolAnnotations(destructiveHint=True, readOnlyHint=False, idempotentHint=True),
)
@deprecated(version="0.4", prefer="``debate(prompt, mode=...)``")
@traced
async def debate_delete(debate_id: str) -> dict:
    """Permanently delete a debate and all its data."""
    svc = await _init()
    return await svc.delete(debate_id, owner_id=_current_owner())


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
@deprecated(version="0.4", prefer="``debate(prompt, mode=...)``")
@traced
async def debate_status(debate_id: str) -> dict:
    """Get current state of a debate."""
    svc = await _init()
    return await svc.status(debate_id, owner_id=_current_owner())


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True),
)
@deprecated(version="0.4", prefer="``debate(prompt, mode=...)``")
@traced
async def debate_history(limit: int = 50) -> dict:
    """Retrieve past debates and their outcomes."""
    svc = await _init()
    return await svc.history(limit, owner_id=_current_owner())


@mcp.tool(
    annotations=ToolAnnotations(destructiveHint=True, readOnlyHint=False, idempotentHint=False),
)
@deprecated(version="0.4", prefer="``debate(prompt, mode=...)``")
@traced
async def debate_solo(
    prompt: str,
    deep_position: str,
    fresh_position: str,
    deep_challenge: str | None = None,
    fresh_challenge: str | None = None,
    context_documents: list[str] | None = None,
    context_sources: list[str] | None = None,
    blocked_sources: list[str] | None = None,
    target_lease: str | None = None,
    allowed_sources: list[str] | None = None,
    deep_label: str = "Deep",
    fresh_label: str = "Fresh",
) -> dict:
    """Run a complete debate from caller-supplied positions in one call.

    Single-terminal entry point: the caller generates both sides locally
    and submits the texts here. No external API key required.
    """
    svc = await _init()
    owner = _current_owner()
    return await svc.run_solo(
        prompt=prompt,
        deep_position=deep_position,
        fresh_position=fresh_position,
        deep_challenge=deep_challenge,
        fresh_challenge=fresh_challenge,
        context_documents=context_documents,
        context_sources=context_sources,
        blocked_sources=blocked_sources,
        target_lease=target_lease,
        allowed_sources=allowed_sources,
        deep_label=deep_label,
        fresh_label=fresh_label,
        tenant=owner or "global",
        owner_id=owner,
    )


@mcp.tool(
    annotations=ToolAnnotations(destructiveHint=True, readOnlyHint=False, idempotentHint=False),
)
@deprecated(version="0.4", prefer="``debate(prompt, mode=...)``")
@traced
async def debate_auto(
    prompt: str,
    context_documents: list[str] | None = None,
    context_sources: list[str] | None = None,
    blocked_sources: list[str] | None = None,
    target_lease: str | None = None,
    allowed_sources: list[str] | None = None,
    fresh_role: str = "fresh",
    delivery_mode: str = "none",
    pause_at: str | None = None,
    deep_n: int = 1,
    fresh_n: int = 1,
    effort: str = "high",
    injection_mode: str = "raw",
    context_pct: int = 100,
    language: str = "en",
    deep_model: str | None = None,
    fresh_model: str | None = None,
) -> dict:
    """Run a complete debate automatically in a single command.

    Requires non-empty ``context_documents`` and a configured API endpoint.
    Generates positions and one independent challenge per seat with one
    resolved model, runs the protocol, and returns the convergence result.
    """
    svc = await _init()
    owner = _current_owner()
    return await svc.run_auto(
        prompt=prompt,
        context_documents=context_documents,
        context_sources=context_sources,
        blocked_sources=blocked_sources,
        target_lease=target_lease,
        allowed_sources=allowed_sources,
        fresh_role=fresh_role,
        delivery_mode=delivery_mode,
        pause_at=pause_at,
        deep_n=deep_n,
        fresh_n=fresh_n,
        effort=effort,
        injection_mode=injection_mode,
        context_pct=context_pct,
        language=language,
        deep_model=deep_model,
        fresh_model=fresh_model,
        tenant=owner or "global",
        owner_id=owner,
    )


@mcp.tool(
    annotations=ToolAnnotations(destructiveHint=False, readOnlyHint=False, idempotentHint=False),
)
@deprecated(version="0.4", prefer="``debate(prompt, mode=...)``")
@traced
async def debate_review(
    debate_id: str,
    action: str = "approve",
    override_content: str | None = None,
) -> dict:
    """Review and resume a paused auto-debate (HITL).

    Call after ``debate_auto`` with ``pause_at`` paused the run. Action
    is one of 'approve', 'override', or 'reject'.
    """
    svc = await _init()
    return await svc.review(debate_id, action, override_content, owner_id=_current_owner())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Ploidy MCP server."""
    log_level = os.environ.get("PLOIDY_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=(
            "%(asctime)s [%(name)s] %(levelname)s "
            "req=%(request_id)s debate=%(debate_id)s: %(message)s"
        ),
    )
    install_logctx(level=getattr(logging, log_level, logging.INFO))

    # Default to stdio so MCP clients (Claude Code, etc.) can spawn the server
    # on demand. Set PLOIDY_TRANSPORT=streamable-http (or sse) for the
    # multi-client cross-session deployment.
    transport = os.environ.get("PLOIDY_TRANSPORT", "stdio")

    # FastMCP owns the event loop once mcp.run() starts; a signal handler
    # that calls run_until_complete() from inside a running loop raises
    # RuntimeError, so we schedule shutdown on the active loop instead.
    import signal

    def _shutdown_handler(sig: int, frame: object) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(shutdown())
            return
        loop.create_task(shutdown())

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    mcp.run(transport=transport)
