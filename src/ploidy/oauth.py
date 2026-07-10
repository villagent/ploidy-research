"""OAuth 2.0 Authorization Server provider backed by ``DebateStore``.

Implements ``mcp.server.auth.provider.OAuthAuthorizationServerProvider``
so FastMCP can mount the standard discovery / ``/authorize`` / ``/token``
/ DCR / revocation routes. See ``planning/oauth-integration.md`` for
the design rationale; this module is slice 2 of that plan (provider
logic only — no FastMCP wiring yet; that lives in a follow-up PR gated
behind ``PLOIDY_AUTH_MODE``).

Design invariants:

- **Opaque tokens**: access and refresh tokens are 32-byte URL-safe
  random strings, not JWTs. Lookups go through the DB on every tool
  call; revocation is immediate and does not wait for JWT expiry.
- **PKCE S256 only**: ``authorize`` rejects any other code challenge
  method at issue time, so the exchange path can trust the stored
  value without re-validating the algorithm.
- **Single-use codes**: ``consume_oauth_code`` in the store flips
  ``used=1`` atomically. A replayed code cannot redeem twice.
- **Refresh rotation**: ``exchange_refresh_token`` revokes the old
  refresh token before issuing the new pair, matching OAuth 2.1.
- **Scope subsetting**: refresh exchanges can only narrow scopes, never
  widen them.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    RegistrationError,
    TokenError,
    TokenVerifier,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from ploidy.store import DebateStore

# Access tokens are short-lived; refresh tokens are long-lived and
# rotated on exchange. 1h / no-expiry matches typical OAuth 2.1 defaults.
_ACCESS_TOKEN_TTL = timedelta(hours=1)
_CODE_TTL = timedelta(minutes=5)


def _now_iso() -> str:
    # SQLite's ``datetime('now')`` uses ``YYYY-MM-DD HH:MM:SS`` in UTC —
    # match that shape so string comparisons behave as temporal ones.
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _plus_iso(delta: timedelta) -> str:
    return (datetime.now(UTC) + delta).strftime("%Y-%m-%d %H:%M:%S")


def _iso_to_unix(iso: str) -> int:
    return int(datetime.strptime(iso, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp())


def _new_token() -> str:
    return secrets.token_urlsafe(32)


class PloidyOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    """Store-backed OAuth 2.0 Authorization Server provider."""

    def __init__(
        self,
        store: DebateStore,
        fallback_token_verifier: TokenVerifier | None = None,
    ) -> None:
        """Create a provider with an optional legacy bearer fallback.

        FastMCP deliberately rejects configuring both an authorization
        server provider and a separate token verifier. ``both`` mode still
        needs one verification surface, so the provider itself delegates
        unknown access tokens to the static-token verifier.
        """
        self._store = store
        self._fallback_token_verifier = fallback_token_verifier

    async def _ensure_ready(self) -> None:
        """Open the backing DB on first use.

        The provider is typically constructed at module import time with
        an unopened ``DebateStore`` — we cannot ``await`` at import, so
        each method entry point calls this helper. ``DebateStore.initialize``
        is idempotent so repeated calls are cheap.
        """
        await self._store.initialize()

    # ------------------------------------------------------------------
    # Client registration / lookup
    # ------------------------------------------------------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        await self._ensure_ready()
        row = await self._store.get_oauth_client(client_id)
        if row is None:
            return None
        return OAuthClientInformationFull(
            client_id=row["client_id"],
            client_secret=None,  # secrets are stored hashed; never returned
            redirect_uris=row["redirect_uris"],
            grant_types=row["grant_types"],
            response_types=["code"],
            token_endpoint_auth_method=row["token_endpoint_auth_method"],
            client_name=row["client_name"],
        )

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        """Dynamic Client Registration (RFC 7591).

        We require a ``client_id`` set by the caller (FastMCP fills one
        in when the raw request omits it). Redirect URIs must parse as
        strings; the pydantic model stores them as ``AnyUrl``.
        """
        await self._ensure_ready()
        if not client_info.client_id:
            raise RegistrationError(
                error="invalid_client_metadata",
                error_description="client_id is required",
            )
        redirect_uris = [str(u) for u in (client_info.redirect_uris or [])]
        if not redirect_uris:
            raise RegistrationError(
                error="invalid_redirect_uri",
                error_description="at least one redirect_uri is required",
            )
        await self._store.save_oauth_client(
            client_info.client_id,
            redirect_uris=redirect_uris,
            grant_types=list(client_info.grant_types),
            token_endpoint_auth_method=client_info.token_endpoint_auth_method or "none",
            client_name=client_info.client_name,
        )

    # ------------------------------------------------------------------
    # Authorization code flow
    # ------------------------------------------------------------------

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        """Issue an authorization code and return the redirect URL.

        No consent screen — the first slice auto-approves since this is a
        research tool. A real consent page lands in a later slice before
        directory submission.
        """
        await self._ensure_ready()
        code = _new_token()
        scopes = list(params.scopes or ["debate"])
        await self._store.save_oauth_code(
            code,
            client_id=client.client_id or "",
            redirect_uri=str(params.redirect_uri),
            scopes=scopes,
            code_challenge=params.code_challenge,
            code_challenge_method="S256",  # MCP SDK forces S256 upstream
            expires_at=_plus_iso(_CODE_TTL),
        )
        return construct_redirect_uri(
            str(params.redirect_uri),
            code=code,
            state=params.state,
        )

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> AuthorizationCode | None:
        # We do NOT mark the code used here — that happens in
        # ``exchange_authorization_code`` inside a transaction so the
        # full "verify PKCE + atomically consume" sequence is one unit.
        # This method only fetches for the MCP SDK's pre-exchange checks.
        await self._ensure_ready()
        row = await self._store.get_oauth_code_for_load(authorization_code)
        if row is None:
            return None
        if row["client_id"] != client.client_id:
            return None
        return AuthorizationCode(
            code=row["code"],
            scopes=row["scopes"],
            expires_at=float(_iso_to_unix(row["expires_at"])),
            client_id=row["client_id"],
            code_challenge=row["code_challenge"],
            redirect_uri=row["redirect_uri"],
            redirect_uri_provided_explicitly=True,
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        await self._ensure_ready()
        # Atomic consume — second call with the same code returns None.
        consumed = await self._store.consume_oauth_code(authorization_code.code)
        if consumed is None:
            raise TokenError(
                error="invalid_grant",
                error_description="authorization code is expired, invalid, or already used",
            )
        if consumed["client_id"] != client.client_id:
            raise TokenError(
                error="invalid_grant",
                error_description="authorization code was issued to a different client",
            )
        # PKCE verification happens in the FastMCP handlers before this
        # method runs (it compares the request's ``code_verifier`` to the
        # ``code_challenge`` on the ``AuthorizationCode`` we returned
        # from ``load_authorization_code``). The handler then forwards
        # only the unwrapped code here — so consumption + issuance is
        # all this method has to do.
        access_token = _new_token()
        refresh_token = _new_token()
        access_ttl = _ACCESS_TOKEN_TTL
        access_expiry = _plus_iso(access_ttl)
        await self._store.save_oauth_token(
            access_token,
            kind="access",
            client_id=consumed["client_id"],
            scopes=consumed["scopes"],
            expires_at=access_expiry,
        )
        await self._store.save_oauth_token(
            refresh_token,
            kind="refresh",
            client_id=consumed["client_id"],
            scopes=consumed["scopes"],
            expires_at=None,
        )
        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=int(access_ttl.total_seconds()),
            refresh_token=refresh_token,
            scope=" ".join(consumed["scopes"]),
        )

    # ------------------------------------------------------------------
    # Refresh flow
    # ------------------------------------------------------------------

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> RefreshToken | None:
        await self._ensure_ready()
        row = await self._store.get_oauth_token(refresh_token)
        if row is None or row["kind"] != "refresh":
            return None
        if row["client_id"] != client.client_id:
            return None
        return RefreshToken(
            token=row["token"],
            client_id=row["client_id"],
            scopes=row["scopes"],
            expires_at=None,
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        await self._ensure_ready()
        # Requested scopes must be a subset of the refresh token's scopes —
        # widening on refresh would let an attacker escalate.
        original = set(refresh_token.scopes)
        requested = set(scopes) or original
        if not requested.issubset(original):
            raise TokenError(
                error="invalid_scope",
                error_description="requested scopes exceed the original grant",
            )
        # Rotate: revoke old refresh, issue new pair.
        await self._store.revoke_oauth_token(refresh_token.token)
        new_access = _new_token()
        new_refresh = _new_token()
        access_ttl = _ACCESS_TOKEN_TTL
        access_expiry = _plus_iso(access_ttl)
        final_scopes = list(requested)
        await self._store.save_oauth_token(
            new_access,
            kind="access",
            client_id=refresh_token.client_id,
            scopes=final_scopes,
            expires_at=access_expiry,
        )
        await self._store.save_oauth_token(
            new_refresh,
            kind="refresh",
            client_id=refresh_token.client_id,
            scopes=final_scopes,
            expires_at=None,
        )
        return OAuthToken(
            access_token=new_access,
            token_type="Bearer",
            expires_in=int(access_ttl.total_seconds()),
            refresh_token=new_refresh,
            scope=" ".join(final_scopes),
        )

    # ------------------------------------------------------------------
    # Access token resolution / revocation
    # ------------------------------------------------------------------

    async def load_access_token(self, token: str) -> AccessToken | None:
        await self._ensure_ready()
        row = await self._store.get_oauth_token(token)
        if row is not None and row["kind"] == "access":
            return AccessToken(
                token=row["token"],
                client_id=row["client_id"],
                scopes=row["scopes"],
                expires_at=_iso_to_unix(row["expires_at"]) if row["expires_at"] else None,
            )
        if self._fallback_token_verifier is not None:
            return await self._fallback_token_verifier.verify_token(token)
        return None

    async def revoke_token(
        self,
        token: AccessToken | RefreshToken,
    ) -> None:
        """Revoke an access or refresh token.

        Idempotent — revoking an already-revoked or unknown token is
        silently a no-op so operators can retry safely.
        """
        await self._ensure_ready()
        await self._store.revoke_oauth_token(token.token)
