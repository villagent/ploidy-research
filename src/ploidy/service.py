"""DebateService — the service layer.

Owns all in-memory debate state and wraps persistence via DebateStore.
The MCP server (``server.py``) is now a thin tool layer that delegates
to a single DebateService instance; previously these tools manipulated
module-level globals directly, which blocked multi-worker deployment
and complicated testing.

A single event loop owns a single DebateService. For multi-process
deployments, replace the in-memory dicts with a shared state backend
(Redis, etc.) behind the same interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from ploidy.context_firewall import ContextManifest, build_context_manifest
from ploidy.convergence import ConvergenceEngine
from ploidy.exceptions import PloidyError, ProtocolError, SessionError
from ploidy.injection import (
    VALID_INJECTION_MODES,
    VALID_LANGUAGES,
    append_language,
    build_deep_prompt,
)
from ploidy.lockprovider import AsyncLockProvider, LockProvider
from ploidy.metrics import metrics, tenant_label
from ploidy.protocol import DebateMessage, DebatePhase, DebateProtocol, SemanticAction
from ploidy.ratelimit import RateLimitError, TokenBucketLimiter
from ploidy.render import render_debate
from ploidy.session import DeliveryMode, EffortLevel, SessionContext, SessionRole
from ploidy.store import DebateStore
from ploidy.stream import ProgressCallback, emit

logger = logging.getLogger("ploidy.service")


_RECOVERY_ROLE_MAP = {
    "deep": SessionRole.DEEP,
    "experienced": SessionRole.DEEP,
    "semi_fresh": SessionRole.SEMI_FRESH,
    "fresh": SessionRole.FRESH,
}

# Distinct from ``None`` (which is a valid "unscoped" owner) so
# dict.get can differentiate "not cached" from "owned by nobody".
_SENTINEL_UNKNOWN: object = object()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _aggregate_positions(positions: list[str] | tuple[str, ...], role_label: str) -> str:
    if len(positions) == 1:
        return positions[0]
    parts = [
        f"--- {role_label} Session {i + 1}/{len(positions)} ---\n{pos}"
        for i, pos in enumerate(positions)
    ]
    return "\n\n".join(parts)


def _parse_dominant_action(challenge_content: str) -> SemanticAction:
    """Parse the dominant semantic action from a challenge response.

    Counts word-boundary matches for each action keyword; PROPOSE_ALTERNATIVE
    and bare ALTERNATIVE collapse into one pattern so that "PROPOSE_ALTERNATIVE"
    is not double-counted.
    """
    import re

    upper = challenge_content.upper()
    counts = {
        SemanticAction.AGREE: len(re.findall(r"\bAGREE\b", upper)),
        SemanticAction.CHALLENGE: len(re.findall(r"\bCHALLENGE\b", upper)),
        SemanticAction.SYNTHESIZE: len(re.findall(r"\bSYNTHESIZE\b", upper)),
        SemanticAction.PROPOSE_ALTERNATIVE: len(re.findall(r"\b(?:PROPOSE_)?ALTERNATIVE\b", upper)),
    }
    return max(counts, key=counts.get) if any(counts.values()) else SemanticAction.CHALLENGE


class DebateService:
    """Stateful debate orchestrator. Instance-scoped, loop-scoped."""

    def __init__(
        self,
        store: DebateStore | None = None,
        *,
        use_llm_convergence: bool = False,
        max_prompt_len: int = 10000,
        max_content_len: int = 50000,
        max_context_docs: int = 10,
        max_sessions_per_debate: int = 5,
        max_context_tokens: int | None = None,
        rate_limiter: TokenBucketLimiter | None = None,
        lock_provider: LockProvider | None = None,
        retention_days: int = 0,
        retention_interval_seconds: int = 3600,
        retention_vacuum: bool = True,
    ) -> None:
        self.store = store or DebateStore()
        self.use_llm_convergence = use_llm_convergence
        self.max_prompt_len = max_prompt_len
        self.max_content_len = max_content_len
        self.max_context_docs = max_context_docs
        self.max_sessions_per_debate = max_sessions_per_debate
        # Hard ceiling on the combined ``context_documents`` length so a
        # single huge-context debate cannot silently burn $50 of input
        # tokens. None means no cap (research behaviour).
        self.max_context_tokens = max_context_tokens
        self.retention_days = retention_days
        self.retention_interval_seconds = retention_interval_seconds
        self.retention_vacuum = retention_vacuum
        # When no limiter is supplied, fall back to a disabled one so callers
        # can always `await self.rate_limiter.acquire(...)` without branching.
        self.rate_limiter = rate_limiter or TokenBucketLimiter(capacity=0, rate_per_sec=0)
        self.lock_provider: LockProvider = lock_provider or AsyncLockProvider()
        self._retention_task: asyncio.Task[None] | None = None

        self.protocols: dict[str, DebateProtocol] = {}
        self.sessions: dict[str, SessionContext] = {}
        self.debate_sessions: dict[str, list[str]] = {}
        self.session_to_debate: dict[str, str] = {}
        self.paused_debates: dict[str, dict] = {}
        # None entries keep legacy (unscoped) debates accessible to any caller
        # so existing databases stay usable; new rows that carry an owner_id
        # are strictly enforced on every lookup.
        self.debate_owners: dict[str, str | None] = {}

        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.store.initialize()
        await self._recover_state()
        if self.retention_days > 0:
            self._retention_task = asyncio.create_task(self._retention_loop())
        self._initialized = True

    async def shutdown(self) -> None:
        if self._retention_task is not None:
            self._retention_task.cancel()
            try:
                await self._retention_task
            except (asyncio.CancelledError, Exception):
                pass
            self._retention_task = None
        if self.store is not None:
            await self.store.close()
        try:
            await self.lock_provider.close()
        except Exception:
            logger.exception("LockProvider close failed")
        self.protocols.clear()
        self.sessions.clear()
        self.debate_sessions.clear()
        self.session_to_debate.clear()
        self.paused_debates.clear()
        self.debate_owners.clear()
        self._initialized = False

    async def run_retention_once(self) -> int:
        """Purge completed/cancelled debates older than retention_days
        plus expired/consumed OAuth codes and expired/revoked tokens.

        Returns the number of rows removed (debates + OAuth rows). Running
        with ``retention_days <= 0`` keeps debate purging off but still
        sweeps OAuth state — expired codes and tokens have their own
        short-lived expiries and must not pile up in the DB even when the
        operator has not enabled debate retention.
        """
        from datetime import timedelta

        debate_removed = 0
        if self.retention_days > 0:
            cutoff = datetime.now(UTC) - timedelta(days=self.retention_days)
            # SQLite's datetime('now') column uses " " as separator — match it.
            cutoff_iso = cutoff.strftime("%Y-%m-%d %H:%M:%S")
            debate_removed = await self.store.purge_terminal_before(cutoff_iso)
            if debate_removed > 0:
                logger.info(
                    "Retention purged %d terminal debate(s) older than %s",
                    debate_removed,
                    cutoff_iso,
                )

        # OAuth hygiene always runs — short-lived codes (5 min) and
        # expired access tokens (1 hour) need sweeping regardless of the
        # debate retention setting.
        oauth_removed = await self.store.purge_oauth_expired()
        if oauth_removed > 0:
            logger.info("Retention purged %d OAuth row(s)", oauth_removed)

        if debate_removed > 0 and self.retention_vacuum:
            try:
                await self.store.vacuum()
            except Exception:
                logger.exception("VACUUM after retention purge failed")
        return debate_removed + oauth_removed

    async def _retention_loop(self) -> None:
        """Periodic retention task. Sleeps between runs and swallows errors."""
        while True:
            try:
                await asyncio.sleep(self.retention_interval_seconds)
                await self.run_retention_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Retention pass failed; will retry on next tick")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_debate(self, session_id: str) -> str:
        debate_id = self.session_to_debate.get(session_id)
        if debate_id is None:
            raise SessionError(f"No debate found for session {session_id}")
        return debate_id

    def _get_lock(self, debate_id: str):
        """Return an async context manager guarding this debate.

        Delegates to the configured :class:`LockProvider`; local instances
        still key ``asyncio.Lock`` objects atomically via ``dict.setdefault``
        and distributed instances spin a Redis SET NX lock.
        """
        return self.lock_provider.lock(debate_id)

    def _validate_length(self, text: str, max_len: int, field: str) -> None:
        if len(text) > max_len:
            raise ProtocolError(f"{field} exceeds maximum length ({len(text)} > {max_len})")

    def _enforce_context_budget(self, docs: list[str]) -> None:
        """Reject context_documents larger than the configured token ceiling.

        Uses a conservative 4-char-per-token approximation — good enough
        for a cost guardrail, avoids pulling in a tiktoken dependency on
        the hot path.
        """
        if self.max_context_tokens is None or not docs:
            return
        approx_tokens = sum(len(d) for d in docs) // 4
        if approx_tokens > self.max_context_tokens:
            raise ProtocolError(
                f"context_documents total ~{approx_tokens} tokens exceeds "
                f"configured ceiling of {self.max_context_tokens}. "
                "Trim the documents or raise PLOIDY_MAX_CONTEXT_TOKENS."
            )

    def _build_context_manifest(
        self,
        docs: list[str],
        context_sources: list[str] | None,
        blocked_sources: list[str] | None,
    ) -> ContextManifest:
        """Return provenance evidence for loaded context after policy checks."""
        return build_context_manifest(
            docs,
            context_sources=context_sources,
            blocked_sources=blocked_sources,
        )

    def _cleanup_debate(self, debate_id: str) -> None:
        self.protocols.pop(debate_id, None)
        self.debate_owners.pop(debate_id, None)
        # Local providers hold one asyncio.Lock per key and leak memory if we
        # never drop them; Redis provider is keyless and ignores the hint.
        if isinstance(self.lock_provider, AsyncLockProvider):
            self.lock_provider.pop(debate_id)
        session_ids = self.debate_sessions.pop(debate_id, [])
        for sid in session_ids:
            self.sessions.pop(sid, None)
            self.session_to_debate.pop(sid, None)

    async def _acquire_or_count(self, tenant: str, cost: float = 1.0) -> None:
        """Acquire a rate-limit token, recording metrics on rejection."""
        try:
            await self.rate_limiter.acquire(tenant, cost=cost)
        except RateLimitError:
            metrics().rate_limit_rejections.labels(tenant=tenant).inc()
            raise

    @staticmethod
    def _resolve_tenant(tenant: str | None, owner_id: str | None) -> str:
        """Collapse (tenant, owner_id) into the single key used for limits/metrics."""
        return tenant or owner_id or "global"

    async def _provision_sessions(
        self,
        *,
        debate_id: str,
        count: int,
        role: SessionRole,
        persisted_role: str,
        prompt: str,
        context_documents: list[str],
        delivery_mode: DeliveryMode,
        effort: str,
        effort_level: EffortLevel,
        model: str | None,
        prefix: str,
    ) -> list[SessionContext]:
        """Create and persist ``count`` sessions of one role.

        Centralises the shape that ``run_auto`` applied twice (once for
        deep, once for fresh/semi-fresh) — identifier minting, session
        context construction, DB row insertion, and in-memory bookkeeping.
        """
        created: list[SessionContext] = []
        save_kwargs: dict[str, Any] = {
            "delivery_mode": delivery_mode.value,
            "model": model,
            "effort": effort,
        }
        if context_documents:
            save_kwargs["context_documents"] = context_documents
        for i in range(count):
            sid = f"{debate_id}-{prefix}-{i}-{uuid.uuid4().hex[:6]}"
            ctx = SessionContext(
                session_id=sid,
                role=role,
                base_prompt=prompt,
                context_documents=context_documents,
                delivery_mode=delivery_mode,
                effort=effort_level,
                model=model,
            )
            await self.store.save_session(sid, debate_id, persisted_role, prompt, **save_kwargs)
            self.sessions[sid] = ctx
            self.debate_sessions[debate_id].append(sid)
            self.session_to_debate[sid] = debate_id
            created.append(ctx)
        return created

    def _require_owner(self, debate_id: str, caller: str | None) -> None:
        """Raise PloidyError if ``caller`` is not allowed to touch the debate.

        Legacy (unscoped) debates — ``owner_id=None`` — are visible to every
        caller so single-tenant deployments and tests don't break. Once a
        debate has an owner, only that owner passes.
        """
        owner = self.debate_owners.get(debate_id, _SENTINEL_UNKNOWN)
        if owner is _SENTINEL_UNKNOWN:
            # Cache miss — the debate may live only in the database (cold
            # replica, pre-hydration recovery path). Hide the miss behind a
            # not-found error; callers surface the row via explicit store
            # lookups if they need the DB-only path.
            raise PloidyError(f"Debate {debate_id} not found")
        if owner is None:
            return
        if caller != owner:
            raise PloidyError(f"Debate {debate_id} not found")

    async def _delete_failed_debate(self, debate_id: str) -> None:
        self._cleanup_debate(debate_id)
        try:
            await self.store.delete_debate(debate_id)
        except Exception:
            logger.exception("Failed to clean up partially created debate %s", debate_id)

    def _hydrate_session(self, s: dict) -> SessionContext:
        role = _RECOVERY_ROLE_MAP.get(s["role"])
        if role is None:
            logger.warning(
                "Unknown session role %r for session %s; defaulting to FRESH",
                s.get("role"),
                s.get("id"),
            )
            role = SessionRole.FRESH
        try:
            delivery_mode = DeliveryMode(s.get("delivery_mode", "none"))
        except ValueError:
            delivery_mode = DeliveryMode.NONE
        try:
            effort_level = EffortLevel(s.get("effort", "high"))
        except ValueError:
            effort_level = EffortLevel.HIGH
        return SessionContext(
            session_id=s["id"],
            role=role,
            base_prompt=s["base_prompt"],
            context_documents=s.get("context_documents", []),
            delivery_mode=delivery_mode,
            effort=effort_level,
            compressed_summary=s.get("compressed_summary"),
            model=s.get("model"),
            metadata=s.get("metadata", {}),
        )

    async def _recover_state(self) -> None:
        """Rebuild in-memory state from persisted rows (active + paused)."""
        active_debates = await self.store.list_active_debates()
        recovered = 0
        for debate in active_debates:
            debate_id = debate["id"]
            if debate_id in self.protocols:
                continue

            protocol = DebateProtocol(debate_id, debate["prompt"])
            sessions = await self.store.get_sessions(debate_id)
            messages = await self.store.get_messages(debate_id)

            session_ids = []
            for s in sessions:
                ctx = self._hydrate_session(s)
                self.sessions[ctx.session_id] = ctx
                self.session_to_debate[ctx.session_id] = debate_id
                session_ids.append(ctx.session_id)

            phase_order = list(DebatePhase)
            for m in messages:
                phase = DebatePhase(m["phase"])
                advances = 0
                while protocol.phase != phase and advances < len(phase_order):
                    try:
                        protocol.advance_phase()
                        advances += 1
                    except ProtocolError:
                        logger.warning(
                            "Cannot advance to %s during recovery of debate %s",
                            phase.value,
                            debate_id,
                        )
                        break
                action = SemanticAction(m["action"]) if m["action"] else None
                msg = DebateMessage(
                    session_id=m["session_id"],
                    phase=phase,
                    content=m["content"],
                    timestamp=m["timestamp"] or _now(),
                    action=action,
                )
                protocol.messages.append(msg)

            if protocol.phase == DebatePhase.POSITION:
                position_sessions = {
                    m.session_id for m in protocol.messages if m.phase == DebatePhase.POSITION
                }
                if len(session_ids) >= 2 and set(session_ids) <= position_sessions:
                    try:
                        protocol.advance_phase()
                    except ProtocolError:
                        pass

            self.protocols[debate_id] = protocol
            self.debate_sessions[debate_id] = session_ids
            self.debate_owners[debate_id] = debate.get("owner_id")
            recovered += 1

        if recovered:
            logger.info("Recovered %d active debate(s) from database", recovered)

        paused_debates = await self.store.list_paused_debates()
        paused_recovered = 0
        for debate in paused_debates:
            debate_id = debate["id"]
            if debate_id in self.paused_debates:
                continue

            paused_ctx_raw = debate.get("paused_context")
            if not paused_ctx_raw:
                logger.warning("Paused debate %s has no persisted context, skipping", debate_id)
                continue

            paused_ctx = (
                json.loads(paused_ctx_raw) if isinstance(paused_ctx_raw, str) else paused_ctx_raw
            )

            protocol = DebateProtocol(debate_id, debate["prompt"])
            saved_phase = paused_ctx.get("protocol_phase", "position")
            phase_order = [p.value for p in DebatePhase]
            target_idx = phase_order.index(saved_phase) if saved_phase in phase_order else 1
            for _ in range(target_idx):
                try:
                    protocol.advance_phase()
                except ProtocolError:
                    break

            messages = await self.store.get_messages(debate_id)
            for m in messages:
                action = SemanticAction(m["action"]) if m["action"] else None
                msg = DebateMessage(
                    session_id=m["session_id"],
                    phase=DebatePhase(m["phase"]),
                    content=m["content"],
                    timestamp=m["timestamp"] or _now(),
                    action=action,
                )
                protocol.messages.append(msg)

            sessions = await self.store.get_sessions(debate_id)
            session_ids = []
            for s in sessions:
                ctx = self._hydrate_session(s)
                self.sessions[ctx.session_id] = ctx
                self.session_to_debate[ctx.session_id] = debate_id
                session_ids.append(ctx.session_id)

            self.protocols[debate_id] = protocol
            self.debate_sessions[debate_id] = session_ids
            self.debate_owners[debate_id] = debate.get("owner_id")
            self.paused_debates[debate_id] = paused_ctx
            paused_recovered += 1

        if paused_recovered:
            logger.info("Recovered %d paused debate(s) from database", paused_recovered)

    # ------------------------------------------------------------------
    # Tool methods
    # ------------------------------------------------------------------

    async def start_debate(
        self,
        prompt: str,
        context_documents: list[str] | None = None,
        *,
        context_sources: list[str] | None = None,
        blocked_sources: list[str] | None = None,
        tenant: str | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        tenant = self._resolve_tenant(tenant, owner_id)
        await self._acquire_or_count(tenant)
        self._validate_length(prompt, self.max_prompt_len, "prompt")
        docs = context_documents or []
        if len(docs) > self.max_context_docs:
            raise ProtocolError(
                f"Too many context documents ({len(docs)} > {self.max_context_docs})"
            )
        for i, doc in enumerate(docs):
            self._validate_length(doc, self.max_content_len, f"context_documents[{i}]")
        self._enforce_context_budget(docs)
        manifest = self._build_context_manifest(docs, context_sources, blocked_sources)

        debate_id = uuid.uuid4().hex[:12]
        config = {"context_manifest": manifest.as_dict()}
        await self.store.save_debate(debate_id, prompt, config=config, owner_id=owner_id)

        deep_id = f"{debate_id}-deep-{uuid.uuid4().hex[:6]}"
        deep_ctx = SessionContext(
            session_id=deep_id,
            role=SessionRole.DEEP,
            base_prompt=prompt,
            context_documents=docs,
        )
        await self.store.save_session(
            deep_id,
            debate_id,
            "deep",
            prompt,
            context_documents=docs,
            delivery_mode=deep_ctx.delivery_mode.value,
            compressed_summary=deep_ctx.compressed_summary,
            metadata=deep_ctx.metadata,
        )

        self.sessions[deep_id] = deep_ctx
        self.debate_sessions[debate_id] = [deep_id]
        self.session_to_debate[deep_id] = debate_id

        protocol = DebateProtocol(debate_id, prompt)
        self.protocols[debate_id] = protocol
        self.debate_owners[debate_id] = owner_id

        metrics().debate_started.labels(tenant=tenant_label(owner_id), mode="two_terminal").inc()
        logger.info("Debate started: %s by session %s", debate_id, deep_id)

        return {
            "debate_id": debate_id,
            "session_id": deep_id,
            "role": "deep",
            "phase": protocol.phase.value,
            "prompt": prompt,
            "context_manifest": manifest.as_dict(),
            "message": f"Debate created. Share this debate_id with the Fresh session: {debate_id}",
        }

    async def join_debate(
        self,
        debate_id: str,
        role: str = "fresh",
        delivery_mode: str = "none",
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        protocol = self.protocols.get(debate_id)
        if protocol is None:
            raise PloidyError(f"Debate {debate_id} not found")
        self._require_owner(debate_id, owner_id)

        current_count = len(self.debate_sessions.get(debate_id, []))
        if current_count >= self.max_sessions_per_debate:
            raise ProtocolError(
                f"Debate already has {current_count} sessions (max {self.max_sessions_per_debate})"
            )

        role_map = {"fresh": SessionRole.FRESH, "semi_fresh": SessionRole.SEMI_FRESH}
        session_role = role_map.get(role)
        if session_role is None:
            raise ProtocolError(f"Invalid role '{role}'. Must be 'fresh' or 'semi_fresh'")

        dm_map = {
            "none": DeliveryMode.NONE,
            "passive": DeliveryMode.PASSIVE,
            "active": DeliveryMode.ACTIVE,
        }
        dm = dm_map.get(delivery_mode, DeliveryMode.NONE)

        prefix = "sf" if session_role == SessionRole.SEMI_FRESH else "fresh"
        sid = f"{debate_id}-{prefix}-{uuid.uuid4().hex[:6]}"
        ctx = SessionContext(
            session_id=sid,
            role=session_role,
            base_prompt=protocol.prompt,
            context_documents=[],
            delivery_mode=dm,
        )
        await self.store.save_session(
            sid,
            debate_id,
            role,
            protocol.prompt,
            context_documents=ctx.context_documents,
            delivery_mode=ctx.delivery_mode.value,
            compressed_summary=ctx.compressed_summary,
            metadata=ctx.metadata,
        )

        self.sessions[sid] = ctx
        self.debate_sessions[debate_id].append(sid)
        self.session_to_debate[sid] = debate_id

        logger.info("Session %s joined debate %s as %s", sid, debate_id, role)

        return {
            "debate_id": debate_id,
            "session_id": sid,
            "role": role,
            "delivery_mode": delivery_mode,
            "phase": protocol.phase.value,
            "prompt": protocol.prompt,
        }

    async def submit_position(
        self, session_id: str, content: str, *, owner_id: str | None = None
    ) -> dict[str, Any]:
        self._validate_length(content, self.max_content_len, "content")

        if session_id not in self.sessions:
            raise SessionError(f"Session {session_id} not found")

        debate_id = self._find_debate(session_id)
        self._require_owner(debate_id, owner_id)
        lock = self._get_lock(debate_id)

        async with lock:
            protocol = self.protocols[debate_id]

            if protocol.phase == DebatePhase.INDEPENDENT:
                protocol.advance_phase()

            if protocol.phase != DebatePhase.POSITION:
                raise ProtocolError(f"Cannot submit position in phase {protocol.phase.value}")

            msg = DebateMessage(
                session_id=session_id,
                phase=DebatePhase.POSITION,
                content=content,
                timestamp=_now(),
            )
            protocol.submit_message(msg)
            await self.store.save_message(debate_id, session_id, "position", content)
            metrics().messages_recorded.labels(
                tenant=tenant_label(self.debate_owners.get(debate_id)),
                phase="position",
            ).inc()

            session_ids = set(self.debate_sessions[debate_id])
            submitted = {m.session_id for m in protocol.messages if m.phase == DebatePhase.POSITION}
            all_in = len(session_ids) >= 2 and session_ids <= submitted

            if all_in:
                protocol.advance_phase()

        logger.info("Position from %s in debate %s (all_in=%s)", session_id, debate_id, all_in)

        return {
            "session_id": session_id,
            "debate_id": debate_id,
            "phase": protocol.phase.value,
            "status": "recorded",
            "content_length": len(content),
            "all_positions_in": all_in,
        }

    async def submit_challenge(
        self,
        session_id: str,
        content: str,
        action: str = "challenge",
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        self._validate_length(content, self.max_content_len, "content")

        if session_id not in self.sessions:
            raise SessionError(f"Session {session_id} not found")

        debate_id = self._find_debate(session_id)
        self._require_owner(debate_id, owner_id)
        lock = self._get_lock(debate_id)

        async with lock:
            protocol = self.protocols[debate_id]

            if protocol.phase != DebatePhase.CHALLENGE:
                raise ProtocolError(f"Cannot submit challenge in phase {protocol.phase.value}")

            try:
                semantic_action = SemanticAction(action)
            except ValueError:
                raise ProtocolError(
                    "Invalid action. Must be one of: "
                    "agree, challenge, propose_alternative, synthesize"
                )

            msg = DebateMessage(
                session_id=session_id,
                phase=DebatePhase.CHALLENGE,
                content=content,
                timestamp=_now(),
                action=semantic_action,
            )
            protocol.submit_message(msg)
            await self.store.save_message(debate_id, session_id, "challenge", content, action)
            metrics().messages_recorded.labels(
                tenant=tenant_label(self.debate_owners.get(debate_id)),
                phase="challenge",
            ).inc()

        logger.info("Challenge from %s in debate %s (action=%s)", session_id, debate_id, action)

        return {
            "session_id": session_id,
            "debate_id": debate_id,
            "phase": protocol.phase.value,
            "action": action,
            "status": "recorded",
            "content_length": len(content),
        }

    async def converge(self, debate_id: str, *, owner_id: str | None = None) -> dict[str, Any]:
        protocol = self.protocols.get(debate_id)
        if protocol is None:
            raise PloidyError(f"Debate {debate_id} not found")
        self._require_owner(debate_id, owner_id)
        tenant = tenant_label(self.debate_owners.get(debate_id))

        lock = self._get_lock(debate_id)

        async with lock:
            if protocol.phase != DebatePhase.CHALLENGE:
                raise ProtocolError(
                    f"Cannot converge from phase {protocol.phase.value}, must be in CHALLENGE"
                )

            protocol.advance_phase()  # → CONVERGENCE

            engine = ConvergenceEngine(use_llm=self.use_llm_convergence)
            session_roles = {
                sid: self.sessions[sid].role.value.capitalize()
                for sid in self.debate_sessions.get(debate_id, [])
                if sid in self.sessions
            }
            started = time.perf_counter()
            result = await engine.analyze(protocol, session_roles)
            metrics().convergence_duration.labels(tenant=tenant, mode="two_terminal").observe(
                time.perf_counter() - started
            )

            protocol.advance_phase()  # → COMPLETE

        points_json = json.dumps(
            [
                {
                    "category": p.category,
                    "summary": p.summary,
                    "session_a_view": p.session_a_view,
                    "session_b_view": p.session_b_view,
                    "resolution": p.resolution,
                }
                for p in result.points
            ]
        )
        await self.store.save_convergence_and_complete(
            debate_id, result.synthesis, result.confidence, points_json
        )

        self._cleanup_debate(debate_id)
        metrics().debate_completed.labels(tenant=tenant, mode="two_terminal").inc()

        logger.info(
            "Debate %s converged (confidence=%.2f, points=%d)",
            debate_id,
            result.confidence,
            len(result.points),
        )

        # Two-terminal flow — pull transcript off the protocol for the
        # same render treatment every other mode gets.
        deep_sids = [sid for sid in session_roles if session_roles[sid] == "Deep"]
        fresh_sids = [sid for sid in session_roles if session_roles[sid] != "Deep"]
        msgs_by_sp: dict[tuple[str, DebatePhase], str] = {}
        for msg in protocol.messages:
            msgs_by_sp[(msg.session_id, msg.phase)] = msg.content
        deep_positions_text = [msgs_by_sp.get((sid, DebatePhase.POSITION), "") for sid in deep_sids]
        fresh_positions_text = [
            msgs_by_sp.get((sid, DebatePhase.POSITION), "") for sid in fresh_sids
        ]
        deep_challenge_text = next(
            (msgs_by_sp.get((sid, DebatePhase.CHALLENGE)) for sid in deep_sids), None
        )
        fresh_challenge_text = next(
            (msgs_by_sp.get((sid, DebatePhase.CHALLENGE)) for sid in fresh_sids), None
        )
        rendered_markdown = render_debate(
            prompt=protocol.prompt,
            deep_label="Deep",
            fresh_label=(session_roles[fresh_sids[0]] if fresh_sids else "Fresh"),
            deep_positions=deep_positions_text,
            fresh_positions=fresh_positions_text,
            deep_challenge=deep_challenge_text,
            fresh_challenge=fresh_challenge_text,
            points=result.points,
            synthesis=result.synthesis,
            confidence=result.confidence,
            debate_id=debate_id,
            mode="two_terminal",
        )

        return {
            "debate_id": debate_id,
            "phase": "complete",
            "synthesis": result.synthesis,
            "rendered_markdown": rendered_markdown,
            "confidence": result.confidence,
            "points": [
                {
                    "category": p.category,
                    "summary": p.summary,
                    "resolution": p.resolution,
                }
                for p in result.points
            ],
        }

    async def cancel(self, debate_id: str, *, owner_id: str | None = None) -> dict[str, Any]:
        protocol = self.protocols.get(debate_id)
        if protocol is None:
            raise PloidyError(f"Debate {debate_id} not found")
        self._require_owner(debate_id, owner_id)

        if protocol.phase == DebatePhase.COMPLETE:
            raise ProtocolError("Cannot cancel a completed debate")

        tenant = tenant_label(self.debate_owners.get(debate_id))
        await self.store.update_debate_status(debate_id, "cancelled")
        self._cleanup_debate(debate_id)
        metrics().debate_cancelled.labels(tenant=tenant, outcome="cancelled").inc()

        logger.info("Debate %s cancelled", debate_id)

        return {"debate_id": debate_id, "status": "cancelled"}

    async def delete(self, debate_id: str, *, owner_id: str | None = None) -> dict[str, Any]:
        debate = await self.store.get_debate(debate_id, owner_id=owner_id)
        if debate is None:
            raise PloidyError(f"Debate {debate_id} not found")

        self._cleanup_debate(debate_id)
        await self.store.delete_debate(debate_id)

        logger.info("Debate %s permanently deleted", debate_id)

        return {"debate_id": debate_id, "status": "deleted"}

    async def status(self, debate_id: str, *, owner_id: str | None = None) -> dict[str, Any]:
        protocol = self.protocols.get(debate_id)
        if protocol is None:
            raise PloidyError(f"Debate {debate_id} not found")
        self._require_owner(debate_id, owner_id)

        session_ids = self.debate_sessions.get(debate_id, [])
        sessions_info = []
        for sid in session_ids:
            ctx = self.sessions.get(sid)
            if ctx:
                sessions_info.append({"session_id": sid, "role": ctx.role.value})

        messages_by_phase: dict[str, list[dict]] = {}
        for msg in protocol.messages:
            phase = msg.phase.value
            messages_by_phase.setdefault(phase, []).append(
                {
                    "session_id": msg.session_id,
                    "content": msg.content,
                    "action": msg.action.value if msg.action else None,
                    "timestamp": msg.timestamp,
                }
            )

        return {
            "debate_id": debate_id,
            "phase": protocol.phase.value,
            "prompt": protocol.prompt,
            "message_count": len(protocol.messages),
            "sessions": sessions_info,
            "messages": messages_by_phase,
        }

    async def history(self, limit: int = 50, *, owner_id: str | None = None) -> dict[str, Any]:
        clamped = min(max(limit, 1), 200)
        debates = await self.store.list_debates(clamped, owner_id=owner_id)
        return {"debates": debates, "total": len(debates), "limit": clamped}

    # ------------------------------------------------------------------
    # Multi-step operations: run_solo, run_auto, review
    # ------------------------------------------------------------------

    async def run_solo(
        self,
        prompt: str,
        deep_position: str,
        fresh_position: str,
        deep_challenge: str | None = None,
        fresh_challenge: str | None = None,
        context_documents: list[str] | None = None,
        context_sources: list[str] | None = None,
        blocked_sources: list[str] | None = None,
        deep_label: str = "Deep",
        fresh_label: str = "Fresh",
        *,
        tenant: str | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        tenant = self._resolve_tenant(tenant, owner_id)
        await self._acquire_or_count(tenant)
        self._validate_length(prompt, self.max_prompt_len, "prompt")
        self._validate_length(deep_position, self.max_content_len, "deep_position")
        self._validate_length(fresh_position, self.max_content_len, "fresh_position")
        if deep_challenge is not None:
            self._validate_length(deep_challenge, self.max_content_len, "deep_challenge")
        if fresh_challenge is not None:
            self._validate_length(fresh_challenge, self.max_content_len, "fresh_challenge")

        docs = context_documents or []
        if len(docs) > self.max_context_docs:
            raise ProtocolError(
                f"Too many context documents ({len(docs)} > {self.max_context_docs})"
            )
        for i, doc in enumerate(docs):
            self._validate_length(doc, self.max_content_len, f"context_documents[{i}]")
        self._enforce_context_budget(docs)
        manifest = self._build_context_manifest(docs, context_sources, blocked_sources)

        debate_id = uuid.uuid4().hex[:12]
        config = {
            "mode": "solo",
            "deep_label": deep_label,
            "fresh_label": fresh_label,
            "context_manifest": manifest.as_dict(),
        }
        metrics().debate_started.labels(tenant=tenant_label(owner_id), mode="solo").inc()

        try:
            async with self.store.transaction():
                await self.store.save_debate(debate_id, prompt, config=config, owner_id=owner_id)

                deep_id = f"{debate_id}-deep-{uuid.uuid4().hex[:6]}"
                fresh_id = f"{debate_id}-fresh-{uuid.uuid4().hex[:6]}"
                deep_ctx = SessionContext(
                    session_id=deep_id,
                    role=SessionRole.DEEP,
                    base_prompt=prompt,
                    context_documents=docs,
                    delivery_mode=DeliveryMode.PASSIVE,
                )
                fresh_ctx = SessionContext(
                    session_id=fresh_id,
                    role=SessionRole.FRESH,
                    base_prompt=prompt,
                    context_documents=[],
                    delivery_mode=DeliveryMode.NONE,
                )
                await self.store.save_session(
                    deep_id,
                    debate_id,
                    SessionRole.DEEP.value,
                    prompt,
                    context_documents=docs,
                    delivery_mode=deep_ctx.delivery_mode.value,
                )
                await self.store.save_session(
                    fresh_id,
                    debate_id,
                    SessionRole.FRESH.value,
                    prompt,
                    context_documents=[],
                    delivery_mode=fresh_ctx.delivery_mode.value,
                )
                self.sessions[deep_id] = deep_ctx
                self.sessions[fresh_id] = fresh_ctx
                self.debate_sessions[debate_id] = [deep_id, fresh_id]
                self.session_to_debate[deep_id] = debate_id
                self.session_to_debate[fresh_id] = debate_id

                protocol = DebateProtocol(debate_id, prompt)
                self.protocols[debate_id] = protocol
                self.debate_owners[debate_id] = owner_id

                protocol.advance_phase()
                for sid, content in ((deep_id, deep_position), (fresh_id, fresh_position)):
                    msg = DebateMessage(
                        session_id=sid,
                        phase=DebatePhase.POSITION,
                        content=content,
                        timestamp=_now(),
                    )
                    protocol.submit_message(msg)
                    await self.store.save_message(
                        debate_id, sid, DebatePhase.POSITION.value, content
                    )

                protocol.advance_phase()

                for sid, content in (
                    (deep_id, deep_challenge),
                    (fresh_id, fresh_challenge),
                ):
                    if not content:
                        continue
                    action = _parse_dominant_action(content)
                    ch_msg = DebateMessage(
                        session_id=sid,
                        phase=DebatePhase.CHALLENGE,
                        content=content,
                        timestamp=_now(),
                        action=action,
                    )
                    protocol.submit_message(ch_msg)
                    await self.store.save_message(
                        debate_id, sid, DebatePhase.CHALLENGE.value, content, action.value
                    )

                protocol.advance_phase()

                engine = ConvergenceEngine(use_llm=self.use_llm_convergence)
                session_roles = {deep_id: deep_label, fresh_id: fresh_label}
                result = await engine.analyze(protocol, session_roles)

                protocol.advance_phase()

                points_json = json.dumps(
                    [
                        {
                            "category": p.category,
                            "summary": p.summary,
                            "session_a_view": p.session_a_view,
                            "session_b_view": p.session_b_view,
                            "resolution": p.resolution,
                            "root_cause": p.root_cause,
                        }
                        for p in result.points
                    ]
                )
                await self.store.save_convergence_and_complete(
                    debate_id,
                    result.synthesis,
                    result.confidence,
                    points_json,
                    meta_analysis=result.meta_analysis,
                )
            self._cleanup_debate(debate_id)
        except Exception:
            await self._delete_failed_debate(debate_id)
            raise

        metrics().debate_completed.labels(tenant=tenant_label(owner_id), mode="solo").inc()
        logger.info(
            "Solo debate %s complete (confidence=%.2f, challenges=%d)",
            debate_id,
            result.confidence,
            sum(1 for c in (deep_challenge, fresh_challenge) if c),
        )

        rendered_markdown = render_debate(
            prompt=prompt,
            deep_label=deep_label,
            fresh_label=fresh_label,
            deep_positions=[deep_position],
            fresh_positions=[fresh_position],
            deep_challenge=deep_challenge,
            fresh_challenge=fresh_challenge,
            points=result.points,
            synthesis=result.synthesis,
            confidence=result.confidence,
            meta_analysis=result.meta_analysis,
            debate_id=debate_id,
            mode="solo",
        )

        return {
            "debate_id": debate_id,
            "phase": "complete",
            "mode": "solo",
            "config": config,
            "context_manifest": manifest.as_dict(),
            "synthesis": result.synthesis,
            "rendered_markdown": rendered_markdown,
            "confidence": result.confidence,
            "meta_analysis": result.meta_analysis,
            "points": [
                {
                    "category": p.category,
                    "summary": p.summary,
                    "resolution": p.resolution,
                    "root_cause": p.root_cause,
                }
                for p in result.points
            ],
        }

    async def run_auto(
        self,
        prompt: str,
        context_documents: list[str] | None = None,
        context_sources: list[str] | None = None,
        blocked_sources: list[str] | None = None,
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
        *,
        tenant: str | None = None,
        owner_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        tenant = self._resolve_tenant(tenant, owner_id)
        await self._acquire_or_count(tenant, cost=float(deep_n + fresh_n))
        # ── Validate inputs ─────────────────────────────────────────────
        self._validate_length(prompt, self.max_prompt_len, "prompt")
        docs = context_documents or []
        if len(docs) > self.max_context_docs:
            raise ProtocolError(
                f"Too many context documents ({len(docs)} > {self.max_context_docs})"
            )
        for i, doc in enumerate(docs):
            self._validate_length(doc, self.max_content_len, f"context_documents[{i}]")
        self._enforce_context_budget(docs)
        manifest = self._build_context_manifest(docs, context_sources, blocked_sources)

        role_map = {"fresh": SessionRole.FRESH, "semi_fresh": SessionRole.SEMI_FRESH}
        auto_role = role_map.get(fresh_role)
        if auto_role is None:
            raise ProtocolError(
                f"Invalid fresh_role '{fresh_role}'. Must be 'fresh' or 'semi_fresh'"
            )

        dm_map = {
            "none": DeliveryMode.NONE,
            "passive": DeliveryMode.PASSIVE,
            "active": DeliveryMode.ACTIVE,
            "selective": DeliveryMode.SELECTIVE,
        }
        dm = dm_map.get(delivery_mode)
        if dm is None:
            raise ProtocolError(
                f"Invalid delivery_mode '{delivery_mode}'. "
                "Must be 'none', 'passive', 'active', or 'selective'"
            )
        if auto_role == SessionRole.FRESH and dm != DeliveryMode.NONE:
            raise ProtocolError("Fresh auto sessions must use delivery_mode='none'")
        if auto_role == SessionRole.SEMI_FRESH and dm == DeliveryMode.NONE:
            raise ProtocolError(
                "Semi-fresh auto sessions must use 'passive', 'active', or 'selective'"
            )

        if pause_at not in {None, "challenge", "convergence"}:
            raise ProtocolError(
                f"Invalid pause_at '{pause_at}'. Must be 'challenge' or 'convergence'"
            )
        if deep_n < 1 or fresh_n < 1:
            raise ProtocolError("deep_n and fresh_n must be >= 1")
        if deep_n + fresh_n > self.max_sessions_per_debate:
            raise ProtocolError(
                f"Total sessions ({deep_n}+{fresh_n}) exceeds max ({self.max_sessions_per_debate})"
            )
        try:
            effort_level = EffortLevel(effort)
        except ValueError:
            raise ProtocolError(f"Invalid effort '{effort}'. Must be low/medium/high/max")
        if injection_mode not in VALID_INJECTION_MODES:
            valid = sorted(VALID_INJECTION_MODES)
            raise ProtocolError(
                f"Invalid injection_mode '{injection_mode}'. Must be one of {valid}"
            )
        if not (0 <= context_pct <= 100):
            raise ProtocolError("context_pct must be 0..100")
        if language not in VALID_LANGUAGES:
            raise ProtocolError(
                f"Invalid language '{language}'. Must be one of {sorted(VALID_LANGUAGES)}"
            )

        try:
            from ploidy.api_client import (
                compress_failures_only,
                compress_position,
                generate_challenge,
                generate_experienced_position,
                generate_fresh_position,
                generate_semi_fresh_position,
                is_api_available,
            )
        except ImportError:
            raise PloidyError("API client not available. Install with: pip install ploidy[api]")

        if not is_api_available():
            raise PloidyError(
                "API not configured. Set PLOIDY_API_BASE_URL (or provide "
                "ANTHROPIC_API_KEY to auto-configure the Anthropic endpoint)."
            )

        raw_context = "\n\n".join(docs) if docs else ""
        deep_user_prompt, deep_sys_prompt = build_deep_prompt(
            raw_context, prompt, mode=injection_mode, context_pct=context_pct
        )
        deep_user_prompt = append_language(deep_user_prompt, language)
        fresh_prompt = append_language(prompt, language)

        config = {
            "deep_n": deep_n,
            "fresh_n": fresh_n,
            "effort": effort,
            "injection_mode": injection_mode,
            "context_pct": context_pct,
            "language": language,
            "deep_model": deep_model,
            "fresh_model": fresh_model,
            "fresh_role": fresh_role,
            "delivery_mode": delivery_mode,
            "context_manifest": manifest.as_dict(),
        }

        debate_id = uuid.uuid4().hex[:12]
        await self.store.save_debate(debate_id, prompt, config=config, owner_id=owner_id)
        metrics().debate_started.labels(tenant=tenant_label(owner_id), mode="auto").inc()

        protocol = DebateProtocol(debate_id, prompt)
        self.protocols[debate_id] = protocol
        self.debate_sessions[debate_id] = []
        self.debate_owners[debate_id] = owner_id

        deep_sessions = await self._provision_sessions(
            debate_id=debate_id,
            count=deep_n,
            role=SessionRole.DEEP,
            persisted_role="deep",
            prompt=prompt,
            context_documents=docs,
            delivery_mode=DeliveryMode.PASSIVE,
            effort=effort,
            effort_level=effort_level,
            model=deep_model,
            prefix="deep",
        )
        fresh_sessions = await self._provision_sessions(
            debate_id=debate_id,
            count=fresh_n,
            role=auto_role,
            persisted_role=fresh_role,
            prompt=prompt,
            context_documents=[],
            delivery_mode=dm,
            effort=effort,
            effort_level=effort_level,
            model=fresh_model,
            prefix="sf" if auto_role == SessionRole.SEMI_FRESH else "fresh",
        )

        logger.info(
            "Auto-debate %s: Deep(%d) x %s(%d), effort=%s, injection=%s",
            debate_id,
            deep_n,
            fresh_role,
            fresh_n,
            effort,
            injection_mode,
        )

        try:
            protocol.advance_phase()  # → POSITION
            await emit(
                progress,
                "phase_started",
                phase="position",
                debate_id=debate_id,
                deep_n=deep_n,
                fresh_n=fresh_n,
            )

            deep_tasks = [
                generate_experienced_position(
                    deep_user_prompt,
                    context_documents=(
                        None if injection_mode != "raw" or context_pct < 100 else docs
                    ),
                    effort=effort,
                    model=deep_model,
                )
                for _ in range(deep_n)
            ]
            deep_positions = await asyncio.gather(*deep_tasks)
            await emit(
                progress,
                "positions_generated",
                side="deep",
                count=len(deep_positions),
                previews=[p[:300] for p in deep_positions],
            )

            compressed = None
            if auto_role == SessionRole.SEMI_FRESH:
                deep_aggregate = _aggregate_positions(deep_positions, "Deep")
                if delivery_mode == "selective":
                    compressed = await compress_failures_only(deep_aggregate, model=deep_model)
                else:
                    compressed = await compress_position(deep_aggregate, model=deep_model)
                for ctx in fresh_sessions:
                    ctx.compressed_summary = compressed
                    await self.store.update_session_context(
                        ctx.session_id, compressed_summary=compressed
                    )

            fresh_tasks = []
            for _ in range(fresh_n):
                if auto_role == SessionRole.SEMI_FRESH and compressed:
                    fresh_tasks.append(
                        generate_semi_fresh_position(
                            fresh_prompt,
                            compressed,
                            delivery_mode=delivery_mode,
                            effort=effort,
                            model=fresh_model,
                        )
                    )
                else:
                    fresh_tasks.append(
                        generate_fresh_position(fresh_prompt, effort=effort, model=fresh_model)
                    )
            fresh_positions = await asyncio.gather(*fresh_tasks)
            await emit(
                progress,
                "positions_generated",
                side="fresh",
                count=len(fresh_positions),
                previews=[p[:300] for p in fresh_positions],
            )

            # Batch every position insert into one SQLite commit instead of
            # 2N fsyncs. aiosqlite serialises writes on its single
            # connection, so the wrapping transaction is the real win.
            async with self.store.transaction():
                for ctx, pos in zip(deep_sessions, deep_positions):
                    msg = DebateMessage(
                        session_id=ctx.session_id,
                        phase=DebatePhase.POSITION,
                        content=pos,
                        timestamp=_now(),
                    )
                    protocol.submit_message(msg)
                    await self.store.save_message(debate_id, ctx.session_id, "position", pos)

                for ctx, pos in zip(fresh_sessions, fresh_positions):
                    msg = DebateMessage(
                        session_id=ctx.session_id,
                        phase=DebatePhase.POSITION,
                        content=pos,
                        timestamp=_now(),
                    )
                    protocol.submit_message(msg)
                    await self.store.save_message(debate_id, ctx.session_id, "position", pos)

            if pause_at == "challenge":
                paused_ctx = {
                    "deep_ids": [s.session_id for s in deep_sessions],
                    "fresh_ids": [s.session_id for s in fresh_sessions],
                    "deep_positions": list(deep_positions),
                    "fresh_positions": list(fresh_positions),
                    "fresh_role": fresh_role,
                    "delivery_mode": delivery_mode,
                    "effort": effort,
                    "deep_model": deep_model,
                    "fresh_model": fresh_model,
                    "paused_phase": "challenge",
                    "protocol_phase": protocol.phase.value,
                }
                self.paused_debates[debate_id] = paused_ctx
                await self.store.update_debate_status(debate_id, "paused")
                await self.store.save_paused_context(debate_id, paused_ctx)
                return {
                    "debate_id": debate_id,
                    "phase": "paused",
                    "paused_before": "challenge",
                    "mode": "auto_hitl",
                    "config": config,
                    "context_manifest": manifest.as_dict(),
                    "positions": {
                        "deep": [p[:500] for p in deep_positions],
                        "fresh": [p[:500] for p in fresh_positions],
                    },
                    "message": "Debate paused for human review. Use debate_review to continue.",
                }

            protocol.advance_phase()  # → CHALLENGE
            await emit(progress, "phase_started", phase="challenge", debate_id=debate_id)

            deep_aggregate = _aggregate_positions(deep_positions, "Deep")
            fresh_aggregate = _aggregate_positions(
                fresh_positions, fresh_role.replace("_", "-").title()
            )

            # Challenges run concurrently — each LLM call is independent.
            deep_challenge, fresh_challenge = await asyncio.gather(
                generate_challenge(
                    own_position=deep_aggregate,
                    other_position=fresh_aggregate,
                    own_role="deep",
                    other_role=fresh_role,
                    effort=effort,
                    model=deep_model,
                ),
                generate_challenge(
                    own_position=fresh_aggregate,
                    other_position=deep_aggregate,
                    own_role=fresh_role,
                    other_role="deep",
                    effort=effort,
                    model=fresh_model,
                ),
            )

            deep_action = _parse_dominant_action(deep_challenge)
            fresh_action = _parse_dominant_action(fresh_challenge)
            await emit(
                progress,
                "challenges_generated",
                deep_action=deep_action.value,
                fresh_action=fresh_action.value,
                deep_preview=deep_challenge[:300],
                fresh_preview=fresh_challenge[:300],
            )

            async with self.store.transaction():
                for ctx in deep_sessions:
                    ch_msg = DebateMessage(
                        session_id=ctx.session_id,
                        phase=DebatePhase.CHALLENGE,
                        content=deep_challenge,
                        timestamp=_now(),
                        action=deep_action,
                    )
                    protocol.submit_message(ch_msg)
                    await self.store.save_message(
                        debate_id,
                        ctx.session_id,
                        "challenge",
                        deep_challenge,
                        deep_action.value,
                    )

                for ctx in fresh_sessions:
                    ch_msg = DebateMessage(
                        session_id=ctx.session_id,
                        phase=DebatePhase.CHALLENGE,
                        content=fresh_challenge,
                        timestamp=_now(),
                        action=fresh_action,
                    )
                    protocol.submit_message(ch_msg)
                    await self.store.save_message(
                        debate_id,
                        ctx.session_id,
                        "challenge",
                        fresh_challenge,
                        fresh_action.value,
                    )

            if pause_at == "convergence":
                paused_ctx = {
                    "deep_ids": [s.session_id for s in deep_sessions],
                    "fresh_ids": [s.session_id for s in fresh_sessions],
                    "deep_positions": list(deep_positions),
                    "fresh_positions": list(fresh_positions),
                    "deep_challenge": deep_challenge,
                    "fresh_challenge": fresh_challenge,
                    "fresh_role": fresh_role,
                    "delivery_mode": delivery_mode,
                    "effort": effort,
                    "deep_model": deep_model,
                    "fresh_model": fresh_model,
                    "paused_phase": "convergence",
                    "protocol_phase": protocol.phase.value,
                }
                self.paused_debates[debate_id] = paused_ctx
                await self.store.update_debate_status(debate_id, "paused")
                await self.store.save_paused_context(debate_id, paused_ctx)
                return {
                    "debate_id": debate_id,
                    "phase": "paused",
                    "paused_before": "convergence",
                    "mode": "auto_hitl",
                    "config": config,
                    "context_manifest": manifest.as_dict(),
                    "challenges": {
                        "deep": deep_challenge[:500],
                        "fresh": fresh_challenge[:500],
                    },
                    "message": "Debate paused for human review. Use debate_review to continue.",
                }

            protocol.advance_phase()  # → CONVERGENCE
            await emit(progress, "phase_started", phase="convergence", debate_id=debate_id)

            engine = ConvergenceEngine(use_llm=self.use_llm_convergence)
            session_roles: dict[str, str] = {}
            for ctx in deep_sessions:
                session_roles[ctx.session_id] = "Deep"
            for ctx in fresh_sessions:
                session_roles[ctx.session_id] = fresh_role.replace("_", "-").title()
            result = await engine.analyze(protocol, session_roles)

            protocol.advance_phase()  # → COMPLETE

            points_json = json.dumps(
                [
                    {
                        "category": p.category,
                        "summary": p.summary,
                        "session_a_view": p.session_a_view,
                        "session_b_view": p.session_b_view,
                        "resolution": p.resolution,
                        "root_cause": p.root_cause,
                    }
                    for p in result.points
                ]
            )
            await self.store.save_convergence_and_complete(
                debate_id,
                result.synthesis,
                result.confidence,
                points_json,
                meta_analysis=result.meta_analysis,
            )
            self._cleanup_debate(debate_id)
        except Exception:
            await self._delete_failed_debate(debate_id)
            raise

        metrics().debate_completed.labels(tenant=tenant_label(owner_id), mode="auto").inc()
        await emit(
            progress,
            "completed",
            debate_id=debate_id,
            confidence=result.confidence,
            points=len(result.points),
        )
        logger.info(
            "Auto-debate %s complete (confidence=%.2f, ploidy=%dn)",
            debate_id,
            result.confidence,
            deep_n,
        )

        rendered_markdown = render_debate(
            prompt=prompt,
            deep_label="Deep",
            fresh_label=fresh_role.replace("_", "-").title(),
            deep_positions=list(deep_positions),
            fresh_positions=list(fresh_positions),
            deep_challenge=deep_challenge,
            fresh_challenge=fresh_challenge,
            points=result.points,
            synthesis=result.synthesis,
            confidence=result.confidence,
            meta_analysis=result.meta_analysis,
            debate_id=debate_id,
            mode="auto",
        )

        return {
            "debate_id": debate_id,
            "phase": "complete",
            "mode": "auto",
            "config": config,
            "context_manifest": manifest.as_dict(),
            "synthesis": result.synthesis,
            "rendered_markdown": rendered_markdown,
            "confidence": result.confidence,
            "meta_analysis": result.meta_analysis,
            "points": [
                {
                    "category": p.category,
                    "summary": p.summary,
                    "resolution": p.resolution,
                    "root_cause": p.root_cause,
                }
                for p in result.points
            ],
        }

    async def review(
        self,
        debate_id: str,
        action: str = "approve",
        override_content: str | None = None,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        if debate_id not in self.paused_debates:
            raise PloidyError(f"Debate {debate_id} is not paused or does not exist")
        self._require_owner(debate_id, owner_id)

        if action not in ("approve", "override", "reject"):
            raise ProtocolError(
                f"Invalid action '{action}'. Must be 'approve', 'override', or 'reject'"
            )

        if action == "override" and not override_content:
            raise ProtocolError("override_content is required when action='override'")

        ctx = self.paused_debates.pop(debate_id)
        await self.store.clear_paused_context(debate_id)
        protocol = self.protocols.get(debate_id)

        if protocol is None:
            raise PloidyError(f"Protocol state lost for debate {debate_id}")

        if action == "reject":
            tenant = tenant_label(self.debate_owners.get(debate_id))
            await self.store.update_debate_status(debate_id, "cancelled")
            self._cleanup_debate(debate_id)
            metrics().debate_cancelled.labels(tenant=tenant, outcome="rejected").inc()
            logger.info("Auto-debate %s rejected by human reviewer", debate_id)
            return {
                "debate_id": debate_id,
                "phase": "cancelled",
                "mode": "auto_hitl",
                "message": "Debate rejected and cancelled by reviewer.",
            }

        try:
            from ploidy.api_client import generate_challenge
        except ImportError:
            raise PloidyError("API client not available. Install with: pip install ploidy[api]")

        deep_ids = ctx.get("deep_ids", [])
        fresh_ids = ctx.get("fresh_ids", [])
        deep_id = deep_ids[0] if deep_ids else None
        auto_id = fresh_ids[0] if fresh_ids else None
        fresh_role = ctx["fresh_role"]

        await self.store.update_debate_status(debate_id, "active")

        if ctx["paused_phase"] == "challenge":
            deep_positions = ctx.get("deep_positions", [])
            fresh_positions = ctx.get("fresh_positions", [])
            deep_pos = _aggregate_positions(deep_positions, "Deep")
            auto_pos = _aggregate_positions(fresh_positions, fresh_role.replace("_", "-").title())

            if action == "override" and override_content and auto_id:
                auto_pos = override_content
                msg = DebateMessage(
                    session_id=auto_id,
                    phase=DebatePhase.POSITION,
                    content=auto_pos,
                    timestamp=_now(),
                )
                protocol.messages = [
                    m
                    for m in protocol.messages
                    if not (m.session_id == auto_id and m.phase == DebatePhase.POSITION)
                ]
                protocol.submit_message(msg)
                await self.store.save_message(debate_id, auto_id, "position", auto_pos)

            protocol.advance_phase()  # → CHALLENGE

            effort = ctx.get("effort", "high")
            d_model = ctx.get("deep_model")
            f_model = ctx.get("fresh_model")

            deep_challenge, auto_challenge = await asyncio.gather(
                generate_challenge(
                    own_position=deep_pos,
                    other_position=auto_pos,
                    own_role="deep",
                    other_role=fresh_role,
                    effort=effort,
                    model=d_model,
                ),
                generate_challenge(
                    own_position=auto_pos,
                    other_position=deep_pos,
                    own_role=fresh_role,
                    other_role="deep",
                    effort=effort,
                    model=f_model,
                ),
            )

            for sid, content in [(deep_id, deep_challenge), (auto_id, auto_challenge)]:
                if sid is None:
                    continue
                ch_action = _parse_dominant_action(content)
                ch_msg = DebateMessage(
                    session_id=sid,
                    phase=DebatePhase.CHALLENGE,
                    content=content,
                    timestamp=_now(),
                    action=ch_action,
                )
                protocol.submit_message(ch_msg)
                await self.store.save_message(debate_id, sid, "challenge", content, ch_action.value)

        elif ctx["paused_phase"] == "convergence":
            if action == "override" and override_content and auto_id:
                auto_challenge = override_content
                protocol.messages = [
                    m
                    for m in protocol.messages
                    if not (m.session_id == auto_id and m.phase == DebatePhase.CHALLENGE)
                ]
                ch_action = _parse_dominant_action(auto_challenge)
                ch_msg = DebateMessage(
                    session_id=auto_id,
                    phase=DebatePhase.CHALLENGE,
                    content=auto_challenge,
                    timestamp=_now(),
                    action=ch_action,
                )
                protocol.submit_message(ch_msg)
                await self.store.save_message(
                    debate_id, auto_id, "challenge", auto_challenge, ch_action.value
                )

        protocol.advance_phase()  # → CONVERGENCE

        engine = ConvergenceEngine(use_llm=self.use_llm_convergence)
        session_roles: dict[str, str] = {}
        for sid in deep_ids:
            session_roles[sid] = "Deep"
        for sid in fresh_ids:
            session_roles[sid] = fresh_role.replace("_", "-").title()
        result = await engine.analyze(protocol, session_roles)

        protocol.advance_phase()  # → COMPLETE

        points_json = json.dumps(
            [
                {
                    "category": p.category,
                    "summary": p.summary,
                    "session_a_view": p.session_a_view,
                    "session_b_view": p.session_b_view,
                    "resolution": p.resolution,
                    "root_cause": p.root_cause,
                }
                for p in result.points
            ]
        )
        await self.store.save_convergence_and_complete(
            debate_id,
            result.synthesis,
            result.confidence,
            points_json,
            meta_analysis=result.meta_analysis,
        )
        tenant = tenant_label(self.debate_owners.get(debate_id))
        self._cleanup_debate(debate_id)
        metrics().debate_completed.labels(tenant=tenant, mode="auto_hitl").inc()

        logger.info(
            "Auto-debate %s resumed and completed via HITL (confidence=%.2f)",
            debate_id,
            result.confidence,
        )

        # Pull final positions/challenges off the protocol so the rendered
        # markdown includes whichever branch (override, approve) ran above.
        msg_by_session_phase: dict[tuple[str, DebatePhase], str] = {}
        for msg in protocol.messages:
            msg_by_session_phase[(msg.session_id, msg.phase)] = msg.content
        deep_positions_text = [
            msg_by_session_phase.get((sid, DebatePhase.POSITION), "") for sid in deep_ids
        ]
        fresh_positions_text = [
            msg_by_session_phase.get((sid, DebatePhase.POSITION), "") for sid in fresh_ids
        ]
        deep_challenge_text = (
            msg_by_session_phase.get((deep_id, DebatePhase.CHALLENGE)) if deep_id else None
        )
        fresh_challenge_text = (
            msg_by_session_phase.get((auto_id, DebatePhase.CHALLENGE)) if auto_id else None
        )

        rendered_markdown = render_debate(
            prompt=protocol.prompt,
            deep_label="Deep",
            fresh_label=fresh_role.replace("_", "-").title(),
            deep_positions=deep_positions_text,
            fresh_positions=fresh_positions_text,
            deep_challenge=deep_challenge_text,
            fresh_challenge=fresh_challenge_text,
            points=result.points,
            synthesis=result.synthesis,
            confidence=result.confidence,
            meta_analysis=result.meta_analysis,
            debate_id=debate_id,
            mode="auto_hitl",
        )

        return {
            "debate_id": debate_id,
            "phase": "complete",
            "mode": "auto_hitl",
            "reviewer_action": action,
            "fresh_role": fresh_role,
            "synthesis": result.synthesis,
            "rendered_markdown": rendered_markdown,
            "confidence": result.confidence,
            "meta_analysis": result.meta_analysis,
            "points": [
                {
                    "category": p.category,
                    "summary": p.summary,
                    "resolution": p.resolution,
                    "root_cause": p.root_cause,
                }
                for p in result.points
            ],
        }
