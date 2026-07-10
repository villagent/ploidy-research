"""Context manifest and blocking helpers for debate inputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ploidy.exceptions import ProtocolError


@dataclass(frozen=True)
class ContextManifestEntry:
    """Evidence record for one loaded context document."""

    index: int
    source: str
    sha256: str
    chars: int
    approx_tokens: int

    def as_dict(self) -> dict[str, Any]:
        """Serialize the entry for tool results and persisted config."""
        return {
            "index": self.index,
            "source": self.source,
            "sha256": self.sha256,
            "chars": self.chars,
            "approx_tokens": self.approx_tokens,
        }


@dataclass(frozen=True)
class ContextManifest:
    """Manifest for all context loaded into the deep side."""

    entries: list[ContextManifestEntry]
    target_lease: str | None
    allowed_sources: list[str]
    blocked_sources: list[str]
    blocked_matches: list[str]
    rejected_sources: list[str]
    total_chars: int
    approx_tokens: int

    def as_dict(self) -> dict[str, Any]:
        """Serialize the manifest for tool results and persisted config."""
        return {
            "entries": [entry.as_dict() for entry in self.entries],
            "target_lease": self.target_lease,
            "allowed_sources": self.allowed_sources,
            "blocked_sources": self.blocked_sources,
            "blocked_matches": self.blocked_matches,
            "rejected_sources": self.rejected_sources,
            "total_chars": self.total_chars,
            "approx_tokens": self.approx_tokens,
        }

    def scope_policy(self) -> dict[str, Any]:
        """Return the policy child sessions and subagents must inherit."""
        return {
            "target_lease": self.target_lease,
            "allowed_sources": self.allowed_sources,
            "blocked_sources": self.blocked_sources,
            "subagent_must_inherit": True,
            "compaction_resume_guard": bool(self.target_lease or self.entries),
            "requires_context_manifest": True,
            "requires_target_lease": self.target_lease is not None,
        }


def build_context_manifest(
    context_documents: list[str],
    *,
    context_sources: list[str] | None = None,
    blocked_sources: list[str] | None = None,
    target_lease: str | None = None,
    allowed_sources: list[str] | None = None,
) -> ContextManifest:
    """Build a manifest and reject documents that match blocked sources.

    ``context_sources`` names the provenance of each document. ``blocked_sources``
    is intentionally string-based so callers can block repo names, memory labels,
    pasted transcript markers, or other source identifiers without Ploidy needing
    to understand their filesystem. ``target_lease`` pins the debate to one
    intended target; when it or ``allowed_sources`` is set, each source label must
    contain at least one allowed source token.
    """
    sources = context_sources or [
        f"context_documents[{idx}]" for idx, _ in enumerate(context_documents)
    ]
    if len(sources) != len(context_documents):
        raise ProtocolError(
            "context_sources length must match context_documents "
            f"({len(sources)} != {len(context_documents)})"
        )

    blocked_terms = [term for term in (blocked_sources or []) if term.strip()]
    blocked_terms_lower = [term.lower() for term in blocked_terms]
    target = target_lease.strip() if target_lease and target_lease.strip() else None
    allowed_terms = [term for term in (allowed_sources or []) if term.strip()]
    if target and not allowed_terms:
        allowed_terms = [target]
    allowed_terms_lower = [term.lower() for term in allowed_terms]
    blocked_matches: list[str] = []
    rejected_sources: list[str] = []
    entries: list[ContextManifestEntry] = []

    for index, (source, document) in enumerate(zip(sources, context_documents)):
        source_lower = source.lower()
        document_lower = document.lower()
        if allowed_terms_lower and not any(term in source_lower for term in allowed_terms_lower):
            rejected_sources.append(f"source[{index}]:{source}")
        for original, lowered in zip(blocked_terms, blocked_terms_lower):
            if lowered in source_lower:
                blocked_matches.append(f"source[{index}]:{original}")
            if lowered in document_lower:
                blocked_matches.append(f"content[{index}]:{original}")

        chars = len(document)
        entries.append(
            ContextManifestEntry(
                index=index,
                source=source,
                sha256=hashlib.sha256(document.encode("utf-8")).hexdigest(),
                chars=chars,
                approx_tokens=chars // 4,
            )
        )

    if rejected_sources:
        allowed = ", ".join(allowed_terms)
        rejected = ", ".join(sorted(set(rejected_sources)))
        raise ProtocolError(
            f"Context source outside target lease: {rejected}. Allowed source tokens: {allowed}"
        )

    if blocked_matches:
        raise ProtocolError(
            "Blocked context source detected: " + ", ".join(sorted(set(blocked_matches)))
        )

    total_chars = sum(entry.chars for entry in entries)
    return ContextManifest(
        entries=entries,
        target_lease=target,
        allowed_sources=allowed_terms,
        blocked_sources=blocked_terms,
        blocked_matches=[],
        rejected_sources=[],
        total_chars=total_chars,
        approx_tokens=total_chars // 4,
    )
