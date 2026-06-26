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
    blocked_sources: list[str]
    blocked_matches: list[str]
    total_chars: int
    approx_tokens: int

    def as_dict(self) -> dict[str, Any]:
        """Serialize the manifest for tool results and persisted config."""
        return {
            "entries": [entry.as_dict() for entry in self.entries],
            "blocked_sources": self.blocked_sources,
            "blocked_matches": self.blocked_matches,
            "total_chars": self.total_chars,
            "approx_tokens": self.approx_tokens,
        }


def build_context_manifest(
    context_documents: list[str],
    *,
    context_sources: list[str] | None = None,
    blocked_sources: list[str] | None = None,
) -> ContextManifest:
    """Build a manifest and reject documents that match blocked sources.

    ``context_sources`` names the provenance of each document. ``blocked_sources``
    is intentionally string-based so callers can block repo names, memory labels,
    pasted transcript markers, or other source identifiers without Ploidy needing
    to understand their filesystem.
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
    blocked_matches: list[str] = []
    entries: list[ContextManifestEntry] = []

    for index, (source, document) in enumerate(zip(sources, context_documents)):
        source_lower = source.lower()
        document_lower = document.lower()
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

    if blocked_matches:
        raise ProtocolError(
            "Blocked context source detected: " + ", ".join(sorted(set(blocked_matches)))
        )

    total_chars = sum(entry.chars for entry in entries)
    return ContextManifest(
        entries=entries,
        blocked_sources=blocked_terms,
        blocked_matches=[],
        total_chars=total_chars,
        approx_tokens=total_chars // 4,
    )
