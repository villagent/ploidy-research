"""Tests for context provenance manifests and source blocking."""

import json

import pytest

import ploidy.api_client as api_client
from ploidy import server
from ploidy.context_firewall import build_context_manifest
from ploidy.exceptions import ProtocolError


@pytest.fixture(autouse=True)
async def _reset_state():
    """Drop the shared DebateService between tests."""
    if server._service is not None:
        await server._service.shutdown()
    server._service = None
    yield
    if server._service is not None:
        await server._service.shutdown()
    server._service = None


def test_manifest_defaults_sources_and_hashes_documents():
    """Documents without explicit sources still get stable manifest entries."""
    manifest = build_context_manifest(["alpha", "beta"])

    assert manifest.total_chars == 9
    assert manifest.approx_tokens == 2
    assert [entry.source for entry in manifest.entries] == [
        "context_documents[0]",
        "context_documents[1]",
    ]
    assert all(len(entry.sha256) == 64 for entry in manifest.entries)


def test_context_sources_must_match_document_count():
    """Source labels are one-to-one with loaded documents."""
    with pytest.raises(ProtocolError, match="context_sources length must match"):
        build_context_manifest(["alpha"], context_sources=["one", "two"])


def test_blocked_source_label_rejects_manifest():
    """Blocked source strings reject contaminated provenance labels."""
    with pytest.raises(ProtocolError, match=r"source\[0\]:skillBridge"):
        build_context_manifest(
            ["target repo context"],
            context_sources=["repo:skillBridge"],
            blocked_sources=["skillBridge"],
        )


def test_blocked_document_content_rejects_manifest():
    """Blocked source strings also reject contaminated document bodies."""
    with pytest.raises(ProtocolError, match=r"content\[0\]:skillBridge"):
        build_context_manifest(
            ["This paste discusses skillBridge instead."],
            context_sources=["repo:trashmonster"],
            blocked_sources=["skillBridge"],
        )


async def test_solo_records_context_manifest_in_result_and_config():
    """A solo debate exposes and persists the exact loaded-context manifest."""
    result = await server.debate_solo(
        prompt="Review target repo",
        deep_position="The target repo needs scope guards.",
        fresh_position="The target repo needs provenance checks.",
        context_documents=["trashmonster code context"],
        context_sources=["repo:trashmonster"],
        blocked_sources=["skillBridge"],
    )

    manifest = result["context_manifest"]
    assert manifest["entries"][0]["source"] == "repo:trashmonster"
    assert manifest["blocked_sources"] == ["skillBridge"]
    assert manifest["total_chars"] == len("trashmonster code context")

    svc = server._service
    assert svc is not None
    debate = await svc.store.get_debate(result["debate_id"])
    assert debate is not None
    config = json.loads(debate["config_json"])
    assert config["context_manifest"] == manifest


async def test_solo_blocks_forbidden_context_before_persisting():
    """A contaminated solo context fails before any debate row is saved."""
    svc = await server._init()
    prompt = "Blocked context should not persist"

    with pytest.raises(ProtocolError, match="Blocked context source detected"):
        await server.debate_solo(
            prompt=prompt,
            deep_position="A",
            fresh_position="B",
            context_documents=["skillBridge context leaked in"],
            context_sources=["repo:trashmonster"],
            blocked_sources=["skillBridge"],
        )

    history = await svc.history()
    assert all(debate["prompt"] != prompt for debate in history["debates"])


async def test_auto_blocks_forbidden_context_before_api_configuration_check(monkeypatch):
    """The firewall runs before API availability so contamination is the error."""
    monkeypatch.setattr(api_client, "is_api_available", lambda: False)

    with pytest.raises(ProtocolError, match="Blocked context source detected"):
        await server.debate_auto(
            prompt="Review target repo",
            context_documents=["skillBridge context leaked in"],
            context_sources=["repo:trashmonster"],
            blocked_sources=["skillBridge"],
        )
