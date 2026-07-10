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


def test_target_lease_allows_only_matching_source_labels():
    """A target lease turns source labels into an allowlist."""
    manifest = build_context_manifest(
        ["target repo context"],
        context_sources=["repo:trashmonster:/src"],
        target_lease="repo:trashmonster",
    )

    assert manifest.target_lease == "repo:trashmonster"
    assert manifest.allowed_sources == ["repo:trashmonster"]
    assert manifest.scope_policy()["subagent_must_inherit"] is True


def test_target_lease_rejects_out_of_scope_source_labels():
    """Context from a sibling repo cannot enter a leased target debate."""
    with pytest.raises(ProtocolError, match="Context source outside target lease"):
        build_context_manifest(
            ["sibling repo context"],
            context_sources=["repo:skillBridge:/src"],
            target_lease="repo:trashmonster",
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
        target_lease="repo:trashmonster",
    )

    manifest = result["context_manifest"]
    assert manifest["entries"][0]["source"] == "repo:trashmonster"
    assert manifest["target_lease"] == "repo:trashmonster"
    assert manifest["blocked_sources"] == ["skillBridge"]
    assert manifest["total_chars"] == len("trashmonster code context")
    assert result["scope_policy"]["target_lease"] == "repo:trashmonster"

    svc = server._service
    assert svc is not None
    debate = await svc.store.get_debate(result["debate_id"])
    assert debate is not None
    config = json.loads(debate["config_json"])
    assert config["context_manifest"] == manifest
    assert config["scope_policy"] == result["scope_policy"]


async def test_status_and_history_expose_context_scope_policy():
    """Active debate inspection surfaces the manifest instead of hiding it in JSON."""
    start = await server.debate_start(
        prompt="Review target repo",
        context_documents=["trashmonster code context"],
        context_sources=["repo:trashmonster"],
        blocked_sources=["skillBridge"],
        target_lease="repo:trashmonster",
    )

    status = await server.debate_status(start["debate_id"])
    assert status["context_manifest"]["entries"][0]["source"] == "repo:trashmonster"
    assert status["target_lease"] == "repo:trashmonster"
    assert status["scope_policy"]["subagent_must_inherit"] is True
    assert status["sessions"][0]["scope_policy"] == status["scope_policy"]

    history = await server.debate_history()
    record = next(debate for debate in history["debates"] if debate["id"] == start["debate_id"])
    assert record["context_manifest"] == status["context_manifest"]
    assert record["target_lease"] == "repo:trashmonster"
    assert record["scope_policy"] == status["scope_policy"]


async def test_joined_session_inherits_scope_policy():
    """Fresh and semi-fresh sessions inherit the parent target lease policy."""
    start = await server.debate_start(
        prompt="Review target repo",
        context_documents=["trashmonster code context"],
        context_sources=["repo:trashmonster"],
        target_lease="repo:trashmonster",
    )
    join = await server.debate_join(start["debate_id"], role="semi_fresh", delivery_mode="active")

    ctx = server._service.sessions[join["session_id"]]
    assert ctx.metadata["scope_policy"]["target_lease"] == "repo:trashmonster"
    assert ctx.metadata["scope_policy"]["subagent_must_inherit"] is True


async def test_trashmonster_target_rejects_skillbridge_context_regression():
    """Regression for target drift from a trashmonster request into skillBridge."""
    with pytest.raises(ProtocolError, match="Context source outside target lease"):
        await server.debate_solo(
            prompt="Critically review trashmonster",
            deep_position="A",
            fresh_position="B",
            context_documents=["skillBridge code and transcript context"],
            context_sources=["repo:skillBridge"],
            blocked_sources=["skillBridge"],
            target_lease="repo:trashmonster",
        )


async def test_target_lease_requires_explicit_matching_sources():
    """A leased debate cannot rely on anonymous context_documents labels."""
    with pytest.raises(ProtocolError, match="Context source outside target lease"):
        await server.debate_start(
            prompt="Review target repo",
            context_documents=["trashmonster code context"],
            target_lease="repo:trashmonster",
        )


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


async def test_auto_resume_guard_blocks_missing_scope_after_pause(monkeypatch):
    """HITL resume stops if compaction/recovery loses the scope proof."""
    monkeypatch.setattr(api_client, "is_api_available", lambda: True)

    async def fake_deep(prompt, context_documents=None, effort="high", model=None):
        return "deep position"

    async def fake_fresh(prompt, effort="high", model=None):
        return "fresh position"

    monkeypatch.setattr(api_client, "generate_experienced_position", fake_deep)
    monkeypatch.setattr(api_client, "generate_fresh_position", fake_fresh)

    paused = await server.debate_auto(
        prompt="Review target repo",
        context_documents=["trashmonster code context"],
        context_sources=["repo:trashmonster"],
        target_lease="repo:trashmonster",
        pause_at="challenge",
    )
    debate_id = paused["debate_id"]
    assert paused["resume_guard"]["requires_target_lease"] is True

    paused_context = server._service.paused_debates[debate_id]
    paused_context.pop("context_manifest")

    with pytest.raises(ProtocolError, match="Scope confirmation required before resume"):
        await server.debate_review(debate_id)

    assert debate_id in server._service.paused_debates
