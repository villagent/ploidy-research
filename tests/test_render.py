"""Tests for ploidy.render.render_debate and the ``rendered_markdown``
field on every completion path (solo / auto / two-terminal / HITL).

The renderer should produce answer-first markdown: a confidence headline
up top, the synthesis and full transcript inside ``<details>`` blocks,
optional meta-analysis collapsed separately. Clients that render
markdown natively (Claude.ai, Desktop, Code) then show just the
headline until the user expands a section.
"""

import pytest

import ploidy.api_client as api_client
from ploidy import server
from ploidy.convergence import ConvergencePoint
from ploidy.render import render_debate


@pytest.fixture(autouse=True)
async def _reset_state():
    if server._service is not None:
        await server._service.shutdown()
    server._service = None
    yield
    if server._service is not None:
        await server._service.shutdown()
    server._service = None


def _point(category: str, summary: str, resolution: str | None = None) -> ConvergencePoint:
    return ConvergencePoint(
        category=category,
        summary=summary,
        session_a_view="",
        session_b_view="",
        resolution=resolution,
    )


class TestRenderDebate:
    def test_no_challenges_headline_suppresses_misleading_zero_pct(self):
        """When the only point is the no_challenges marker, the headline
        must not read 'Confidence: 0%' — that misrepresents positions
        that substantively agree but never exchanged challenges."""
        md = render_debate(
            prompt="q",
            deep_label="Deep",
            fresh_label="Fresh",
            deep_positions=["we should ship"],
            fresh_positions=["yes, ship"],
            deep_challenge=None,
            fresh_challenge=None,
            points=[_point("no_challenges", "positions only")],
            synthesis="s",
            confidence=0.0,
        )
        assert "Confidence: 0%" not in md
        assert "Positions recorded — no challenges exchanged" in md

    def test_headline_includes_confidence_and_tallies(self):
        md = render_debate(
            prompt="should we ship?",
            deep_label="Deep",
            fresh_label="Fresh",
            deep_positions=["yes, battle-tested."],
            fresh_positions=["no, untested edge cases."],
            deep_challenge=None,
            fresh_challenge=None,
            points=[
                _point("agreement", "both agree test coverage is low"),
                _point("irreducible", "risk tolerance differs"),
            ],
            synthesis="## Debate: ship?\n...",
            confidence=0.5,
        )
        assert "Confidence: 50%" in md
        assert "✅ 1" in md and "🔴 1" in md

    def test_synthesis_is_collapsed(self):
        md = render_debate(
            prompt="q",
            deep_label="Deep",
            fresh_label="Fresh",
            deep_positions=["a"],
            fresh_positions=["b"],
            deep_challenge=None,
            fresh_challenge=None,
            points=[],
            synthesis="the full synthesis text",
            confidence=0.0,
        )
        assert "<details>" in md
        assert "<summary><strong>Synthesis</strong></summary>" in md
        assert "the full synthesis text" in md

    def test_full_transcript_includes_positions_and_challenges(self):
        md = render_debate(
            prompt="q",
            deep_label="Deep",
            fresh_label="Fresh",
            deep_positions=["DEEP POS"],
            fresh_positions=["FRESH POS"],
            deep_challenge="DEEP CH",
            fresh_challenge="FRESH CH",
            points=[],
            synthesis="s",
            confidence=1.0,
        )
        assert "DEEP POS" in md
        assert "FRESH POS" in md
        assert "DEEP CH" in md
        assert "FRESH CH" in md
        assert "Challenges" in md

    def test_meta_analysis_collapses_separately_when_present(self):
        md_with = render_debate(
            prompt="q",
            deep_label="Deep",
            fresh_label="Fresh",
            deep_positions=["a"],
            fresh_positions=["b"],
            deep_challenge=None,
            fresh_challenge=None,
            points=[],
            synthesis="s",
            confidence=0.5,
            meta_analysis="root cause narrative",
        )
        md_without = render_debate(
            prompt="q",
            deep_label="Deep",
            fresh_label="Fresh",
            deep_positions=["a"],
            fresh_positions=["b"],
            deep_challenge=None,
            fresh_challenge=None,
            points=[],
            synthesis="s",
            confidence=0.5,
        )
        assert "Meta-analysis" in md_with
        assert "root cause narrative" in md_with
        assert "Meta-analysis" not in md_without

    def test_single_position_omits_sequence_label(self):
        md = render_debate(
            prompt="q",
            deep_label="Deep",
            fresh_label="Fresh",
            deep_positions=["only one"],
            fresh_positions=["also one"],
            deep_challenge=None,
            fresh_challenge=None,
            points=[],
            synthesis="s",
            confidence=0.5,
        )
        # No "Deep 1/1" / "Fresh 1/1" noise for the common single case.
        assert "1/1" not in md

    def test_multi_position_labels_each(self):
        md = render_debate(
            prompt="q",
            deep_label="Deep",
            fresh_label="Fresh",
            deep_positions=["first", "second"],
            fresh_positions=["one"],
            deep_challenge=None,
            fresh_challenge=None,
            points=[],
            synthesis="s",
            confidence=0.5,
        )
        assert "Deep 1/2" in md
        assert "Deep 2/2" in md

    def test_points_grouped_by_category_with_emojis(self):
        md = render_debate(
            prompt="q",
            deep_label="Deep",
            fresh_label="Fresh",
            deep_positions=["a"],
            fresh_positions=["b"],
            deep_challenge=None,
            fresh_challenge=None,
            points=[
                _point("agreement", "shared finding"),
                _point("productive_disagreement", "trade-off X", resolution="pick A"),
                _point("irreducible", "values differ"),
            ],
            synthesis="s",
            confidence=0.5,
        )
        assert "✅ Agreements" in md
        assert "🟡 Productive disagreements" in md
        assert "🔴 Irreducible disagreements" in md
        assert "**Resolution:** pick A" in md

    def test_footer_shows_mode_and_debate_id(self):
        md = render_debate(
            prompt="q",
            deep_label="Deep",
            fresh_label="Fresh",
            deep_positions=["a"],
            fresh_positions=["b"],
            deep_challenge=None,
            fresh_challenge=None,
            points=[],
            synthesis="s",
            confidence=0.5,
            debate_id="abc123",
            mode="solo",
        )
        assert "debate_id: `abc123`" in md
        assert "mode: `solo`" in md


class TestRenderedMarkdownOnReturn:
    """End-to-end: every completion path populates ``rendered_markdown``."""

    async def test_solo_path(self):
        result = await server.debate(
            prompt="ship or wait?",
            mode="solo",
            deep_position="ship: tests cover the happy path.",
            fresh_position="wait: flakes in CI last week.",
            deep_challenge="CHALLENGE: CI flakes are infra, not code.",
            fresh_challenge="CHALLENGE: infra flakes mask product bugs.",
        )
        assert "rendered_markdown" in result
        md = result["rendered_markdown"]
        assert "Confidence:" in md
        assert "ship or wait?" in md
        assert "<details>" in md

    async def test_auto_path(self, monkeypatch):
        monkeypatch.setattr(api_client, "is_api_available", lambda: True)

        async def fake_deep(prompt, context_documents=None, effort="high", model=None):
            return "deep auto position"

        async def fake_fresh(prompt, effort="high", model=None):
            return "fresh auto position"

        async def fake_challenge(
            own_position,
            other_position,
            own_role="fresh",
            other_role="deep",
            effort="high",
            model=None,
        ):
            return f"CHALLENGE from {own_role}"

        monkeypatch.setattr(api_client, "generate_experienced_position", fake_deep)
        monkeypatch.setattr(api_client, "generate_fresh_position", fake_fresh)
        monkeypatch.setattr(api_client, "generate_challenge", fake_challenge)

        result = await server.debate(
            prompt="auto flow", mode="auto", context_documents=["project context"]
        )
        assert "rendered_markdown" in result
        md = result["rendered_markdown"]
        assert "mode: `auto`" in md
        assert "deep auto position" in md

    async def test_two_terminal_path(self):
        """Legacy debate_converge also returns rendered_markdown now."""
        start = await server.debate_start(prompt="mono vs poly?")
        join = await server.debate_join(start["debate_id"])
        await server.debate_position(start["session_id"], "monorepo: shared libs")
        await server.debate_position(join["session_id"], "polyrepo: independent deploys")
        await server.debate_challenge(
            start["session_id"], "polyrepo ignores shared auth", "challenge"
        )
        await server.debate_challenge(
            join["session_id"], "monorepo blocks independent cadence", "challenge"
        )
        result = await server.debate_converge(start["debate_id"])
        assert "rendered_markdown" in result
        md = result["rendered_markdown"]
        assert "mode: `two_terminal`" in md
        assert "monorepo: shared libs" in md
