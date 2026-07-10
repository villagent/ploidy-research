"""Tests for the single-page web UI.

Narrow checks — the renderer is a static HTML string, so the only
interesting assertions are "it contains the things clients depend on"
and "the custom route serves it with the right content-type".
"""

import pytest

from ploidy import server
from ploidy.webapp import index_html


@pytest.fixture(autouse=True)
async def _reset_state():
    if server._service is not None:
        await server._service.shutdown()
    server._service = None
    yield
    if server._service is not None:
        await server._service.shutdown()
    server._service = None


def test_index_html_has_required_elements():
    html = index_html()
    assert "<title>Ploidy — live debate</title>" in html
    assert 'id="debate-form"' in html
    assert 'id="prompt"' in html
    assert 'id="context"' in html
    assert 'name="context"' in html
    assert "context_documents: [context]" in html
    assert 'id="token"' in html
    # Points at the SSE endpoint, not some placeholder.
    assert "/v1/debate/stream" in html
    # Model output is rendered as text, never through an HTML injection sink.
    assert "pre.textContent = md" in html
    assert "innerHTML" not in html
    # Bearer credentials are memory-only and disappear on navigation.
    assert "localStorage" not in html


def test_index_html_is_self_contained():
    """The zero-build UI has no external script or stylesheet dependency."""
    html = index_html()
    # No broken link placeholders.
    assert "TODO" not in html
    assert "{{" not in html
    # Inline CSS + JS only.
    assert html.count("<link rel") == 0
    assert html.count("<script") == 1


async def test_web_route_serves_html():
    """Exercise the Starlette route handler directly."""
    from ploidy.server import _webapp

    class _FakeRequest:
        pass

    resp = await _webapp(_FakeRequest())
    assert resp.status_code == 200
    assert resp.media_type == "text/html"
    body = resp.body.decode("utf-8")
    assert "<title>Ploidy — live debate</title>" in body
