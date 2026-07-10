"""Unit tests for the API client with mocked OpenAI calls."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _set_api_env(monkeypatch):
    """Configure API environment for all tests in this module."""
    monkeypatch.setenv("PLOIDY_API_BASE_URL", "http://fake:1234")
    monkeypatch.setenv("PLOIDY_API_KEY", "test-key")
    monkeypatch.setenv("PLOIDY_API_MODEL", "test-model")


class TestResolveApiConfig:
    """Zero-config fallback so ``mode='auto'`` works without .mcp.json edits."""

    def test_ploidy_vars_win_when_set(self, monkeypatch):
        from ploidy.api_client import _resolve_api_config

        monkeypatch.setenv("PLOIDY_API_BASE_URL", "http://custom:9999")
        monkeypatch.setenv("PLOIDY_API_KEY", "ploidy-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        base_url, api_key, _ = _resolve_api_config()
        assert base_url == "http://custom:9999"
        assert api_key == "ploidy-key"

    def test_anthropic_key_auto_configures_endpoint(self, monkeypatch):
        from ploidy.api_client import _ANTHROPIC_OPENAI_COMPAT_URL, _resolve_api_config

        monkeypatch.delenv("PLOIDY_API_BASE_URL", raising=False)
        monkeypatch.delenv("PLOIDY_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        base_url, api_key, _ = _resolve_api_config()
        assert base_url == _ANTHROPIC_OPENAI_COMPAT_URL
        assert api_key == "anthropic-key"

    def test_disabled_when_no_credentials(self, monkeypatch):
        from ploidy.api_client import _resolve_api_config

        monkeypatch.delenv("PLOIDY_API_BASE_URL", raising=False)
        monkeypatch.delenv("PLOIDY_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        base_url, api_key, _ = _resolve_api_config()
        assert base_url is None
        assert api_key == ""

    def test_ploidy_url_without_key_gets_anthropic_key_fallback(self, monkeypatch):
        from ploidy.api_client import _resolve_api_config

        # Custom proxy URL but no explicit PLOIDY_API_KEY → fall back on
        # ANTHROPIC_API_KEY rather than sending an empty auth header.
        monkeypatch.setenv("PLOIDY_API_BASE_URL", "http://proxy:8080")
        monkeypatch.delenv("PLOIDY_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        base_url, api_key, _ = _resolve_api_config()
        assert base_url == "http://proxy:8080"
        assert api_key == "anthropic-key"


def _mock_completion(content: str = "test response"):
    """Create a mock chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestGenerateResponse:
    """Tests for generate_response with retry logic."""

    async def test_success_on_first_try(self, monkeypatch):
        from ploidy import api_client

        # Reload config after env is set
        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        monkeypatch.setattr(api_client, "_API_KEY", "test-key")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion("hello"))

        with patch.object(api_client, "_get_client", return_value=mock_client):
            result = await api_client.generate_response("test prompt")
        assert result == "hello"

    async def test_retries_on_rate_limit(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        monkeypatch.setattr(api_client, "_RETRY_BASE_DELAY", 0.01)

        # Create a rate limit error with the right class name
        rate_err = type("RateLimitError", (Exception,), {})("rate limited")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[rate_err, _mock_completion("recovered")]
        )

        with patch.object(api_client, "_get_client", return_value=mock_client):
            result = await api_client.generate_response("test prompt")
        assert result == "recovered"
        assert mock_client.chat.completions.create.call_count == 2

    async def test_non_retryable_error_fails_immediately(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=ValueError("bad input"))

        with (
            patch.object(api_client, "_get_client", return_value=mock_client),
            pytest.raises(RuntimeError, match="bad input"),
        ):
            await api_client.generate_response("test prompt")
        assert mock_client.chat.completions.create.call_count == 1

    async def test_max_retries_exhausted(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        monkeypatch.setattr(api_client, "_RETRY_BASE_DELAY", 0.01)

        timeout_err = type("APITimeoutError", (Exception,), {})("timeout")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=timeout_err)

        with (
            patch.object(api_client, "_get_client", return_value=mock_client),
            pytest.raises(RuntimeError, match="timeout"),
        ):
            await api_client.generate_response("test prompt")
        assert mock_client.chat.completions.create.call_count == 3


class TestIsApiAvailable:
    """Tests for API availability check."""

    def test_available_when_configured(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        assert api_client.is_api_available() is True

    def test_unavailable_when_empty(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "")
        assert api_client.is_api_available() is False

    def test_unavailable_when_none(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", None)
        assert api_client.is_api_available() is False


class TestFreshPosition:
    """Tests for generate_fresh_position."""

    async def test_fresh_has_no_system_prompt(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion("fresh view"))

        with patch.object(api_client, "_get_client", return_value=mock_client):
            result = await api_client.generate_fresh_position("test debate")
        assert result == "fresh view"
        # Verify the system prompt enforces zero-context independence
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if system_msgs:
            content = system_msgs[0].get("content", "").lower()
            assert "no background" in content or "no context" in content


def _messages_from(mock_client) -> list[dict]:
    call_args = mock_client.chat.completions.create.call_args
    return call_args.kwargs.get("messages", call_args[1].get("messages", []))


def _system_content(mock_client) -> str:
    return next(
        (m["content"] for m in _messages_from(mock_client) if m.get("role") == "system"),
        "",
    )


def _user_content(mock_client) -> str:
    return next(
        (m["content"] for m in _messages_from(mock_client) if m.get("role") == "user"),
        "",
    )


class TestMalformedResponse:
    async def test_empty_choices_raises_malformed(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")

        empty = MagicMock()
        empty.choices = []

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=empty)

        with (
            patch.object(api_client, "_get_client", return_value=mock_client),
            pytest.raises(RuntimeError, match="empty or malformed"),
        ):
            await api_client.generate_response("test")


class TestExperiencedPosition:
    async def test_context_documents_are_injected_into_prompt(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion("deep view"))

        with patch.object(api_client, "_get_client", return_value=mock_client):
            await api_client.generate_experienced_position(
                "Should we migrate?",
                context_documents=["Decision 17: picked Postgres.", "Incident: broken migration."],
            )
        user = _user_content(mock_client)
        assert "Decision 17" in user
        assert "Incident" in user
        assert "Should we migrate?" in user
        assert "experienced session" in _system_content(mock_client).lower()

    async def test_no_context_documents_omits_context_block(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion("ok"))

        with patch.object(api_client, "_get_client", return_value=mock_client):
            await api_client.generate_experienced_position("Q?", context_documents=None)
        user = _user_content(mock_client)
        assert "Project context documents" not in user
        assert "Q?" in user

    async def test_injected_system_context_reaches_system_message_only(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion("ok"))

        with patch.object(api_client, "_get_client", return_value=mock_client):
            await api_client.generate_experienced_position(
                "Q?",
                system_prompt="SYSTEM_CONTEXT_SENTINEL",
            )

        assert "SYSTEM_CONTEXT_SENTINEL" in _system_content(mock_client)
        assert "SYSTEM_CONTEXT_SENTINEL" not in _user_content(mock_client)


class TestSemiFreshPosition:
    async def test_passive_puts_summary_before_question(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion("sf"))

        with patch.object(api_client, "_get_client", return_value=mock_client):
            await api_client.generate_semi_fresh_position(
                "Main question", "Prior: auth flaw", delivery_mode="passive"
            )
        user = _user_content(mock_client)
        assert user.index("PRIOR ANALYSIS SUMMARY") < user.index("Main question")

    async def test_active_puts_independent_review_before_summary(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion("sf"))

        with patch.object(api_client, "_get_client", return_value=mock_client):
            await api_client.generate_semi_fresh_position(
                "Main question", "Prior: auth flaw", delivery_mode="active"
            )
        user = _user_content(mock_client)
        assert user.index("Main question") < user.index("PRIOR ANALYSIS")
        assert "independent" in user.lower()

    async def test_selective_frames_prior_as_uncertainty(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion("sf"))

        with patch.object(api_client, "_get_client", return_value=mock_client):
            await api_client.generate_semi_fresh_position(
                "Main question", "Auth flaw uncertain", delivery_mode="selective"
            )
        user = _user_content(mock_client)
        assert "AREAS OF UNCERTAINTY" in user
        assert "PRIOR ANALYSIS SUMMARY" not in user


class TestCompression:
    async def test_compress_position_requests_short_structured_summary(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion("compressed"))

        with patch.object(api_client, "_get_client", return_value=mock_client):
            result = await api_client.compress_position("Very long analysis body")
        assert result == "compressed"
        user = _user_content(mock_client)
        assert "Compress" in user
        assert "Very long analysis body" in user
        assert "300 words" in user

    async def test_compress_failures_excludes_confident_findings(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion("digest"))

        with patch.object(api_client, "_get_client", return_value=mock_client):
            await api_client.compress_failures_only("Full analysis")
        user = _user_content(mock_client)
        assert "uncertain" in user.lower()
        assert "Do NOT include issues the reviewer was confident about" in user
        assert "200 words" in user


class TestAnalyzeConvergence:
    async def test_transcript_contains_roles_positions_and_challenges(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_mock_completion("meta"))

        with patch.object(api_client, "_get_client", return_value=mock_client):
            await api_client.analyze_convergence(
                debate_prompt="Migrate now?",
                positions={"s1": "Yes because history", "s2": "No because risk"},
                challenges=[
                    {"session_id": "s1", "content": "Risk mitigation exists", "action": "rebut"},
                ],
                session_roles={"s1": "Deep", "s2": "Fresh"},
            )
        user = _user_content(mock_client)
        assert "Deep" in user
        assert "Fresh" in user
        assert "Yes because history" in user
        assert "No because risk" in user
        assert "Risk mitigation exists" in user
        # Root-cause taxonomy is load-bearing for the paper's §6 claims.
        assert "Context anchoring" in user
        assert "Stochastic variance" in user


class TestGetClientCaching:
    async def test_second_call_returns_cached_client(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        monkeypatch.setattr(api_client, "_API_KEY", "test-key")
        monkeypatch.setattr(api_client, "_cached_client", None)

        # ``openai`` is an optional extra; fake the module so the inner
        # ``from openai import AsyncOpenAI`` resolves without the real
        # dependency installed. Also lets us count instantiations.
        fake_constructor = MagicMock(return_value=MagicMock(name="openai_client"))
        fake_openai = SimpleNamespace(AsyncOpenAI=fake_constructor)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        c1 = await api_client._get_client()
        c2 = await api_client._get_client()
        assert c1 is c2
        assert fake_constructor.call_count == 1

    async def test_missing_openai_package_raises_importerror(self, monkeypatch):
        from ploidy import api_client

        monkeypatch.setattr(api_client, "_API_BASE_URL", "http://fake:1234")
        monkeypatch.setattr(api_client, "_cached_client", None)

        # Force the inner ``from openai import AsyncOpenAI`` to fail even
        # if the real package is available in the test env.
        monkeypatch.setitem(sys.modules, "openai", None)

        with pytest.raises(ImportError, match="openai package required"):
            await api_client._get_client()
