"""OpenAI-compatible API client for Ploidy v0.2.

Provides automated session generation via any OpenAI-compatible endpoint,
enabling single-terminal debate workflows where the server auto-generates
Fresh or Semi-Fresh responses.

Supported backends (via configurable base_url):
- Anthropic (default)
- OpenAI
- Ollama (local)
- OpenRouter
- Google (Gemini via OpenAI-compatible proxy)

Environment variables:
    PLOIDY_API_BASE_URL: API endpoint URL (None = disabled)
    PLOIDY_API_KEY: API key for authentication
    PLOIDY_API_MODEL: Model identifier (default: claude-opus-4-6)

Zero-config fallback:
    If ``PLOIDY_API_BASE_URL`` is unset but ``ANTHROPIC_API_KEY`` is
    present (the default for any Claude Code session), auto mode is
    enabled against Anthropic's OpenAI-compat endpoint. This removes
    the "edit ``.mcp.json`` + restart Claude Code" friction and lets
    ``mode="solo"`` / ``mode="auto"`` be a pure per-call toggle.
"""

import asyncio
import logging
import os

from ploidy.metrics import metrics

logger = logging.getLogger("ploidy.api")

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_PER_CALL_TIMEOUT = 60.0  # seconds per individual API call
_CLIENT_TIMEOUT = 120.0  # seconds for client-level HTTP timeout


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ANTHROPIC_OPENAI_COMPAT_URL = "https://api.anthropic.com/v1/openai"


def _resolve_api_config() -> tuple[str | None, str, str]:
    """Resolve ``(base_url, api_key, model)`` from env with fallbacks.

    ``PLOIDY_*`` vars win when set. Otherwise, a present
    ``ANTHROPIC_API_KEY`` auto-configures the Anthropic OpenAI-compat
    endpoint so auto mode works out of the box from Claude Code.
    """
    base_url = os.environ.get("PLOIDY_API_BASE_URL")
    api_key = os.environ.get("PLOIDY_API_KEY", "")
    model = os.environ.get("PLOIDY_API_MODEL", "claude-opus-4-6")

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not base_url and anthropic_key:
        base_url = _ANTHROPIC_OPENAI_COMPAT_URL
    if not api_key and anthropic_key:
        api_key = anthropic_key

    return base_url, api_key, model


_API_BASE_URL, _API_KEY, _API_MODEL = _resolve_api_config()
# Opt-in prompt caching. Anthropic's OpenAI-compat endpoint honours
# cache_control breakpoints when passed via structured content blocks;
# leaving this off preserves single-block behaviour for providers that
# would reject the shape.
_CACHE_ENABLED = os.environ.get("PLOIDY_API_CACHE", "").lower() in ("1", "true", "yes")


def _provider_supports_cache_control() -> bool:
    """Heuristic: Anthropic's endpoint understands cache_control blocks."""
    if not _CACHE_ENABLED or not _API_BASE_URL:
        return False
    return "anthropic.com" in _API_BASE_URL.lower()


def is_api_available() -> bool:
    """Check whether the API fallback is configured and available."""
    return _API_BASE_URL is not None and _API_BASE_URL != ""


def configured_model() -> str:
    """Return the single configured model used by context-asymmetric runs."""
    return _API_MODEL


_cached_client = None


async def _get_client():
    """Lazily create and cache an AsyncOpenAI client."""
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "openai package required for v0.2 API fallback. Install with: pip install ploidy[api]"
        )

    kwargs = {"api_key": _API_KEY or "not-needed", "timeout": _CLIENT_TIMEOUT}
    if _API_BASE_URL:
        kwargs["base_url"] = _API_BASE_URL
    _cached_client = AsyncOpenAI(**kwargs)
    return _cached_client


def _build_user_content(
    prompt: str | None,
    cacheable_prefix: str | None,
) -> str | list[dict]:
    """Return user content in string form by default; structured blocks
    when cache_control pass-through is enabled and a prefix is supplied.

    The structured form splits the user message into (prefix, tail) with
    a cache breakpoint on the prefix, which matches the Anthropic
    OpenAI-compat contract for ephemeral caching.
    """
    if cacheable_prefix and _provider_supports_cache_control():
        return [
            {
                "type": "text",
                "text": cacheable_prefix,
                "cache_control": {"type": "ephemeral"},
            },
            *([{"type": "text", "text": prompt}] if prompt else []),
        ]
    # Fall back to a plain string — providers doing automatic prefix
    # caching (OpenAI, DeepSeek, etc.) still benefit because the shared
    # ``cacheable_prefix`` stays byte-identical across sibling calls.
    if cacheable_prefix and prompt:
        return cacheable_prefix + prompt
    return cacheable_prefix or prompt or ""


async def generate_response(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    effort: str = "high",
    max_tokens: int = 4096,
    cacheable_prefix: str | None = None,
) -> str:
    """Generate a response from the configured API endpoint.

    Args:
        prompt: The user prompt to send.
        system_prompt: Optional system-level instruction.
        model: Model override (defaults to PLOIDY_API_MODEL).
        effort: Effort level hint (low/medium/high/max).
        max_tokens: Maximum tokens in response.

    Returns:
        The model's response text.

    Raises:
        ImportError: If openai package is not installed.
        RuntimeError: If API call fails.
    """
    client = await _get_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": _build_user_content(prompt, cacheable_prefix)})

    # Map effort level to max_tokens budget
    effort_tokens = {"low": 1024, "medium": 2048, "high": 4096, "max": 8192}
    effective_max_tokens = effort_tokens.get(effort, max_tokens)

    retryable_names = ("RateLimitError", "APITimeoutError", "APIConnectionError")

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model or _API_MODEL,
                    messages=messages,
                    max_tokens=effective_max_tokens,
                ),
                timeout=_PER_CALL_TIMEOUT,
            )
        except Exception as e:
            is_retryable = isinstance(e, TimeoutError) or type(e).__name__ in retryable_names
            last_error = e
            if is_retryable and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "API call failed (attempt %d/%d, retrying in %.1fs): %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                continue
            logger.error("API call failed: %s (%s)", e, type(e).__name__)
            metrics().api_calls.labels(tenant="unscoped", outcome="error").inc()
            raise RuntimeError(f"Ploidy API call failed: {e}") from e
        try:
            content = response.choices[0].message.content or ""
        except (IndexError, KeyError, AttributeError) as e:
            metrics().api_calls.labels(tenant="unscoped", outcome="malformed").inc()
            raise RuntimeError(f"Ploidy API returned empty or malformed choices: {e}") from e
        metrics().api_calls.labels(tenant="unscoped", outcome="ok").inc()
        return content
    raise RuntimeError(f"Ploidy API call failed after {_MAX_RETRIES} retries: {last_error}")


async def generate_fresh_position(
    debate_prompt: str,
    effort: str = "high",
    model: str | None = None,
) -> str:
    """Generate a Fresh session position via API.

    The Fresh session receives only the debate prompt with no project context,
    ensuring maximum independence from accumulated biases.

    Args:
        debate_prompt: The decision question to analyze.
        effort: Effort level for reasoning depth.
        model: Optional model override.

    Returns:
        The Fresh session's position text.
    """
    system = (
        "You are participating in a structured debate. You have NO background context "
        "about this system or project. Review based purely on the code/question itself. "
        "List every bug, risk, or issue you can find. Be specific and technical. "
        "For each issue, classify your confidence as HIGH, MEDIUM, or LOW."
    )
    return await generate_response(
        prompt=debate_prompt,
        system_prompt=system,
        effort=effort,
        model=model,
    )


async def generate_experienced_position(
    debate_prompt: str,
    context_documents: list[str] | None = None,
    effort: str = "high",
    model: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Generate an Experienced session position via API.

    ``system_prompt`` carries injection-mode context supplied by the service.
    ``context_documents`` remains supported for direct callers, but the two
    mechanisms should not be used for the same context in one request.
    """
    context_block = ""
    if context_documents:
        joined = "\n\n".join(context_documents)
        context_block = f"Project context documents:\n{joined}\n\n"

    role_system = (
        "You are the experienced session in a structured debate. "
        "You have access to project-specific context that the other session may not see. "
        "Incorporate that context explicitly. List every bug, risk, or issue you can find. "
        "Be specific and technical. For each issue, classify your confidence as HIGH, "
        "MEDIUM, or LOW."
    )
    system = f"{role_system}\n\n{system_prompt}" if system_prompt else role_system
    return await generate_response(
        prompt=f"{context_block}Debate prompt:\n{debate_prompt}",
        system_prompt=system,
        effort=effort,
        model=model,
    )


async def generate_semi_fresh_position(
    debate_prompt: str,
    compressed_summary: str,
    delivery_mode: str = "active",
    effort: str = "high",
    model: str | None = None,
) -> str:
    """Generate a Semi-Fresh session position via API.

    The Semi-Fresh session receives compressed context from the Deep session's
    analysis, delivered either passively (in prompt) or actively (after
    independent analysis).

    Args:
        debate_prompt: The decision question to analyze.
        compressed_summary: Compressed summary of Deep session's analysis.
        delivery_mode: 'passive' (summary in prompt) or 'active' (after independent).
        effort: Effort level for reasoning depth.
        model: Optional model override.

    Returns:
        The Semi-Fresh session's position text.
    """
    if delivery_mode == "passive":
        prompt = (
            f"A previous reviewer analyzed this code/system and produced this summary:\n\n"
            f"--- PRIOR ANALYSIS SUMMARY ---\n{compressed_summary}\n--- END SUMMARY ---\n\n"
            f"Now perform your own independent review:\n\n{debate_prompt}\n\n"
            f"Use the prior summary as background context, but form your own conclusions.\n"
            f"List every bug, risk, or issue you can find. Be specific and technical.\n"
            f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW."
        )
    elif delivery_mode == "selective":
        prompt = (
            f"A previous reviewer flagged these areas of uncertainty:\n\n"
            f"--- AREAS OF UNCERTAINTY ---\n{compressed_summary}\n--- END ---\n\n"
            f"Use this as a starting point, but perform your own comprehensive review:\n\n"
            f"{debate_prompt}\n\n"
            f"List every bug, risk, or issue you can find. Be specific and technical.\n"
            f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW."
        )
    else:  # active
        prompt = (
            f"{debate_prompt}\n\n"
            f"You have NO background context about this system.\n"
            f"However, a previous reviewer has already analyzed this code. "
            f"Their compressed summary is available below if you want to consult it "
            f"AFTER forming your initial assessment.\n\n"
            f"INSTRUCTION: First, write your own independent analysis. "
            f"Then, consult the prior summary and note any additional issues or "
            f"disagreements.\n\n"
            f"List every bug, risk, or issue you can find. Be specific and technical.\n"
            f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.\n\n"
            f"--- PRIOR ANALYSIS (consult after your independent review) ---\n"
            f"{compressed_summary}\n--- END ---"
        )

    return await generate_response(
        prompt=prompt,
        effort=effort,
        model=model,
    )


def _build_challenge_shared_prefix(deep_position: str, fresh_position: str) -> str:
    """Byte-stable position block reused across both challenge calls.

    Both the deep→fresh and fresh→deep challenge calls see this exact
    text as the prefix of the user message. Keeping the ordering and
    wording fixed lets the provider's prefix cache deduplicate the
    ~2000-token block instead of billing for it twice per debate.
    """
    return (
        "Two sessions reviewed the same question with different context depths.\n\n"
        "**Deep session** (full project context):\n\n"
        f"{deep_position}\n\n"
        "**Fresh session** (zero background):\n\n"
        f"{fresh_position}\n\n"
    )


async def generate_challenge(
    own_position: str,
    other_position: str,
    own_role: str = "fresh",
    other_role: str = "experienced",
    effort: str = "high",
    model: str | None = None,
) -> str:
    """Generate a challenge response via API.

    Challenges come in pairs (deep vs fresh, fresh vs deep). We structure
    the user prompt as a cacheable position block followed by a tiny
    role-specific tail; providers that support prompt caching
    deduplicate the shared block, and those that don't still see a
    byte-stable prefix that many auto-caching providers catch anyway.

    Args:
        own_position: This session's position.
        other_position: The other session's position to challenge.
        own_role: This session's role description.
        other_role: The other session's role description.
        effort: Effort level for reasoning depth.
        model: Optional model override.

    Returns:
        The challenge response text.
    """
    if own_role in ("fresh", "semi_fresh"):
        bias_frame = "seems like rationalization or context-anchored bias"
    else:
        bias_frame = "wrong/misleading given project context"

    # Normalise the position pair into a fixed (deep, fresh) order so the
    # prefix is identical no matter which side's challenge is running.
    if own_role == "deep":
        deep_position, fresh_position = own_position, other_position
    else:
        deep_position, fresh_position = other_position, own_position

    cacheable_prefix = _build_challenge_shared_prefix(deep_position, fresh_position)
    tail = (
        f"---\n\nYour role: **{own_role}** session.\n\n"
        f"For EACH point in the opposing ({other_role}) session's position, respond:\n"
        f"- AGREE: valid finding\n"
        f"- CHALLENGE: {bias_frame}, explain why\n"
        f"- SYNTHESIZE: partially right, here's the nuance\n\n"
        f"Also list anything you found that they missed."
    )
    return await generate_response(
        prompt=tail,
        cacheable_prefix=cacheable_prefix,
        effort=effort,
        model=model,
    )


async def compress_position(position: str, model: str | None = None) -> str:
    """Compress a Deep position into a structured summary for Semi-Fresh sessions.

    Args:
        position: The full position text to compress.
        model: Optional model override.

    Returns:
        A compressed summary (max ~300 words).
    """
    return await generate_response(
        prompt=(
            f"Compress the following code review / architecture analysis into a SHORT "
            f"structured summary.\nInclude ONLY:\n"
            f"- Key issues found (one line each)\n"
            f"- Approaches considered\n"
            f"- Constraints mentioned\n\n"
            f"Do NOT include the full reasoning or project narrative. Max 300 words.\n\n"
            f"Analysis to compress:\n{position}"
        ),
        model=model,
    )


async def compress_failures_only(position: str, model: str | None = None) -> str:
    """Extract only failure/uncertainty information from a Deep position.

    Used for the SF-Selective delivery mode, where the Semi-Fresh session
    receives only what the Deep session was uncertain about -- excluding
    confident findings.

    Args:
        position: The full position text to extract failures from.
        model: Optional model override.

    Returns:
        A failure-only digest (max ~200 words).
    """
    return await generate_response(
        prompt=(
            f"From the following analysis, extract ONLY:\n"
            f"- Issues that were flagged as uncertain or low-confidence\n"
            f"- Risks or concerns that were noted but not fully resolved\n"
            f"- Limitations or gaps acknowledged by the reviewer\n\n"
            f"Do NOT include issues the reviewer was confident about.\n"
            f"Do NOT include the project context or background. Max 200 words.\n\n"
            f"Analysis:\n{position}"
        ),
        model=model,
    )


async def analyze_convergence(
    debate_prompt: str,
    positions: dict[str, str],
    challenges: list[dict],
    session_roles: dict[str, str],
    model: str | None = None,
) -> str:
    """LLM-based convergence meta-analysis.

    Analyzes WHY sessions disagreed, attributing disagreements to specific
    context factors (e.g., sunk cost bias from project history, missing
    domain constraints in Fresh session).

    Args:
        debate_prompt: The original debate prompt.
        positions: Map of session_id to position text.
        challenges: List of challenge messages with session_id, content, action.
        session_roles: Map of session_id to role name.
        model: Optional model override.

    Returns:
        Structured meta-analysis text.
    """
    parts = [f"## Debate Prompt\n{debate_prompt}\n"]

    for sid, pos in positions.items():
        role = session_roles.get(sid, sid[:8])
        parts.append(f"## {role} Session Position\n{pos}\n")

    if challenges:
        parts.append("## Challenges Exchanged")
        for ch in challenges:
            role = session_roles.get(ch.get("session_id", ""), "Unknown")
            action = ch.get("action", "challenge")
            parts.append(f"\n### {role} ({action}):\n{ch.get('content', '')}")

    transcript = "\n".join(parts)

    return await generate_response(
        prompt=(
            f"You are a meta-analyst evaluating a structured debate between AI sessions "
            f"with different context depths.\n\n"
            f"DEBATE TRANSCRIPT:\n{transcript}\n\n"
            f"Produce a structured analysis:\n\n"
            f"1. **Root Cause of Disagreements**: For each disagreement, identify WHETHER "
            f"it was caused by:\n"
            f"   - Context anchoring (Deep session rationalized due to sunk cost/familiarity)\n"
            f"   - Missing constraints (Fresh session missed domain requirements)\n"
            f"   - Genuine trade-off (both perspectives have merit)\n"
            f"   - Stochastic variance (disagreement is random, not context-driven)\n\n"
            f"2. **Context Attribution**: Which specific pieces of context in the Deep "
            f"session's history caused their position to diverge from the Fresh session?\n\n"
            f"3. **Synthesis**: What is the best decision considering both perspectives?\n\n"
            f"4. **Confidence Assessment**: How confident should the user be in this "
            f"synthesis? (0.0 to 1.0)"
        ),
        model=model,
    )
