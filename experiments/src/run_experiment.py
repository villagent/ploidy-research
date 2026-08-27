"""
Ploidy Experiment Runner
========================
Evaluates context-asymmetric debate across the Context Asymmetry Spectrum.

Uses `claude --print` CLI (no API key needed, uses Max/Pro subscription).
Each call = fresh session = perfect context isolation.

Methods (12):
  single          - Single session, full context
  second_opinion  - Two independent sessions, concatenated
  ccr             - CCR: deep produces, fresh reviews
  symmetric       - Symmetric debate (both full context)
  ploidy          - Asymmetric debate (deep vs fresh)
  self_consistency - 5 independent runs + majority vote (same token budget as ploidy)
  stochastic_n    - N independent sessions, same context (Event B baseline, scales with --ploidy)
  sf_passive      - Semi-Fresh (Passive): compressed summary in prompt
  sf_active       - Semi-Fresh (Active): summary available, must retrieve after independent analysis
  sf_selective    - Semi-Fresh (Selective): only failure/uncertainty info provided
  sf_passive_indep - Ablation: passive + independent-first instruction
  sf_passive_bottom - Ablation: passive with summary at bottom

Experimental Variables:
  --effort         Effort level (low/medium/high/max) — controls LLM reasoning depth
  --model          Model identifier
  --long           Use long-context tasks with anchoring biases
  --injection      Context injection mode (raw/system_prompt/memory/skills/claude_md)
  --lang           Output language (en/ko/ja/zh)

Usage:
    python experiments/run_experiment.py
    python experiments/run_experiment.py --tasks 0,1,2 --methods ploidy,single
    python experiments/run_experiment.py --long --effort high --methods single,ploidy,sf_active
    python experiments/run_experiment.py --long --effort-sweep   # Run all effort levels
    python experiments/run_experiment.py --injection-sweep        # Run all injection modes
    python experiments/run_experiment.py --injection memory --methods ploidy,single
"""

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from task_model import Task

MODEL = "claude-opus-4-7"
JUDGE_MODEL = "claude-opus-4-7"
EFFORT = "high"  # default effort level
LANGUAGE = "en"  # default output language
INJECTION_MODE = "raw"  # default context injection mode
DEEP_N = 1  # number of Deep sessions (stochastic sampling)
FRESH_N = 1  # number of Fresh sessions (stochastic sampling)
CONTEXT_PCT = 100  # percentage of context given to Deep sessions (0-100)

# Valid effort levels for Claude Code
EFFORT_LEVELS = ["low", "medium", "high", "max"]

# Languages for localization variable testing
LANGUAGES = {
    "en": "Respond in English.",
    "ko": "한국어로 답변하세요. 기술적 용어는 가능한 한 한국어로 번역하세요.",
    "ja": "日本語で回答してください。技術用語はできる限り日本語に翻訳してください。",
    "zh": "请用中文回答。尽可能将技术术语翻译成中文。",
}

# Context injection modes — tests whether the *form* of context delivery
# affects model behavior and debate outcomes, independent of content.
#
# Hypotheses:
# - H1: Memory-style injection (accumulated observations) may produce more
#        anchored/sycophantic responses than skills-style (declarative rules)
# - H2: System-prompt injection may create stronger priors than user-message
#        injection due to positional authority bias
# - H3: The interaction between injection mode and context asymmetry may
#        differ — Fresh sessions may be differentially affected by how the
#        Deep session's context was originally formed
INJECTION_MODES = {
    "raw": {
        "description": "Raw context in user prompt (baseline)",
        "format": lambda ctx: f"Context about this code/system:\n{ctx}",
    },
    "system_prompt": {
        "description": "Context as system-level instruction",
        "format": lambda ctx: (
            f"You are a senior engineer with deep knowledge of this system. "
            f"You have the following project context that informs your analysis:\n\n{ctx}"
        ),
    },
    "memory": {
        "description": "Context as accumulated memories (CLAUDE.md / memory.md style)",
        "format": lambda ctx: (
            "The following memories were accumulated from your prior work sessions "
            "on this project. They represent observations, decisions, and lessons "
            "learned over time:\n\n"
            "---\n"
            "## Project Memory\n\n"
            + "\n".join(
                f"- **Observation #{i + 1}**: {line.strip()}"
                for i, line in enumerate(ctx.strip().split("\n"))
                if line.strip()
            )
            + "\n---"
        ),
    },
    "skills": {
        "description": "Context as declarative rules/capabilities (skills.md style)",
        "format": lambda ctx: (
            "# Project Skills & Rules\n\n"
            "When analyzing this codebase, apply the following rules and constraints:\n\n"
            + "\n".join(
                f"- RULE: {line.strip()}"
                for i, line in enumerate(ctx.strip().split("\n"))
                if line.strip()
            )
        ),
    },
    "claude_md": {
        "description": "Context as CLAUDE.md project instructions",
        "format": lambda ctx: (
            f"<project-instructions>\n"
            f"# CLAUDE.md\n\n"
            f"## Project Context\n\n{ctx}\n\n"
            f"## Conventions\n"
            f"- Be thorough in code review\n"
            f"- Flag all security issues\n"
            f"- Consider operational context\n"
            f"</project-instructions>"
        ),
    },
}


# ─── Context Injection Helper ────────────────────────────────────────────────


def format_deep_context(context: str, mode: str = None) -> str:
    """Format context for the Deep session using the specified injection mode.

    Args:
        context: Raw context string from the task.
        mode: Injection mode key (raw/system_prompt/memory/skills/claude_md).

    Returns:
        Formatted context string.
    """
    actual_mode = mode or INJECTION_MODE
    if actual_mode not in INJECTION_MODES:
        actual_mode = "raw"
    return INJECTION_MODES[actual_mode]["format"](context)


def truncate_context(context: str, pct: float) -> str:
    """Truncate context to a given percentage, snapping to sentence boundaries.

    Args:
        context: Full context string.
        pct: Percentage of context to retain (0-100).

    Returns:
        Truncated context string.
    """
    if pct >= 100:
        return context
    if pct <= 0:
        return ""
    target_len = int(len(context) * pct / 100)
    truncated = context[:target_len]
    last_period = truncated.rfind(".")
    if last_period > target_len * 0.7:
        return truncated[: last_period + 1]
    return truncated


def build_deep_prompt(task_context: str, task_prompt: str, mode: str = None) -> str:
    """Build the full Deep session prompt with injection-mode-appropriate context.

    For system_prompt mode, context goes into --system-prompt flag via call_llm.
    For all other modes, context is prepended to the user prompt.

    Args:
        task_context: The task's context string.
        task_prompt: The task's prompt string.
        mode: Injection mode key.

    Returns:
        The user prompt (context may be embedded or separate depending on mode).
    """
    actual_mode = mode or INJECTION_MODE
    task_context = truncate_context(task_context, CONTEXT_PCT)
    formatted_ctx = format_deep_context(task_context, actual_mode)

    if actual_mode == "system_prompt":
        # Context goes via --system-prompt; user prompt has only the task
        return task_prompt
    else:
        return f"{formatted_ctx}\n\n{task_prompt}"


def get_system_prompt_for_mode(task_context: str, mode: str = None) -> str | None:
    """Return system prompt if injection mode uses it, else None.

    Args:
        task_context: The task's context string.
        mode: Injection mode key.

    Returns:
        System prompt string, or None.
    """
    actual_mode = mode or INJECTION_MODE
    if actual_mode == "system_prompt":
        return format_deep_context(task_context, actual_mode)
    return None


# ─── LLM Backend ────────────────────────────────────────────────────────────

BACKEND = "claude"  # claude | gemini | openai
TEMPERATURE = 0.0  # fixed for reproducibility (controls Event B variance)
MAX_TOKENS = 8192  # sufficient for debate synthesis phases

# ─── Token Tracking ─────────────────────────────────────────────────────────
# Tracks cumulative token usage per experiment run.
# CLI backends (claude, gemini, codex) don't report tokens, so we estimate
# using chars/4 heuristic. API backends report exact usage.

# Per-thread token tracker. The outer task loop now runs all methods of one
# task concurrently (Single + CCR + Ploidy in parallel) and multiple tasks
# concurrently too — a single global tracker would conflate token attribution
# across methods. ``threading.local()`` gives each method-thread its own
# counter. ``method_ploidy``'s *internal* ThreadPoolExecutor threads
# (Deep/Fresh position+challenge phases) explicitly inherit the parent
# thread's tracker via ``_bind_parent_tracker`` so their token writes still
# land on the right per-method counter.
_token_tls = threading.local()
_token_tracker_lock = threading.Lock()


def _new_tracker() -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "estimated": True,  # True if any call used char-based estimation
    }


def _current_tracker() -> dict:
    """Return this thread's tracker dict (creating one on first access)."""
    t = getattr(_token_tls, "tracker", None)
    if t is None:
        t = _new_tracker()
        _token_tls.tracker = t
    return t


def _bind_parent_tracker(tracker: dict) -> None:
    """Force the current thread to write into the supplied tracker dict.

    Used by ``method_ploidy``'s internal ThreadPoolExecutor so child-thread
    ``call_llm()`` writes still accumulate on the parent method's counter.
    """
    _token_tls.tracker = tracker


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English, ~2 for CJK."""
    return max(1, len(text) // 4)


def _track_tokens(prompt_tokens: int, completion_tokens: int, exact: bool = False):
    """Accumulate token usage for the current thread's tracker.

    Lock-protected because ``method_ploidy`` and ``method_stochastic_n``
    invoke ``call_llm`` from multiple threads whose ``_token_tls`` has been
    bound to the same parent dict. Concurrent dict increments would race
    without the lock.
    """
    t = _current_tracker()
    with _token_tracker_lock:
        t["prompt_tokens"] += prompt_tokens
        t["completion_tokens"] += completion_tokens
        t["total_tokens"] += prompt_tokens + completion_tokens
        t["calls"] += 1
        if exact:
            t["estimated"] = False  # at least one exact measurement


def reset_token_tracker():
    """Reset tracker for a new method run on this thread."""
    _token_tls.tracker = _new_tracker()


def get_token_usage() -> dict:
    """Return a copy of the current thread's token usage."""
    return dict(_current_tracker())


# Backend-specific model defaults
BACKEND_DEFAULTS: dict[str, str] = {
    "claude": "claude-opus-4-7",
    "gemini": "gemini-3.1-pro",
    "codex": "codex-default",
    "openai": "gpt-4.1",
}


# A scratch directory that has no associated `~/.claude/projects/<encoded-cwd>/memory/`
# tree, used as the working directory for every experimental CLI call so the
# model never auto-loads the ploidy project's memory entries. Resolved lazily.
_NEUTRAL_CWD: str | None = None


def _neutral_cwd() -> str:
    """Return a stable cwd whose encoded-cwd memory dir is empty.

    The Claude Code CLI auto-loads every ``*.md`` file under
    ``~/.claude/projects/<encoded-cwd>/memory/`` into the model's context.
    Running ``claude --print`` from the repo root therefore injects ~500
    architecture-review / fabrication-casebook memory files into every
    cell — which contaminated the first 4th-sweep attempt by causing the
    model to pattern-match new (legitimate) gradient prompts as the next
    recurrence of an earlier refusal case (``F1 = 0.000`` on ~43% of
    cells regardless of method). Invoking from a scratch directory whose
    encoded path has no associated memory tree neutralises the leak
    without using ``--bare`` (which would also disable OAuth and break
    the Max-subscription zero-cost flow).
    """
    global _NEUTRAL_CWD
    if _NEUTRAL_CWD is None:
        import tempfile

        _NEUTRAL_CWD = tempfile.mkdtemp(prefix="ploidy-exp-cwd-")
    return _NEUTRAL_CWD


def _call_claude(prompt: str, model: str, effort: str, system_prompt: str = None) -> str:
    """Call via claude CLI --print. Free with Max/Pro subscription.

    Runs from a memory-neutral cwd (see ``_neutral_cwd``) so the
    invocation does not auto-load the ploidy project's
    ``~/.claude/projects/<encoded-cwd>/memory/*.md`` files into the
    model's context.
    """
    cmd = ["claude", "--print", "--model", model]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    if effort and effort != "high":
        cmd.extend(["--effort", effort])
    cmd.append(prompt)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=_neutral_cwd(),
    )
    if result.returncode != 0:
        err_msg = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"claude CLI error: {err_msg}")
    output = result.stdout.strip()
    # Detect the Claude-Max quota-exhaustion banner that ``claude --print``
    # emits to stdout (with returncode 0) when the subscription cycle is
    # exhausted. The original check used bare ``"hit your limit"`` /
    # ``"resets"`` substrings, which fired on ordinary review content like
    # *"rate limit resets every 60s"* or *"users hit their limit on login"*
    # — long_auth_overhaul cells fail this way ~100% of the time. Require
    # multi-word phrases that only appear in the actual CLI quota banner.
    low = output.lower()
    quota_banners = [
        "claude usage limit reached",
        "you've reached your usage limit",
        "you have reached your usage limit",
        "claude max usage limit",
        "you've hit your usage limit",
        "you've hit your subscription limit",
        "hit your usage limit",
        "usage limit. resets",
        "usage limit. try again",
        "limit reached. resets",
        "limit reached. try again",
    ]
    if any(phrase in low for phrase in quota_banners):
        raise RuntimeError(f"claude CLI error: {output[:200]}")
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    _track_tokens(_estimate_tokens(full_prompt), _estimate_tokens(output))
    return output


def _call_codex(prompt: str, model: str, effort: str, system_prompt: str = None) -> str:
    """Call via codex exec. Free with ChatGPT Free/Plus."""
    import tempfile

    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    outfile = tempfile.mktemp(suffix=".txt")
    cmd = ["codex", "exec", "--output-last-message", outfile]
    if model and model != "codex-default":
        cmd.extend(["-m", model])
    cmd.append(full_prompt)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"codex CLI error: {result.stderr.strip()}")
    try:
        with open(outfile) as f:
            output = f.read().strip()
        _track_tokens(_estimate_tokens(full_prompt), _estimate_tokens(output))
        return output
    finally:
        import os

        os.unlink(outfile) if os.path.exists(outfile) else None


def _call_gemini(prompt: str, model: str, effort: str, system_prompt: str = None) -> str:
    """Call via gemini CLI -p. Free with Gemini CLI."""
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    cmd = ["gemini", "-p", full_prompt]
    # Only pass -m if user explicitly specified a non-default model
    if model and model not in ("gemini-default", "gemini-3.1-pro"):
        cmd.extend(["-m", model])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"gemini CLI error: {result.stderr.strip()}")
    output = result.stdout.strip()
    _track_tokens(_estimate_tokens(full_prompt), _estimate_tokens(output))
    return output


def _call_openai_api(prompt: str, model: str, effort: str, system_prompt: str = None) -> str:
    """Call via OpenAI-compatible API.

    Unified backend for all non-Claude models. Set base_url to target different providers:
      - OpenAI direct:  OPENAI_BASE_URL=https://api.openai.com/v1
      - OpenRouter:     OPENAI_BASE_URL=https://openrouter.ai/api/v1
      - Ollama local:   OPENAI_BASE_URL=http://localhost:11434/v1
      - Anthropic:      OPENAI_BASE_URL=https://api.anthropic.com/v1 (with adapter)
    """
    import os

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    if response.usage:
        _track_tokens(
            response.usage.prompt_tokens or 0,
            response.usage.completion_tokens or 0,
            exact=True,
        )
    else:
        content = response.choices[0].message.content or ""
        full_prompt = (system_prompt or "") + prompt
        _track_tokens(_estimate_tokens(full_prompt), _estimate_tokens(content))
    return response.choices[0].message.content or ""


def _calc_wait_until_reset(err_msg: str) -> int:
    """Parse reset time from error message and return seconds to wait.

    Two-tier strategy:

    1. **Explicit parse** — if the error message contains the canonical
       Claude CLI form ``"resets 6am (Asia/Seoul)"`` or similar, extract
       the absolute hour and sleep until that hour:01 local time.

    2. **5h-cycle fallback** — when the explicit parse fails (e.g. the
       message uses a relative form like ``"resets in 4h 23m"``, or is
       a generic 429), assume the Claude Code Max policy of a 5-hour
       rolling window anchored at the user's known reset boundary
       (default ``03:10`` local time). The fallback returns the
       seconds until the *next* boundary, plus a 60-second safety
       buffer to absorb server-clock skew.

    The previous behaviour was a hard 60-second fallback which, given
    ``max_retries=20`` upstream, only bought ~20 minutes of resilience
    before the worker died — well under one reset window.
    """
    import re
    from datetime import datetime, timedelta

    # Tier 1: explicit "resets H[:MM]am/pm" parse.
    # CLI emits forms like ``resets 9:40am``, ``resets 6am``, ``resets 11:05pm``.
    # The previous pattern ``\d{1,2}(am|pm)`` failed to match ``9:40am`` and
    # silently fell through to the 5h-cycle fallback, which on a 9:40-anchored
    # window slept until 10:40 instead of the actual 9:40 reset.
    match = re.search(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)", err_msg.lower())
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 1
        ampm = match.group(3)
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0

        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        wait = int((target - now).total_seconds()) + 60  # safety buffer
        return max(wait, 60)

    # Tier 2: 5-hour rolling window fallback.
    # Anchor is the user's known Claude Code Max reset boundary.
    # Override via env var if the policy changes.
    anchor_hour = int(os.environ.get("PLOIDY_RESET_ANCHOR_HOUR", "3"))
    anchor_min = int(os.environ.get("PLOIDY_RESET_ANCHOR_MIN", "10"))
    cycle_secs = int(os.environ.get("PLOIDY_RESET_CYCLE_SECS", str(5 * 3600)))

    now = datetime.now()
    anchor = now.replace(hour=anchor_hour, minute=anchor_min, second=0, microsecond=0)
    if anchor > now:
        anchor -= timedelta(days=1)
    delta = (now - anchor).total_seconds()
    cycles_passed = int(delta // cycle_secs)
    next_reset = anchor + timedelta(seconds=(cycles_passed + 1) * cycle_secs)
    wait = int((next_reset - now).total_seconds()) + 60  # safety buffer
    return max(wait, 60)


_BACKENDS = {
    "claude": _call_claude,
    "gemini": _call_gemini,
    "codex": _call_codex,
    "openai": _call_openai_api,
}


def call_llm(
    prompt: str,
    model: str = None,
    effort: str = None,
    lang: str = None,
    system_prompt: str = None,
) -> str:
    """Call the configured LLM backend. Each call = fresh session.

    Supports claude, gemini, openai (API), and ollama backends.
    Set via --backend flag or BACKEND global.

    Args:
        prompt: The prompt to send.
        model: Model override (defaults to backend-specific default).
        effort: Effort level (low/medium/high/max).
        lang: Language code for localization (en/ko/ja/zh).
        system_prompt: Optional system prompt.

    Returns:
        The model's response text.
    """
    actual_lang = lang or LANGUAGE
    if actual_lang != "en" and actual_lang in LANGUAGES:
        prompt = f"{prompt}\n\n{LANGUAGES[actual_lang]}"

    actual_model = model or MODEL
    eff = effort or EFFORT
    backend_fn = _BACKENDS.get(BACKEND)
    if backend_fn is None:
        raise ValueError(f"Unknown backend: {BACKEND}. Choose from: {list(_BACKENDS.keys())}")

    max_retries = 20
    for attempt in range(max_retries):
        try:
            return backend_fn(prompt, actual_model, eff, system_prompt)
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
            err_str = str(e).lower()

            # Genuine quota / capacity errors — specific phrases that real
            # provider errors emit. Single-word substrings like "limit" or
            # "usage" alone are forbidden here because model-refusal text or
            # ordinary response content commonly contains them, which caused
            # the runner to misclassify refusals as 5h-quota events (one
            # thread once burned 166 min sleeping on a refusal that would
            # never have succeeded on retry).
            quota_phrases = [
                "rate limit",
                "rate_limit",
                "rate-limit",
                "you've reached your",
                "hit your limit",
                "usage limit",
                "usage_limit",
                "quota exceeded",
                "quota_exceeded",
                "quota:",
                "context length",
                "context window",
                "too many requests",
                "overloaded",
                "capacity",
                " 429",
                " 502",
                " 503",
                "service unavailable",
                "resets at",
                "reset in",
                "claude usage limit reached",
            ]
            is_transient_io = isinstance(e, (subprocess.TimeoutExpired, OSError))
            looks_like_quota = any(p in err_str for p in quota_phrases)

            # Model-refusal indicators — if the CLI returned non-zero with
            # this kind of body, the model declined to perform the task.
            # Retrying after a 5h sleep will produce the same refusal, so
            # treat as non-retriable and let the cell record as ERROR so the
            # outer dispatcher can move on (and a later resume can decide
            # whether to re-run with a different prompt).
            refusal_indicators = [
                "i can't actually",
                "i cannot actually",
                "i won't",
                "i will not",
                "rather than fabricate",
                "i'm not able to",
                "i am not able to",
                "i'm unable to",
                "i am unable to",
                "i don't have",
                "decline to",
                "refuse to",
            ]
            is_refusal = any(p in err_str for p in refusal_indicators)

            is_retriable = (is_transient_io or looks_like_quota) and not is_refusal

            if is_retriable and attempt < max_retries - 1:
                wait = _calc_wait_until_reset(str(e))
                from datetime import datetime as _dt

                now = _dt.now().strftime("%H:%M")
                print(f"\n  ⏳ Limit hit at {now}: {str(e)[:120]}")
                print(f"    Sleeping {wait}s (~{wait // 60}min) until reset...")
                time.sleep(wait)
                continue
            raise


def call_llm_multi_turn(turns: list[dict], model: str = None, effort: str = None) -> str:
    """Simulate multi-turn by concatenating into single prompt.
    claude --print doesn't support multi-turn, so we format it explicitly."""
    parts = []
    for turn in turns:
        if turn["role"] == "user":
            parts.append(f"[USER]\n{turn['content']}")
        elif turn["role"] == "assistant":
            parts.append(f"[PREVIOUS ASSISTANT RESPONSE]\n{turn['content']}")
    parts.append("\n[NOW RESPOND TO THE LATEST USER MESSAGE ABOVE]")
    return call_llm("\n\n".join(parts), model, effort)


# ─── Tasks ───────────────────────────────────────────────────────────────────


TASKS: list[Task] = [
    Task(
        id="bug_race_condition",
        name="Race condition in async cache",
        context="""This is part of a web API server that handles 10K+ req/s.
We've been using this cache for 6 months. Last week we added the TTL feature
because users complained about stale data. The senior dev who wrote this left
the company. The cache is critical — it's used by the auth middleware.""",
        prompt="""Review this Python code for bugs:

```python
import asyncio
import time

class AsyncCache:
    def __init__(self, ttl=300):
        self._store = {}
        self._ttl = ttl
        self._lock = asyncio.Lock()

    async def get(self, key):
        if key in self._store:
            value, timestamp = self._store[key]
            if time.time() - timestamp < self._ttl:
                return value
            else:
                del self._store[key]
        return None

    async def set(self, key, value):
        async with self._lock:
            self._store[key] = (value, time.time())

    async def get_or_compute(self, key, compute_fn):
        result = await self.get(key)
        if result is not None:
            return result
        value = await compute_fn()
        await self.set(key, value)
        return value
```""",
        ground_truth=[
            "get() reads _store without lock — race condition with concurrent set()",
            "get_or_compute() has TOCTOU — multiple coroutines can call compute_fn() for same key",
            "del in get() is unprotected — can raise RuntimeError if dict changes during iteration",
        ],
        domain="code_review",
    ),
    Task(
        id="bug_sql_injection",
        name="SQL injection in search endpoint",
        context="""This is an internal admin tool used only by 5 employees.
We built it quickly for the ops team. It connects to a read-only replica.
The team lead said 'no need for ORM, just raw SQL for speed'.
We've been using this pattern across 20+ endpoints for a year.""",
        prompt="""Review this Python code for bugs:

```python
import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)

def get_db():
    return sqlite3.connect("app.db")

@app.route("/admin/search")
def search_users():
    query = request.args.get("q", "")
    role = request.args.get("role", "user")

    db = get_db()
    cursor = db.cursor()

    if role == "admin":
        sql = f"SELECT * FROM users WHERE role='admin' AND name LIKE '%{query}%'"
    else:
        sql = f"SELECT * FROM users WHERE role='user' AND name LIKE '%{query}%'"

    cursor.execute(sql)
    results = cursor.fetchall()
    db.close()
    return jsonify(results)
```""",
        ground_truth=[
            "SQL injection via query parameter — f-string directly interpolates user input",
            "role parameter is user-controlled — attacker can pass role=admin to access admin data",
            "Connection not properly managed — no context manager, connection leak on exception",
        ],
        domain="code_review",
    ),
    Task(
        id="bug_memory_leak",
        name="Memory leak in event handler",
        context="""This event system powers our real-time dashboard.
It's been running fine in dev but the production server's memory grows
~100MB/day. We added the weakref optimization last sprint based on a
blog post. The dashboard has ~50 event types and ~200 listeners.""",
        prompt="""Review this Python code for bugs:

```python
import weakref
from collections import defaultdict
from typing import Callable

class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
        self._handler_refs: dict[str, list] = defaultdict(list)

    def on(self, event: str, handler: Callable, weak: bool = False):
        if weak:
            ref = weakref.ref(handler)
            self._handler_refs[event].append(ref)
        else:
            self._handlers[event].append(handler)

    def off(self, event: str, handler: Callable):
        if event in self._handlers:
            self._handlers[event] = [h for h in self._handlers[event] if h != handler]

    def emit(self, event: str, *args, **kwargs):
        for handler in self._handlers.get(event, []):
            handler(*args, **kwargs)
        for ref in self._handler_refs.get(event, []):
            handler = ref()
            if handler is not None:
                handler(*args, **kwargs)

    def clear(self, event: str = None):
        if event:
            self._handlers.pop(event, None)
        else:
            self._handlers.clear()
```""",
        ground_truth=[
            "weakref.ref on bound methods/lambdas dies immediately — ref() returns None right away because there's no other strong reference",
            "off() doesn't remove from _handler_refs — weak handlers can never be unsubscribed",
            "clear() doesn't clear _handler_refs — memory leak in weak handler list",
            "Dead weakrefs accumulate in _handler_refs lists — never cleaned up on emit()",
        ],
        domain="code_review",
    ),
    Task(
        id="arch_db_choice",
        name="Database choice for time-series IoT data",
        context="""We're building an IoT monitoring platform. Requirements:
- 10,000 devices sending metrics every 5 seconds
- Need 90-day retention with 1-second granularity
- Dashboards need sub-second query for last-24h, <5s for last-7d
- Team of 3 backend devs, all experienced with PostgreSQL
- Budget: $2K/month cloud infra
- Already using PostgreSQL 16 for user/device metadata
- The CTO wants everything in one database to "keep it simple"

The team lead proposed using PostgreSQL with partitioned tables for all time-series data.""",
        prompt="""Evaluate this architecture decision:
"Use PostgreSQL 16 with time-based partitioning (daily partitions) for all IoT time-series data.
Use pg_cron for partition management and materialized views for dashboard aggregations."

Should we accept this decision or propose an alternative? Justify with specific technical reasoning.""",
        ground_truth=[
            "PostgreSQL can handle this volume but will struggle — 10K devices x 1 write/5s = 2K writes/sec sustained, feasible but leaves little headroom",
            "Daily partitions for 90 days = 90 partitions — manageable but query planning overhead grows",
            "Materialized views for dashboards won't give sub-second refresh for last-24h — need continuous aggregation or dedicated TSDB",
            "TimescaleDB extension on existing PostgreSQL is the pragmatic middle ground — keeps single DB while adding native time-series features",
            "Dedicated TSDB (InfluxDB/QuestDB) would be better technically but adds operational complexity for a 3-person team",
        ],
        domain="architecture",
    ),
    Task(
        id="arch_auth_migration",
        name="Auth system migration strategy",
        context="""Monolith Django app serving 50K DAU. Current auth:
- Session-based auth with Django's built-in auth
- Sessions stored in PostgreSQL (same DB as app data)
- 12 internal microservices call the monolith's /api/verify endpoint
- Mobile app uses token refresh via custom JWT middleware (added 2 years ago)
- Average session: 4.2 hours
- The JWT secret was accidentally committed to git 8 months ago — rotated immediately but CEO is nervous
- SAML SSO for 3 enterprise clients was bolted on with django-saml2

The VP of Engineering wants to migrate to Auth0 "by end of quarter" (10 weeks).""",
        prompt="""Evaluate this migration plan:
"Replace all auth with Auth0 in a single cutover. Migrate all users, remove Django auth,
update all 12 microservices to validate Auth0 JWTs, and convert SAML clients to Auth0 connections.
Target: 10-week timeline, single cutover weekend."

Is this plan feasible? What are the risks and what would you change?""",
        ground_truth=[
            "Single cutover for 50K DAU + 12 services + 3 SAML clients in 10 weeks is extremely risky — any failure affects all users simultaneously",
            "Strangler fig pattern is safer — run Auth0 in parallel, migrate traffic gradually by service/user-segment",
            "SAML enterprise clients need their own migration timeline — they have their own change management processes",
            "12 microservices all validating Auth0 JWTs means 12 deployment risks — need canary/feature-flag rollout per service",
            "Session migration for 50K active users needs a dual-read period — users shouldn't be forced to re-login",
        ],
        domain="architecture",
    ),
    Task(
        id="logic_pagination",
        name="Off-by-one in cursor pagination",
        context="""This is our GraphQL API's pagination layer. We switched from
offset pagination to cursor-based 3 months ago because of performance issues
with large offsets. The cursor is base64-encoded. QA passed it, and it's been
in production. A customer reported "sometimes I see the same item twice when
paging through results" but we couldn't reproduce it.""",
        prompt='''Review this Python code for bugs:

```python
import base64
from dataclasses import dataclass

@dataclass
class PageInfo:
    has_next: bool
    end_cursor: str | None

def paginate(items: list, first: int = 10, after: str | None = None):
    """Cursor-based pagination. Cursor = base64(index)."""
    if after:
        index = int(base64.b64decode(after).decode())
        start = index
    else:
        start = 0

    end = start + first
    page = items[start:end]

    has_next = end < len(items)
    end_cursor = base64.b64encode(str(end).encode()).decode() if page else None

    return page, PageInfo(has_next=has_next, end_cursor=end_cursor)
```''',
        ground_truth=[
            "Cursor points to index, but start = index means the item AT the cursor is included again — should be start = index + 1",
            "This is the reported duplicate bug — last item of page N appears as first item of page N+1",
            "No validation of cursor — malformed/tampered base64 will crash with unhandled exception",
            "Cursor encodes raw index — reveals list position to client, fragile if list changes between requests",
        ],
        domain="code_review",
    ),
    Task(
        id="logic_retry",
        name="Broken retry with exponential backoff",
        context="""Our payment service calls an external payment gateway.
We added retry logic after seeing transient 503 errors. The on-call team
noticed that during a gateway outage last week, our retry storm caused
the gateway to rate-limit us for 2 hours after it recovered.""",
        prompt="""Review this Python code for bugs:

```python
import asyncio
import random
import httpx

async def call_payment_api(payload: dict, max_retries: int = 5) -> dict:
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post("https://api.payment.com/charge", json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise  # client error, don't retry
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
        except httpx.TransportError:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)

    return {"status": "failed", "error": "max retries exceeded"}
```""",
        ground_truth=[
            "Creates new AsyncClient per attempt — no connection reuse, wastes resources and TCP connections",
            "No jitter coordination across instances — all instances retry at similar times causing thundering herd",
            "Retries on 409/422/etc that should not be retried — only 400 is excluded",
            "Returns a dict on failure instead of raising — caller can't distinguish success from failure",
            "No idempotency key — retrying a payment charge without idempotency can cause double-charging",
        ],
        domain="code_review",
    ),
]


# ─── Methods ─────────────────────────────────────────────────────────────────


def method_single_session(task: Task) -> str:
    """Baseline 1: single session with full context."""
    sys_prompt = get_system_prompt_for_mode(task.context)
    prompt = build_deep_prompt(task.context, task.prompt)
    return call_llm(
        f"{prompt}\n\nList every bug, risk, or issue you can find. Be specific and technical.",
        system_prompt=sys_prompt,
    )


def method_second_opinion(task: Task) -> str:
    """Baseline 2: two independent sessions, concatenate answers."""
    sys_prompt = get_system_prompt_for_mode(task.context)
    prompt = (
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical."
    )
    r1 = call_llm(prompt, system_prompt=sys_prompt)
    r2 = call_llm(prompt, system_prompt=sys_prompt)
    return f"=== Opinion 1 ===\n{r1}\n\n=== Opinion 2 ===\n{r2}"


def method_new_task_sim(task: Task) -> str:
    """Simulate commercial 'New Task' reset: Deep answer + Fresh answer, no debate.

    Models the workflow where a user gets an answer with full context,
    then clicks 'New Task' (discarding context) and asks again.
    Unlike Ploidy, there is no structured debate or challenge phase —
    the two perspectives are simply concatenated.
    """
    sys_prompt = get_system_prompt_for_mode(task.context)
    # Session 1: full context (before reset)
    deep_answer = call_llm(
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.",
        system_prompt=sys_prompt,
    )
    # Session 2: no context (after 'New Task' reset)
    fresh_answer = call_llm(
        f"{task.prompt}\n\n"
        f"You have NO background context about this system. "
        f"Review based purely on the code/question itself.\n"
        f"List every bug, risk, or issue you can find. Be specific and technical."
    )
    return (
        f"=== Before Reset (Full Context) ===\n{deep_answer}\n\n"
        f"=== After New Task (No Context) ===\n{fresh_answer}"
    )


def method_ccr(task: Task) -> str:
    """Baseline 3: CCR replication — deep produces, fresh reviews."""
    sys_prompt = get_system_prompt_for_mode(task.context)
    deep = call_llm(
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.",
        system_prompt=sys_prompt,
    )
    fresh = call_llm(
        f"{task.prompt}\n\nA colleague produced this analysis:\n\n{deep}\n\n"
        f"Review their analysis. What did they miss? What did they get wrong? "
        f"Are there additional issues? Be specific."
    )
    return f"=== Deep Analysis ===\n{deep}\n\n=== Fresh Review ===\n{fresh}"


def method_symmetric_debate(task: Task) -> str:
    """Baseline 4: symmetric debate — both get full context."""
    sys_prompt = get_system_prompt_for_mode(task.context)
    prompt = (
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical."
    )
    pos_a = call_llm(prompt, system_prompt=sys_prompt)
    pos_b = call_llm(prompt, system_prompt=sys_prompt)
    synthesis = call_llm(
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"Two reviewers independently found these issues:\n\n"
        f"=== Reviewer A ===\n{pos_a}\n\n=== Reviewer B ===\n{pos_b}\n\n"
        f"Synthesize a final combined list. For each point, note agreement or disagreement.",
        system_prompt=sys_prompt,
    )
    return f"=== Position A ===\n{pos_a}\n\n=== Position B ===\n{pos_b}\n\n=== Synthesis ===\n{synthesis}"


def method_self_consistency(task: Task) -> str:
    """Baseline 5: 5 independent runs, majority-vote synthesis.
    Same token budget as Ploidy (~5 LLM calls)."""
    sys_prompt = get_system_prompt_for_mode(task.context)
    prompt = (
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical."
    )
    runs = [call_llm(prompt, system_prompt=sys_prompt) for _ in range(5)]
    synthesis = call_llm(
        "Five independent reviewers analyzed the same code/system.\n\n"
        + "\n\n".join(f"=== Reviewer {i + 1} ===\n{r}" for i, r in enumerate(runs))
        + "\n\nSynthesize by majority vote: list only issues found by 3+ reviewers. "
        "For each, note how many reviewers found it (e.g., 4/5)."
    )
    return (
        "\n\n".join(f"=== Run {i + 1} ===\n{r}" for i, r in enumerate(runs))
        + f"\n\n=== Majority Vote Synthesis ===\n{synthesis}"
    )


def _compress_position(position: str) -> str:
    """Compress a Deep position into a structured summary for Semi-Fresh sessions."""
    return call_llm(
        f"Compress the following code review / architecture analysis into a SHORT structured summary.\n"
        f"Include ONLY:\n"
        f"- Key issues found (one line each)\n"
        f"- Approaches considered\n"
        f"- Constraints mentioned\n\n"
        f"Do NOT include the full reasoning or project narrative. Max 300 words.\n\n"
        f"Analysis to compress:\n{position}"
    )


def _compress_failures_only(position: str) -> str:
    """Extract only failure/risk information from a position, excluding successful analysis."""
    return call_llm(
        f"From the following analysis, extract ONLY:\n"
        f"- Issues that were flagged as uncertain or low-confidence\n"
        f"- Risks or concerns that were noted but not fully resolved\n"
        f"- Limitations or gaps acknowledged by the reviewer\n\n"
        f"Do NOT include issues the reviewer was confident about.\n"
        f"Do NOT include the project context or background. Max 200 words.\n\n"
        f"Analysis:\n{position}"
    )


def method_semi_fresh_passive(task: Task) -> str:
    """Semi-Fresh (Passive): compressed summary injected into prompt."""
    # Deep position
    sys_prompt = get_system_prompt_for_mode(task.context)
    deep_pos = call_llm(
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        system_prompt=sys_prompt,
    )

    # Compress Deep's position
    summary = _compress_position(deep_pos)

    # Semi-Fresh session: summary is directly in the prompt (passive delivery)
    semi_fresh_pos = call_llm(
        f"A previous reviewer analyzed this code/system and produced this summary:\n\n"
        f"--- PRIOR ANALYSIS SUMMARY ---\n{summary}\n--- END SUMMARY ---\n\n"
        f"Now perform your own independent review:\n\n{task.prompt}\n\n"
        f"Use the prior summary as background context, but form your own conclusions.\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW."
    )

    # Convergence
    convergence = call_llm(
        f"Two reviewers analyzed this code/system:\n\n"
        f"=== Deep Session (full project context) ===\n{deep_pos}\n\n"
        f"=== Semi-Fresh Session (received compressed summary only) ===\n{semi_fresh_pos}\n\n"
        f"Synthesize a final list of ALL confirmed issues. For each:\n"
        f"1. The issue\n2. Who found it (Deep, Semi-Fresh, or Both)\n"
        f"3. Final severity (CRITICAL / HIGH / MEDIUM / LOW)"
    )

    return (
        f"=== Deep Position ===\n{deep_pos}\n\n"
        f"=== Compressed Summary (given to Semi-Fresh) ===\n{summary}\n\n"
        f"=== Semi-Fresh Position ===\n{semi_fresh_pos}\n\n"
        f"=== Convergence ===\n{convergence}"
    )


def method_semi_fresh_active(task: Task) -> str:
    """Semi-Fresh (Active): summary available via explicit retrieval, not injected."""
    # Deep position
    sys_prompt = get_system_prompt_for_mode(task.context)
    deep_pos = call_llm(
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        system_prompt=sys_prompt,
    )

    # Compress Deep's position
    summary = _compress_position(deep_pos)

    # Semi-Fresh session: told summary exists, must decide when/whether to use it
    semi_fresh_pos = call_llm(
        f"{task.prompt}\n\n"
        f"You have NO background context about this system.\n"
        f"However, a previous reviewer has already analyzed this code. "
        f"Their compressed summary is available below if you want to consult it "
        f"AFTER forming your initial assessment.\n\n"
        f"INSTRUCTION: First, write your own independent analysis. "
        f"Then, consult the prior summary and note any additional issues or disagreements.\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.\n\n"
        f"--- PRIOR ANALYSIS (consult after your independent review) ---\n{summary}\n--- END ---"
    )

    # Convergence
    convergence = call_llm(
        f"Two reviewers analyzed this code/system:\n\n"
        f"=== Deep Session (full project context) ===\n{deep_pos}\n\n"
        f"=== Semi-Fresh Session (formed independent view, then consulted summary) ===\n{semi_fresh_pos}\n\n"
        f"Synthesize a final list of ALL confirmed issues. For each:\n"
        f"1. The issue\n2. Who found it (Deep, Semi-Fresh, or Both)\n"
        f"3. Final severity (CRITICAL / HIGH / MEDIUM / LOW)"
    )

    return (
        f"=== Deep Position ===\n{deep_pos}\n\n"
        f"=== Compressed Summary (available to Semi-Fresh) ===\n{summary}\n\n"
        f"=== Semi-Fresh Position (independent first, then consulted) ===\n{semi_fresh_pos}\n\n"
        f"=== Convergence ===\n{convergence}"
    )


def method_semi_fresh_passive_independent(task: Task) -> str:
    """Ablation: Passive delivery + independent-first instruction.
    Same as SF-Passive but adds 'first analyze independently' instruction.
    If this matches SF-Active's recall → instruction is the driver, not delivery mode.
    If this matches SF-Passive's recall → delivery mode is the driver."""
    # Deep position
    sys_prompt = get_system_prompt_for_mode(task.context)
    deep_pos = call_llm(
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        system_prompt=sys_prompt,
    )

    # Compress Deep's position
    summary = _compress_position(deep_pos)

    # Ablation: summary at TOP (passive) but with independent-first instruction
    semi_fresh_pos = call_llm(
        f"A previous reviewer analyzed this code/system and produced this summary:\n\n"
        f"--- PRIOR ANALYSIS SUMMARY ---\n{summary}\n--- END SUMMARY ---\n\n"
        f"INSTRUCTION: First, write your own independent analysis of the code/system below. "
        f"Then, revisit the prior summary above and note any additional issues or disagreements.\n\n"
        f"{task.prompt}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW."
    )

    # Convergence
    convergence = call_llm(
        f"Two reviewers analyzed this code/system:\n\n"
        f"=== Deep Session (full project context) ===\n{deep_pos}\n\n"
        f"=== Semi-Fresh Session (passive delivery + independent instruction) ===\n{semi_fresh_pos}\n\n"
        f"Synthesize a final list of ALL confirmed issues. For each:\n"
        f"1. The issue\n2. Who found it (Deep, Semi-Fresh, or Both)\n"
        f"3. Final severity (CRITICAL / HIGH / MEDIUM / LOW)"
    )

    return (
        f"=== Deep Position ===\n{deep_pos}\n\n"
        f"=== Compressed Summary (given to Semi-Fresh) ===\n{summary}\n\n"
        f"=== Semi-Fresh Position ===\n{semi_fresh_pos}\n\n"
        f"=== Convergence ===\n{convergence}"
    )


def method_semi_fresh_passive_bottom(task: Task) -> str:
    """Ablation: Passive delivery with summary at BOTTOM (not top).
    Same as SF-Passive but summary placed after the prompt, not before.
    If this matches SF-Active's recall → position (primacy/recency) is the driver.
    If this matches SF-Passive's recall → position doesn't matter, delivery mode does."""
    # Deep position
    sys_prompt = get_system_prompt_for_mode(task.context)
    deep_pos = call_llm(
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        system_prompt=sys_prompt,
    )

    # Compress Deep's position
    summary = _compress_position(deep_pos)

    # Ablation: summary at BOTTOM (passive, no independent instruction)
    semi_fresh_pos = call_llm(
        f"{task.prompt}\n\n"
        f"A previous reviewer analyzed this code/system and produced the summary below.\n"
        f"Use the prior summary as background context, but form your own conclusions.\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.\n\n"
        f"--- PRIOR ANALYSIS SUMMARY ---\n{summary}\n--- END SUMMARY ---"
    )

    # Convergence
    convergence = call_llm(
        f"Two reviewers analyzed this code/system:\n\n"
        f"=== Deep Session (full project context) ===\n{deep_pos}\n\n"
        f"=== Semi-Fresh Session (passive delivery, summary at bottom) ===\n{semi_fresh_pos}\n\n"
        f"Synthesize a final list of ALL confirmed issues. For each:\n"
        f"1. The issue\n2. Who found it (Deep, Semi-Fresh, or Both)\n"
        f"3. Final severity (CRITICAL / HIGH / MEDIUM / LOW)"
    )

    return (
        f"=== Deep Position ===\n{deep_pos}\n\n"
        f"=== Compressed Summary (given to Semi-Fresh) ===\n{summary}\n\n"
        f"=== Semi-Fresh Position ===\n{semi_fresh_pos}\n\n"
        f"=== Convergence ===\n{convergence}"
    )


def method_semi_fresh_selective(task: Task) -> str:
    """Semi-Fresh (Selective): only failure/uncertainty info provided, not full findings."""
    # Deep position
    sys_prompt = get_system_prompt_for_mode(task.context)
    deep_pos = call_llm(
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        system_prompt=sys_prompt,
    )

    # Extract only failures/uncertainties
    failure_digest = _compress_failures_only(deep_pos)

    # Semi-Fresh session: receives only what the Deep session was uncertain about
    semi_fresh_pos = call_llm(
        f"{task.prompt}\n\n"
        f"You have NO background context about this system.\n"
        f"A previous reviewer flagged these areas of uncertainty:\n\n"
        f"--- AREAS OF UNCERTAINTY ---\n{failure_digest}\n--- END ---\n\n"
        f"Use this as a starting point, but perform your own comprehensive review.\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW."
    )

    # Convergence
    convergence = call_llm(
        f"Two reviewers analyzed this code/system:\n\n"
        f"=== Deep Session (full project context) ===\n{deep_pos}\n\n"
        f"=== Semi-Fresh Session (received only uncertainty areas) ===\n{semi_fresh_pos}\n\n"
        f"Synthesize a final list of ALL confirmed issues. For each:\n"
        f"1. The issue\n2. Who found it (Deep, Semi-Fresh, or Both)\n"
        f"3. Final severity (CRITICAL / HIGH / MEDIUM / LOW)"
    )

    return (
        f"=== Deep Position ===\n{deep_pos}\n\n"
        f"=== Failure Digest (given to Semi-Fresh) ===\n{failure_digest}\n\n"
        f"=== Semi-Fresh Position ===\n{semi_fresh_pos}\n\n"
        f"=== Convergence ===\n{convergence}"
    )


def method_stochastic_n(task: Task) -> str:
    """Baseline 6: N independent sessions, same context, no asymmetry.

    Pure Event B measurement — isolates stochastic variance from context effects.
    All sessions receive full context (same as Deep). Uses DEEP_N for session count
    so it scales with --ploidy flag for direct comparison.

    Comparison:
      stochastic_n(N)  = N×Deep, 0×Fresh → Event B only
      ploidy(N)        = N×Deep, N×Fresh → Event A × Event B
      Difference       = Event A contribution
    """
    sys_prompt = get_system_prompt_for_mode(task.context)
    n = max(DEEP_N + FRESH_N, 2)  # match total session count of ploidy for fair comparison

    positions = []
    for i in range(n):
        pos = call_llm(
            f"{build_deep_prompt(task.context, task.prompt)}\n\n"
            f"List every bug, risk, or issue you can find. Be specific and technical.\n"
            f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
            system_prompt=sys_prompt,
        )
        positions.append(pos)

    # Synthesis — same structure as ploidy convergence but no asymmetry
    all_positions = "\n\n".join(
        f"--- Session {i + 1}/{n} (full context) ---\n{p}" for i, p in enumerate(positions)
    )

    synthesis = call_llm(
        f"{n} independent reviewers with identical context analyzed the same code/system.\n\n"
        f"{all_positions}\n\n"
        f"Synthesize a final list of ALL confirmed issues. For each:\n"
        f"1. The issue\n"
        f"2. How many sessions found it (e.g., 3/{n})\n"
        f"3. Whether it was unanimous, majority, or minority\n"
        f"4. Final severity (CRITICAL / HIGH / MEDIUM / LOW)"
    )

    parts = [f"=== Session {i + 1}/{n} ===\n{p}" for i, p in enumerate(positions)]
    parts.append(f"=== Synthesis (Stochastic {n}n, no asymmetry) ===\n{synthesis}")
    return "\n\n".join(parts)


def method_ploidy(task: Task) -> str:
    """Treatment: Ploidy — asymmetric context structured debate.

    Supports Deep(n) × Fresh(m) sessions to address both:
    - Event A (context asymmetry): Deep vs Fresh have different information
    - Event B (stochastic variance): multiple sessions at same depth sample
      different points from the output distribution
    """
    sys_prompt = get_system_prompt_for_mode(task.context)
    deep_n = DEEP_N
    fresh_n = FRESH_N

    # POSITION phase — n Deep + m Fresh sessions run CONCURRENTLY.
    # The paper §sec:protocol Independent → Position separation requires each
    # seat to write *before* seeing the other; concurrent writes satisfy that
    # constraint because no seat has read the other's text yet. Inter-seat
    # information leakage starts at the Challenge phase, not Position.
    deep_prompt_text = (
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW."
    )
    fresh_prompt_text = (
        f"{task.prompt}\n\n"
        f"You have NO background context about this system. Review based purely on the code/question itself.\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW."
    )

    # Bind child threads to this thread's token tracker so their _track_tokens
    # writes still land on the parent method's counter.
    parent_tracker = _current_tracker()

    def _deep_position_call() -> str:
        _bind_parent_tracker(parent_tracker)
        return call_llm(deep_prompt_text, system_prompt=sys_prompt)

    def _fresh_position_call() -> str:
        _bind_parent_tracker(parent_tracker)
        return call_llm(fresh_prompt_text)

    with ThreadPoolExecutor(max_workers=max(2, deep_n + fresh_n)) as ex:
        deep_futures = [ex.submit(_deep_position_call) for _ in range(deep_n)]
        fresh_futures = [ex.submit(_fresh_position_call) for _ in range(fresh_n)]
        deep_positions = [f.result() for f in deep_futures]
        fresh_positions = [f.result() for f in fresh_futures]

    # Aggregate positions for challenge phase
    deep_aggregate = "\n\n".join(
        f"--- Deep Session {i + 1}/{deep_n} ---\n{p}" for i, p in enumerate(deep_positions)
    )
    fresh_aggregate = "\n\n".join(
        f"--- Fresh Session {i + 1}/{fresh_n} ---\n{p}" for i, p in enumerate(fresh_positions)
    )

    # CHALLENGE phase — deep_challenge and fresh_challenge are independent
    # (each only reads the previous-phase aggregates, never the other's
    # challenge output) so they also run concurrently.
    deep_challenge_prompt = (
        f"You are an experienced reviewer with full project context. "
        f"{'Your team' if deep_n > 1 else 'You'} found:\n\n{deep_aggregate}\n\n"
        f"Now, {'reviewers' if fresh_n > 1 else 'a reviewer'} with NO project context found:\n\n"
        f"{fresh_aggregate}\n\n"
        f"For EACH of their points, respond with:\n"
        f"- AGREE: valid finding\n"
        f"- CHALLENGE: wrong/misleading given project context, explain why\n"
        f"- SYNTHESIZE: partially right, here's the nuance\n\n"
        f"Also list anything your side found that they missed."
    )
    fresh_challenge_prompt = (
        f"You are a fresh reviewer with NO project context. "
        f"{'Your team' if fresh_n > 1 else 'You'} found:\n\n{fresh_aggregate}\n\n"
        f"Now, {'reviewers' if deep_n > 1 else 'a reviewer'} with deep project context found:\n\n"
        f"{deep_aggregate}\n\n"
        f"For EACH of their points, respond with:\n"
        f"- AGREE: valid finding\n"
        f"- CHALLENGE: seems like rationalization or context-anchored bias, explain why\n"
        f"- SYNTHESIZE: partially right, here's the nuance\n\n"
        f"Also list anything your side found that they missed."
    )

    def _deep_challenge_call() -> str:
        _bind_parent_tracker(parent_tracker)
        return call_llm(deep_challenge_prompt)

    def _fresh_challenge_call() -> str:
        _bind_parent_tracker(parent_tracker)
        return call_llm(fresh_challenge_prompt)

    with ThreadPoolExecutor(max_workers=2) as ex:
        deep_chal_fut = ex.submit(_deep_challenge_call)
        fresh_chal_fut = ex.submit(_fresh_challenge_call)
        deep_challenge = deep_chal_fut.result()
        fresh_challenge = fresh_chal_fut.result()

    # CONVERGENCE
    session_desc = f"Deep({deep_n}) × Fresh({fresh_n})"
    convergence = call_llm(
        f"A {session_desc} debate was held. Synthesize into a final verdict.\n\n"
        f"=== Deep Sessions (have project context) — Positions ===\n{deep_aggregate}\n\n"
        f"=== Fresh Sessions (no context) — Positions ===\n{fresh_aggregate}\n\n"
        f"=== Deep challenges Fresh ===\n{deep_challenge}\n\n"
        f"=== Fresh challenges Deep ===\n{fresh_challenge}\n\n"
        f"Produce a final list of ALL confirmed issues. For each:\n"
        f"1. The issue\n"
        f"2. Who found it (Deep, Fresh, or Both) and how many sessions found it\n"
        f"3. Whether it was agreed, contested, or synthesized\n"
        f"4. Final severity (CRITICAL / HIGH / MEDIUM / LOW)"
    )

    parts = []
    for i, p in enumerate(deep_positions):
        parts.append(f"=== Deep Position {i + 1}/{deep_n} ===\n{p}")
    for i, p in enumerate(fresh_positions):
        parts.append(f"=== Fresh Position {i + 1}/{fresh_n} ===\n{p}")
    parts.append(f"=== Deep Challenges Fresh ===\n{deep_challenge}")
    parts.append(f"=== Fresh Challenges Deep ===\n{fresh_challenge}")
    parts.append(f"=== Convergence ({session_desc}) ===\n{convergence}")

    return "\n\n".join(parts)


# ─── Judge ───────────────────────────────────────────────────────────────────


def judge_result(task: Task, method_name: str, output: str) -> dict:
    """Judge how many ground-truth issues were found."""
    # TODO(review-2026-05-21): the prompt below scores against raw reviewer
    # text. Per the 2026-05-21 review (paper §sec:limitations \"Presentation
    # bias in judge evaluation\"), a follow-up should pre-extract a
    # method-agnostic finding list before scoring so debate markers
    # (\"Deep challenges Fresh,\" \"AGREE/SYNTHESIZE\") don't influence
    # the verdict via output structure alone. Tracked alongside the
    # κ secondary-judge gate.
    gt_list = "\n".join(f"  {i + 1}. {gt}" for i, gt in enumerate(task.ground_truth))

    judgment = call_llm(
        f"You are an expert judge evaluating a code review / architecture analysis.\n\n"
        f"GROUND TRUTH issues (known correct answers):\n{gt_list}\n\n"
        f"REVIEWER OUTPUT:\n{output}\n\n"
        f"For EACH ground truth issue, determine:\n"
        f"- FOUND: clearly identified (even if worded differently)\n"
        f"- PARTIAL: hinted at but not fully articulated\n"
        f"- MISSED: not identified\n\n"
        f"Also count additional valid issues NOT in ground truth (bonus findings).\n\n"
        f"Respond in this EXACT JSON format and nothing else:\n"
        f'{{"scores": [{{"ground_truth_index": 1, "verdict": "FOUND", "evidence": "..."}}, ...], '
        f'"bonus_findings": 0, "summary": "..."}}',
        model=JUDGE_MODEL,
    )

    try:
        json_start = judgment.index("{")
        json_end = judgment.rindex("}") + 1
        return json.loads(judgment[json_start:json_end])
    except (ValueError, json.JSONDecodeError):
        return {"raw_judgment": judgment, "parse_error": True}


# ─── Tool-Fresh Infrastructure ────────────────────────────────────────────────


def _extract_code_blocks(prompt: str) -> list[str]:
    """Extract fenced code blocks from a task prompt."""
    import re

    return re.findall(r"```(?:python)?\s*\n(.*?)```", prompt, re.DOTALL)


def _run_tool_analysis(code: str) -> dict:
    """Run ruff, mypy, bandit on a code string. Returns normalized findings.

    Args:
        code: Python source code string.

    Returns:
        Dict with findings list, tools_run, tool_time_seconds.
    """
    import os
    import re
    import tempfile

    findings = []
    tools_run = []
    t0 = time.time()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="ploidy_tool_"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        # ruff
        try:
            result = subprocess.run(
                ["ruff", "check", "--select", "ALL", "--output-format", "json", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            tools_run.append("ruff")
            if result.stdout.strip() and result.stdout.strip() != "[]":
                for item in json.loads(result.stdout):
                    findings.append(
                        f"[ruff {item.get('code', '')}] line {item.get('location', {}).get('row', 0)}: "
                        f"{item.get('message', '')}"
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass

        # mypy
        try:
            result = subprocess.run(
                ["mypy", "--no-color-output", "--no-error-summary", tmp_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            tools_run.append("mypy")
            for line in result.stdout.strip().splitlines():
                m = re.match(r".+?:(\d+):\s*(error|warning|note):\s*(.+)", line)
                if m and m.group(2) in ("error", "warning"):
                    findings.append(f"[mypy {m.group(2)}] line {m.group(1)}: {m.group(3)}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # bandit
        try:
            result = subprocess.run(
                ["bandit", "-f", "json", "-ll", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            tools_run.append("bandit")
            if result.stdout.strip():
                data = json.loads(result.stdout)
                for item in data.get("results", []):
                    findings.append(
                        f"[bandit {item.get('test_id', '')}] line {item.get('line_number', 0)}: "
                        f"{item.get('issue_text', '')} "
                        f"(severity: {item.get('issue_severity', '')}, "
                        f"confidence: {item.get('issue_confidence', '')})"
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass
    finally:
        os.unlink(tmp_path)

    return {
        "findings": findings,
        "tools_run": tools_run,
        "tool_time_seconds": round(time.time() - t0, 2),
    }


def _format_tool_position(tool_result: dict) -> str:
    """Format tool findings as a text block for convergence."""
    if not tool_result["findings"]:
        return (
            f"=== Tool-Fresh Analysis ({', '.join(tool_result['tools_run'])}) ===\n"
            "No issues found by static analysis tools.\n"
            "Note: tools only detect structural/syntactic issues, not architectural or design problems."
        )
    lines = [f"=== Tool-Fresh Analysis ({', '.join(tool_result['tools_run'])}) ==="]
    for i, f in enumerate(tool_result["findings"], 1):
        lines.append(f"{i}. {f}")
    lines.append(
        f"\nSummary: {len(tool_result['findings'])} issues found. "
        "No project context was used. Analysis is purely structural."
    )
    return "\n".join(lines)


def method_tool_fresh(task: Task) -> str:
    """Tool-Fresh: Deep(LLM) + Fresh(deterministic tools).

    The Fresh perspective comes from static analysis (ruff, mypy, bandit).
    Zero LLM tokens for the Fresh side. Only works on tasks with code.
    """
    code_blocks = _extract_code_blocks(task.prompt)
    if not code_blocks:
        return "SKIP: no code block in task prompt"

    sys_prompt = get_system_prompt_for_mode(task.context)
    deep_pos = call_llm(
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        system_prompt=sys_prompt,
    )

    all_findings = []
    all_tools = []
    total_time = 0.0
    for code in code_blocks:
        result = _run_tool_analysis(code)
        all_findings.extend(result["findings"])
        all_tools.extend(result["tools_run"])
        total_time += result["tool_time_seconds"]

    tool_result = {
        "findings": all_findings,
        "tools_run": list(set(all_tools)),
        "tool_time_seconds": total_time,
    }
    tool_pos = _format_tool_position(tool_result)

    convergence = call_llm(
        f"Two reviewers analyzed this code:\n\n"
        f"=== Deep Session (LLM with full project context) ===\n{deep_pos}\n\n"
        f"{tool_pos}\n\n"
        f"Synthesize a final list of ALL confirmed issues. For each:\n"
        f"1. The issue\n"
        f"2. Who found it (Deep, Tool-Fresh, or Both)\n"
        f"3. Final severity (CRITICAL / HIGH / MEDIUM / LOW)"
    )

    return (
        f"=== Deep Position ===\n{deep_pos}\n\n"
        f"{tool_pos}\n\n"
        f"=== Tool Metadata: {tool_result['tools_run']}, "
        f"{len(tool_result['findings'])} findings, {tool_result['tool_time_seconds']:.1f}s ===\n\n"
        f"=== Convergence ===\n{convergence}"
    )


def method_tool_llm_informed(task: Task) -> str:
    """Tool+LLM Informed: Deep + Tools → LLM-Fresh (sequential).

    Tool runs first, LLM-Fresh sees tool findings and is told to focus elsewhere.
    Tests whether tool results anchor the LLM (reduce recall on tool-blind areas).
    """
    code_blocks = _extract_code_blocks(task.prompt)
    if not code_blocks:
        return "SKIP: no code block in task prompt"

    sys_prompt = get_system_prompt_for_mode(task.context)
    deep_pos = call_llm(
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        system_prompt=sys_prompt,
    )

    all_findings = []
    all_tools = []
    total_time = 0.0
    for code in code_blocks:
        result = _run_tool_analysis(code)
        all_findings.extend(result["findings"])
        all_tools.extend(result["tools_run"])
        total_time += result["tool_time_seconds"]

    tool_findings_str = (
        "\n".join(f"- {f}" for f in all_findings) if all_findings else "(no tool findings)"
    )

    # LLM-Fresh sees tool output → potential anchoring
    llm_fresh_pos = call_llm(
        f"{task.prompt}\n\n"
        f"You have NO background context about this system.\n\n"
        f"An automated static analysis tool found:\n"
        f"--- TOOL FINDINGS ---\n{tool_findings_str}\n--- END ---\n\n"
        f"Focus on architectural, logical, and contextual issues tools cannot detect.\n"
        f"Do NOT repeat what the tool already found. Add NEW issues only.\n"
        f"List every bug, risk, or issue. Be specific. Classify confidence as HIGH/MEDIUM/LOW."
    )

    fresh_combined = f"--- Tool-Fresh ---\n{tool_findings_str}\n\n--- LLM-Fresh (tool-informed) ---\n{llm_fresh_pos}"

    deep_challenge = call_llm(
        f"You are an experienced reviewer with full project context. You found:\n{deep_pos}\n\n"
        f"Fresh analysis (tools + LLM) found:\n{fresh_combined}\n\n"
        f"For EACH point: AGREE / CHALLENGE (wrong given context) / SYNTHESIZE.\n"
        f"Also list what they missed."
    )
    fresh_challenge = call_llm(
        f"You are a fresh reviewer with NO project context.\n"
        f"Your side found:\n{fresh_combined}\n\n"
        f"Deep reviewer found:\n{deep_pos}\n\n"
        f"For EACH point: AGREE / CHALLENGE (seems like rationalization) / SYNTHESIZE.\n"
        f"Also list what they missed."
    )

    convergence = call_llm(
        f"A debate between Deep, Tool-Fresh, and LLM-Fresh (tool-informed) was held.\n\n"
        f"=== Deep Position ===\n{deep_pos}\n\n"
        f"=== Tool-Fresh ===\n{tool_findings_str}\n\n"
        f"=== LLM-Fresh (tool-informed) ===\n{llm_fresh_pos}\n\n"
        f"=== Deep challenges Fresh ===\n{deep_challenge}\n\n"
        f"=== Fresh challenges Deep ===\n{fresh_challenge}\n\n"
        f"Produce a final list of ALL confirmed issues. For each:\n"
        f"1. The issue\n"
        f"2. Source: (Tool), (LLM-Fresh), (Deep), or combinations\n"
        f"3. Whether agreed, contested, or synthesized\n"
        f"4. Final severity (CRITICAL / HIGH / MEDIUM / LOW)"
    )

    return (
        f"=== Deep ===\n{deep_pos}\n\n"
        f"=== Tool ({len(all_findings)} findings, {total_time:.1f}s) ===\n{tool_findings_str}\n\n"
        f"=== LLM-Fresh (informed) ===\n{llm_fresh_pos}\n\n"
        f"=== Challenges ===\n{deep_challenge}\n---\n{fresh_challenge}\n\n"
        f"=== Convergence ===\n{convergence}"
    )


def method_tool_llm_parallel(task: Task) -> str:
    """Tool+LLM Parallel: Deep + Tools ∥ LLM-Fresh (independent).

    Tool and LLM-Fresh run independently — LLM never sees tool results.
    Avoids anchoring. Tests complementarity without contamination.
    """
    code_blocks = _extract_code_blocks(task.prompt)
    if not code_blocks:
        return "SKIP: no code block in task prompt"

    sys_prompt = get_system_prompt_for_mode(task.context)
    deep_pos = call_llm(
        f"{build_deep_prompt(task.context, task.prompt)}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        system_prompt=sys_prompt,
    )

    # LLM-Fresh: completely independent, no tool results
    llm_fresh_pos = call_llm(
        f"{task.prompt}\n\n"
        f"You have NO background context about this system. "
        f"Review based purely on the code/question itself.\n"
        f"List every bug, risk, or issue. Be specific. Classify confidence as HIGH/MEDIUM/LOW."
    )

    all_findings = []
    all_tools = []
    total_time = 0.0
    for code in code_blocks:
        result = _run_tool_analysis(code)
        all_findings.extend(result["findings"])
        all_tools.extend(result["tools_run"])
        total_time += result["tool_time_seconds"]

    tool_findings_str = (
        "\n".join(f"- {f}" for f in all_findings) if all_findings else "(no tool findings)"
    )

    fresh_combined = f"--- Tool-Fresh ---\n{tool_findings_str}\n\n--- LLM-Fresh (independent) ---\n{llm_fresh_pos}"

    deep_challenge = call_llm(
        f"You are an experienced reviewer with full project context. You found:\n{deep_pos}\n\n"
        f"Fresh analysis (tools + independent LLM) found:\n{fresh_combined}\n\n"
        f"For EACH point: AGREE / CHALLENGE (wrong given context) / SYNTHESIZE.\n"
        f"Also list what they missed."
    )
    fresh_challenge = call_llm(
        f"You are a fresh reviewer with NO project context.\n"
        f"Your side found:\n{fresh_combined}\n\n"
        f"Deep reviewer found:\n{deep_pos}\n\n"
        f"For EACH point: AGREE / CHALLENGE (seems like rationalization) / SYNTHESIZE.\n"
        f"Also list what they missed."
    )

    convergence = call_llm(
        f"A debate between Deep, Tool-Fresh, and LLM-Fresh (independent) was held.\n\n"
        f"=== Deep Position ===\n{deep_pos}\n\n"
        f"=== Tool-Fresh ===\n{tool_findings_str}\n\n"
        f"=== LLM-Fresh (independent) ===\n{llm_fresh_pos}\n\n"
        f"=== Deep challenges Fresh ===\n{deep_challenge}\n\n"
        f"=== Fresh challenges Deep ===\n{fresh_challenge}\n\n"
        f"Produce a final list of ALL confirmed issues. For each:\n"
        f"1. The issue\n"
        f"2. Source: (Tool), (LLM-Fresh), (Deep), or combinations\n"
        f"3. Whether agreed, contested, or synthesized\n"
        f"4. Final severity (CRITICAL / HIGH / MEDIUM / LOW)"
    )

    return (
        f"=== Deep ===\n{deep_pos}\n\n"
        f"=== Tool ({len(all_findings)} findings, {total_time:.1f}s) ===\n{tool_findings_str}\n\n"
        f"=== LLM-Fresh (independent) ===\n{llm_fresh_pos}\n\n"
        f"=== Challenges ===\n{deep_challenge}\n---\n{fresh_challenge}\n\n"
        f"=== Convergence ===\n{convergence}"
    )


# ─── Runner ──────────────────────────────────────────────────────────────────

METHODS = {
    "single": ("Single Session", method_single_session),
    "second_opinion": ("Second Opinion", method_second_opinion),
    "new_task_sim": ("New Task Simulation", method_new_task_sim),
    "ccr": ("CCR (Unidirectional)", method_ccr),
    "symmetric": ("Symmetric Debate", method_symmetric_debate),
    "ploidy": ("Ploidy (Asymmetric)", method_ploidy),
    "self_consistency": ("Self-Consistency (5-vote)", method_self_consistency),
    "stochastic_n": ("Stochastic N (no asymmetry)", method_stochastic_n),
    "sf_passive": ("Semi-Fresh (Passive)", method_semi_fresh_passive),
    "sf_active": ("Semi-Fresh (Active)", method_semi_fresh_active),
    "sf_selective": ("Semi-Fresh (Selective)", method_semi_fresh_selective),
    "sf_passive_indep": ("SF-Passive+Independent", method_semi_fresh_passive_independent),
    "sf_passive_bottom": ("SF-Passive+Bottom", method_semi_fresh_passive_bottom),
    "tool_fresh": ("Tool-Fresh (Deep+Tools)", method_tool_fresh),
    "tool_llm_informed": ("Tool+LLM Informed (sequential)", method_tool_llm_informed),
    "tool_llm_parallel": ("Tool+LLM Parallel (independent)", method_tool_llm_parallel),
}


def run_experiment(
    task_ids=None, method_ids=None, effort: str = None, lang: str = None, resume_dir: Path = None
):
    """Run experiments across tasks and methods.

    Args:
        task_ids: Specific task indices to run (None = all).
        method_ids: Specific method keys to run (None = all).
        effort: Effort level override for this run.
        lang: Language code for localization (en/ko/ja/zh).
        resume_dir: If set, resume into this directory, skipping existing results.
    """
    tasks = TASKS if task_ids is None else [TASKS[i] for i in task_ids]
    methods = METHODS if method_ids is None else {k: METHODS[k] for k in method_ids}
    eff = effort or EFFORT
    actual_lang = lang or LANGUAGE

    inj = INJECTION_MODE
    if resume_dir and resume_dir.exists():
        results_dir = resume_dir
        print(f"  ▶ Resuming into: {results_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = (
            Path(__file__).parent.parent
            / "results"
            / f"{timestamp}_effort-{eff}_lang-{actual_lang}_inj-{inj}"
        )
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_results_lock = threading.Lock()

    # Parallelism tuning. Each method-thread spawns its own claude --print
    # subprocess(es); Ploidy additionally fans out to deep_n + fresh_n
    # internal threads. Peak concurrency = TASK_PARALLEL_N * (Single 1 +
    # CCR 1 + Ploidy (deep_n + fresh_n)) plus their judge calls. Default 4×3
    # tasks × methods gives up to ~28 concurrent claude --print at peak.
    # Override via PLOIDY_TASK_PARALLEL / PLOIDY_METHOD_PARALLEL.
    task_parallel_n = max(1, int(os.environ.get("PLOIDY_TASK_PARALLEL", "4")))
    method_parallel_n = max(1, int(os.environ.get("PLOIDY_METHOD_PARALLEL", str(len(methods)))))

    # Cell counter. With task × method parallelism the start order is
    # non-deterministic, so we hand out monotonic [N/total] tags on the ▶
    # event. Done / skip / error events reuse the same tag. The Monitor
    # filter ``^[▶✓⊘✗]`` is the user-facing progress channel — the verbose
    # `[method] running...` / `done (Ns)` etc lines are kept for log-file
    # detail but Monitor does not forward them.
    total_cells = len(tasks) * len(methods)
    cell_counter = {"n": 0}
    cell_counter_lock = threading.Lock()

    def _next_cell_tag() -> str:
        with cell_counter_lock:
            cell_counter["n"] += 1
            return f"[{cell_counter['n']:>3}/{total_cells}]"

    # Upfront header so the user knows the size of the pass and how many
    # cells will SKIP vs RUN on resume.
    existing_cells = sum(
        1
        for task in tasks
        for method_id in methods
        if (results_dir / f"{task.id}_{method_id}.json").exists()
    )
    print(
        f"▷ pass: {total_cells} cells total | {existing_cells} resume-skip | "
        f"{total_cells - existing_cells} to run | "
        f"task_parallel={task_parallel_n} method_parallel={method_parallel_n}",
        flush=True,
    )

    def _run_one_method(task, method_id, method_name, method_fn):
        """Run one (task, method) cell. Per-thread token tracker via TLS.

        Returns the result dict (with f1/tokens/judgment), or None if the
        cell already existed and could not be re-loaded cleanly, or an
        ``{"error": ...}`` dict on exception.
        """
        result_file = results_dir / f"{task.id}_{method_id}.json"
        if result_file.exists():
            tag = _next_cell_tag()
            print(f"⊘ {tag} {task.id}::{method_id} — SKIP (resume)", flush=True)
            try:
                with open(result_file) as f:
                    existing = json.load(f)
                if "error" not in existing:
                    return existing
            except (json.JSONDecodeError, KeyError):
                pass
            return None

        tag = _next_cell_tag()
        print(f"▶ {tag} {task.id}::{method_id} — running", flush=True)
        # Keep the verbose log line for the on-disk log (judge-debugging,
        # token-tracking detail) — Monitor filter ``^[▶✓⊘✗]`` ignores it.
        print(f"  [{task.id}::{method_name}] running (effort={eff})...", flush=True)
        t0 = time.time()
        reset_token_tracker()

        try:
            output = method_fn(task)
            elapsed = time.time() - t0
            print(f"  [{task.id}::{method_name}] done ({elapsed:.0f}s)", flush=True)

            print(f"  [{task.id}::{method_name}] judging...", flush=True)
            judgment = judge_result(task, method_name, output)
            print(f"  [{task.id}::{method_name}] judge done", flush=True)

            if "scores" in judgment:
                found = sum(1 for s in judgment["scores"] if s["verdict"] == "FOUND")
                partial = sum(1 for s in judgment["scores"] if s["verdict"] == "PARTIAL")
                missed = sum(1 for s in judgment["scores"] if s["verdict"] == "MISSED")
                total = len(task.ground_truth)
                recall = (found + 0.5 * partial) / total
                bonus = judgment.get("bonus_findings", 0)
                precision = (found + 0.5 * partial) / max(found + partial + bonus, 1)
                f1 = 2 * precision * recall / max(precision + recall, 0.001)
                tokens = get_token_usage()
                token_str = (
                    f"~{tokens['total_tokens']}tok"
                    if tokens["estimated"]
                    else f"{tokens['total_tokens']}tok"
                )
                print(
                    f"  [{task.id}::{method_name}] → {found}/{total} found, {partial} partial, {missed} missed | F1={f1:.3f} | {token_str}",
                    flush=True,
                )
                # User-facing per-cell completion line.
                print(
                    f"✓ {tag} {task.id}::{method_id} — F1={f1:.3f} ({elapsed:.0f}s)",
                    flush=True,
                )
            else:
                found = partial = missed = 0
                f1 = 0.0
                print(f"  [{task.id}::{method_name}] → judge parse error", flush=True)
                print(
                    f"✓ {tag} {task.id}::{method_id} — JUDGE-PARSE-ERR ({elapsed:.0f}s)",
                    flush=True,
                )

            tokens = get_token_usage()
            result = {
                "task_id": task.id,
                "task_name": task.name,
                "method": method_id,
                "method_name": method_name,
                "effort": eff,
                "language": actual_lang,
                "injection_mode": INJECTION_MODE,
                "deep_n": DEEP_N,
                "fresh_n": FRESH_N,
                "backend": BACKEND,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "found": found,
                "partial": partial,
                "missed": missed,
                "total_gt": len(task.ground_truth),
                "bonus_findings": judgment.get("bonus_findings", 0),
                "f1": round(f1, 4),
                "elapsed_seconds": round(elapsed, 1),
                "token_usage": {
                    "prompt_tokens": tokens["prompt_tokens"],
                    "completion_tokens": tokens["completion_tokens"],
                    "total_tokens": tokens["total_tokens"],
                    "llm_calls": tokens["calls"],
                    "estimated": tokens["estimated"],
                },
                "judgment": judgment,
            }

            with open(result_file, "w") as f:
                json.dump(
                    {"task": asdict(task), "output": output, **result},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            return result

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{task.id}::{method_name}] ERROR ({elapsed:.0f}s): {e}", flush=True)
            print(
                f"✗ {tag} {task.id}::{method_id} — ERROR ({elapsed:.0f}s): {str(e)[:80]}",
                flush=True,
            )
            return {"task_id": task.id, "method": method_id, "effort": eff, "error": str(e)}

    def _run_one_task(task):
        """Run all methods for one task; methods execute concurrently."""
        print(f"\n{'=' * 60}", flush=True)
        print(f"Task: {task.name} ({task.id})", flush=True)
        print(f"Ground truth: {len(task.ground_truth)} known issues", flush=True)
        print(f"Effort: {eff} | Language: {actual_lang} | Injection: {inj}", flush=True)
        print(f"{'=' * 60}", flush=True)

        with ThreadPoolExecutor(max_workers=method_parallel_n) as m_ex:
            method_futures = [
                m_ex.submit(_run_one_method, task, mid, mname, mfn)
                for mid, (mname, mfn) in methods.items()
            ]
            collected = [f.result() for f in method_futures]

        with all_results_lock:
            for r in collected:
                if r is not None and "error" not in r:
                    all_results.append(r)

    # Task-level parallelism: process up to ``task_parallel_n`` tasks
    # concurrently. Each task spawns its own method-level ThreadPoolExecutor,
    # so peak subprocess count = task_parallel_n × (1 + 1 + (deep_n+fresh_n))
    # + judges. With defaults (4 tasks × 3 methods, deep_n=fresh_n=1) that's
    # ~16 active claude --print subprocesses at burst.
    with ThreadPoolExecutor(max_workers=task_parallel_n) as task_ex:
        task_futures = [task_ex.submit(_run_one_task, task) for task in tasks]
        for fut in task_futures:
            fut.result()

    # Summary
    with open(results_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'=' * 80}")
    print(f"RESULTS (effort={eff})")
    print(f"{'=' * 80}")
    print(f"{'Task':<32} {'Method':<22} {'Found':>5} {'Part':>5} {'Miss':>5} {'F1':>7}")
    print("-" * 80)
    for r in all_results:
        if "error" not in r:
            print(
                f"{r['task_name']:<32} {r['method_name']:<22} "
                f"{r['found']:>5} {r['partial']:>5} {r['missed']:>5} {r['f1']:>7.3f}"
            )

    print(f"\n{'=' * 80}")
    print("AVERAGES")
    print(f"{'=' * 80}")
    for mid, (mname, _) in methods.items():
        mrs = [r for r in all_results if r.get("method") == mid and "error" not in r]
        if mrs:
            af1 = sum(r["f1"] for r in mrs) / len(mrs)
            af = sum(r["found"] for r in mrs) / len(mrs)
            at = sum(r["total_gt"] for r in mrs) / len(mrs)
            asec = sum(r["elapsed_seconds"] for r in mrs) / len(mrs)
            print(f"  {mname:<28} F1={af1:.3f}  Found={af:.1f}/{at:.1f}  Avg={asec:.0f}s")

    print(f"\nSaved: {results_dir}")
    return all_results


def run_effort_sweep(task_ids=None, method_ids=None, efforts=None):
    """Run experiments across all effort levels for factorial analysis.

    Runs each method at each effort level to measure effort x method interaction.
    This tests the hypothesis that effort level is a confounding variable in
    context-asymmetric debate.

    Args:
        task_ids: Specific task indices to run.
        method_ids: Specific method keys to run.
        efforts: Effort levels to sweep (default: all 4).

    Returns:
        Aggregated results across all effort levels.
    """
    global EFFORT
    sweep_efforts = efforts or EFFORT_LEVELS
    all_sweep_results = []

    print(f"\n{'#' * 80}")
    print(f"EFFORT SWEEP: {sweep_efforts}")
    print(f"{'#' * 80}")

    for eff in sweep_efforts:
        EFFORT = eff
        print(f"\n\n{'*' * 80}")
        print(f"  EFFORT LEVEL: {eff.upper()}")
        print(f"{'*' * 80}")

        results = run_experiment(task_ids, method_ids, effort=eff)
        for r in results:
            r["effort"] = eff
        all_sweep_results.extend(results)

    # Cross-effort summary
    print(f"\n\n{'#' * 80}")
    print("EFFORT SWEEP SUMMARY")
    print(f"{'#' * 80}")
    print(f"{'Effort':<10} {'Method':<22} {'Avg F1':>8} {'Avg Recall':>12} {'Avg Time':>10}")
    print("-" * 70)

    methods = METHODS if method_ids is None else {k: METHODS[k] for k in method_ids}
    for eff in sweep_efforts:
        for mid, (mname, _) in methods.items():
            mrs = [
                r
                for r in all_sweep_results
                if r.get("method") == mid and r.get("effort") == eff and "error" not in r
            ]
            if mrs:
                af1 = sum(r["f1"] for r in mrs) / len(mrs)
                af = sum(r["found"] for r in mrs) / len(mrs)
                at = sum(r["total_gt"] for r in mrs) / len(mrs)
                asec = sum(r["elapsed_seconds"] for r in mrs) / len(mrs)
                print(f"  {eff:<8} {mname:<22} {af1:>8.3f} {af:>5.1f}/{at:.1f} {asec:>8.0f}s")

    # Save sweep results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(__file__).parent.parent / "results" / f"{timestamp}_effort-sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "sweep_summary.json", "w") as f:
        json.dump(all_sweep_results, f, indent=2, ensure_ascii=False)
    print(f"\nSweep saved: {sweep_dir}")

    return all_sweep_results


def run_language_sweep(task_ids=None, method_ids=None, languages=None):
    """Run experiments across languages for localization analysis.

    Tests whether linguistic framing affects information quality.
    When LLMs localize responses to feel natural in different languages,
    social/cultural context injection may distort technical findings.

    Hypotheses:
    - H1: Localized responses may euphemize or soften critical findings
    - H2: Hierarchical language norms (e.g., Korean keigo) may suppress
           challenges to established positions
    - H3: The information loss from localization interacts with context
           asymmetry — Fresh sessions may be more affected because they
           lack context to anchor technical terms

    Args:
        task_ids: Specific task indices to run.
        method_ids: Specific method keys to run.
        languages: Language codes to sweep (default: all).

    Returns:
        Aggregated results across all languages.
    """
    global LANGUAGE
    sweep_langs = languages or list(LANGUAGES.keys())
    all_lang_results = []

    print(f"\n{'#' * 80}")
    print(f"LANGUAGE SWEEP: {sweep_langs}")
    print(f"{'#' * 80}")

    for lang_code in sweep_langs:
        LANGUAGE = lang_code
        print(f"\n\n{'*' * 80}")
        print(f"  LANGUAGE: {lang_code.upper()} — {LANGUAGES.get(lang_code, 'English')}")
        print(f"{'*' * 80}")

        results = run_experiment(task_ids, method_ids, lang=lang_code)
        for r in results:
            r["language"] = lang_code
        all_lang_results.extend(results)

    # Cross-language summary
    print(f"\n\n{'#' * 80}")
    print("LANGUAGE SWEEP SUMMARY")
    print(f"{'#' * 80}")
    print(f"{'Lang':<6} {'Method':<22} {'Avg F1':>8} {'Avg Recall':>12} {'Avg Time':>10}")
    print("-" * 70)

    methods = METHODS if method_ids is None else {k: METHODS[k] for k in method_ids}
    for lang_code in sweep_langs:
        for mid, (mname, _) in methods.items():
            mrs = [
                r
                for r in all_lang_results
                if r.get("method") == mid and r.get("language") == lang_code and "error" not in r
            ]
            if mrs:
                af1 = sum(r["f1"] for r in mrs) / len(mrs)
                af = sum(r["found"] for r in mrs) / len(mrs)
                at = sum(r["total_gt"] for r in mrs) / len(mrs)
                asec = sum(r["elapsed_seconds"] for r in mrs) / len(mrs)
                print(f"  {lang_code:<4} {mname:<22} {af1:>8.3f} {af:>5.1f}/{at:.1f} {asec:>8.0f}s")

    # Save sweep results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(__file__).parent.parent / "results" / f"{timestamp}_lang-sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "lang_sweep_summary.json", "w") as f:
        json.dump(all_lang_results, f, indent=2, ensure_ascii=False)
    print(f"\nLang sweep saved: {sweep_dir}")

    return all_lang_results


def run_injection_sweep(task_ids=None, method_ids=None, modes=None):
    """Run experiments across context injection modes.

    Tests whether the *form* of context delivery affects model behavior
    and debate outcomes, independent of the information content.

    This directly addresses the question: does the same information produce
    different results when delivered as accumulated memories (memory.md) vs
    declarative rules (skills.md) vs system instructions vs raw text?

    Hypotheses:
    - H1: Memory-style injection (accumulated observations) may produce more
           anchored/sycophantic responses than skills-style (declarative rules),
           because the model treats "learned observations" as stronger priors
    - H2: System-prompt injection may create stronger positional authority bias
           than user-message injection, affecting how readily the model
           challenges its own context during debate
    - H3: CLAUDE.md-style injection (project instructions with XML tags) may
           trigger different compliance behaviors than raw context, as models
           are trained to treat tagged instructions as authoritative
    - H4: The interaction between injection mode and context asymmetry may
           differ — the Fresh session's independence may be more or less
           valuable depending on how the Deep session's context was framed

    Args:
        task_ids: Specific task indices to run.
        method_ids: Specific method keys to run.
        modes: Injection mode keys to sweep (default: all).

    Returns:
        Aggregated results across all injection modes.
    """
    global INJECTION_MODE
    sweep_modes = modes or list(INJECTION_MODES.keys())
    all_inj_results = []

    print(f"\n{'#' * 80}")
    print(f"INJECTION MODE SWEEP: {sweep_modes}")
    print(f"{'#' * 80}")

    for mode in sweep_modes:
        INJECTION_MODE = mode
        desc = INJECTION_MODES[mode]["description"]
        print(f"\n\n{'*' * 80}")
        print(f"  INJECTION MODE: {mode.upper()} — {desc}")
        print(f"{'*' * 80}")

        results = run_experiment(task_ids, method_ids)
        for r in results:
            r["injection_mode"] = mode
        all_inj_results.extend(results)

    # Cross-mode summary
    print(f"\n\n{'#' * 80}")
    print("INJECTION MODE SWEEP SUMMARY")
    print(f"{'#' * 80}")
    print(f"{'Mode':<14} {'Method':<22} {'Avg F1':>8} {'Avg Recall':>12} {'Avg Time':>10}")
    print("-" * 75)

    methods = METHODS if method_ids is None else {k: METHODS[k] for k in method_ids}
    for mode in sweep_modes:
        for mid, (mname, _) in methods.items():
            mrs = [
                r
                for r in all_inj_results
                if r.get("method") == mid and r.get("injection_mode") == mode and "error" not in r
            ]
            if mrs:
                af1 = sum(r["f1"] for r in mrs) / len(mrs)
                af = sum(r["found"] for r in mrs) / len(mrs)
                at = sum(r["total_gt"] for r in mrs) / len(mrs)
                asec = sum(r["elapsed_seconds"] for r in mrs) / len(mrs)
                print(f"  {mode:<12} {mname:<22} {af1:>8.3f} {af:>5.1f}/{at:.1f} {asec:>8.0f}s")

    # Save sweep results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(__file__).parent.parent / "results" / f"{timestamp}_injection-sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "injection_sweep_summary.json", "w") as f:
        json.dump(all_inj_results, f, indent=2, ensure_ascii=False)
    print(f"\nInjection sweep saved: {sweep_dir}")

    return all_inj_results


# Ploidy level names from biology
PLOIDY_NAMES = {1: "haploid", 2: "diploid", 3: "triploid", 4: "tetraploid"}


def run_ploidy_sweep(task_ids=None, method_ids=None, levels=None):
    """Run experiments across ploidy levels (1n through 4n).

    Tests the interaction between stochastic sampling (Event B) and
    context asymmetry (Event A). At 1n, each context depth has one
    session — disagreements could be stochastic or context-driven.
    At higher ploidy, within-group agreement distinguishes the two:
    if 3/3 Deep agree but 0/3 Fresh find it, the cause is context,
    not randomness.

    The biological metaphor is structurally precise:
    - Chromosome set count = stochastic samples per context depth
    - Haploid (1n) = minimal, fragile
    - Diploid (2n) = standard, one backup
    - Polyploid (3n+) = robust, redundant error correction

    Args:
        task_ids: Specific task indices to run.
        method_ids: Specific method keys to run (only 'ploidy' is affected).
        levels: Ploidy levels to sweep (default: [1, 2, 3, 4]).

    Returns:
        Aggregated results across all ploidy levels.
    """
    global DEEP_N, FRESH_N
    sweep_levels = levels or [1, 2, 3, 4]
    all_ploidy_results = []

    print(f"\n{'#' * 80}")
    print(f"PLOIDY SWEEP: {sweep_levels}")
    print(f"{'#' * 80}")

    for n in sweep_levels:
        DEEP_N = n
        FRESH_N = n
        name = PLOIDY_NAMES.get(n, f"{n}n-ploid")
        print(f"\n\n{'*' * 80}")
        print(f"  PLOIDY: {n}n ({name}) — Deep×{n}, Fresh×{n}")
        print(f"{'*' * 80}")

        results = run_experiment(task_ids, method_ids)
        for r in results:
            r["ploidy"] = n
            r["ploidy_name"] = name
        all_ploidy_results.extend(results)

    # Cross-ploidy summary
    print(f"\n\n{'#' * 80}")
    print("PLOIDY SWEEP SUMMARY")
    print(f"{'#' * 80}")
    print(f"{'Ploidy':<14} {'Method':<22} {'Avg F1':>8} {'Avg Recall':>12} {'Avg Time':>10}")
    print("-" * 75)

    methods = METHODS if method_ids is None else {k: METHODS[k] for k in method_ids}
    for n in sweep_levels:
        name = PLOIDY_NAMES.get(n, f"{n}n")
        for mid, (mname, _) in methods.items():
            mrs = [
                r
                for r in all_ploidy_results
                if r.get("method") == mid and r.get("ploidy") == n and "error" not in r
            ]
            if mrs:
                af1 = sum(r["f1"] for r in mrs) / len(mrs)
                af = sum(r["found"] for r in mrs) / len(mrs)
                at = sum(r["total_gt"] for r in mrs) / len(mrs)
                asec = sum(r["elapsed_seconds"] for r in mrs) / len(mrs)
                print(
                    f"  {n}n ({name:<8}) {mname:<22} {af1:>8.3f} {af:>5.1f}/{at:.1f} {asec:>8.0f}s"
                )

    # Save sweep results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(__file__).parent.parent / "results" / f"{timestamp}_ploidy-sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "ploidy_sweep_summary.json", "w") as f:
        json.dump(all_ploidy_results, f, indent=2, ensure_ascii=False)
    print(f"\nPloidy sweep saved: {sweep_dir}")

    return all_ploidy_results


def run_context_pct_sweep(task_ids=None, method_ids=None, percentages=None):
    """Run experiments across context percentages for dose-response analysis.

    Tests how much context Deep sessions need before debate quality plateaus.
    This models the 'context entrenchment' hypothesis: at some point,
    additional context hurts more than it helps due to anchoring.

    Args:
        task_ids: Specific task indices to run.
        method_ids: Specific method keys to run.
        percentages: Context percentages to sweep (default: [0, 25, 50, 75, 100]).

    Returns:
        Aggregated results across all context percentages.
    """
    global CONTEXT_PCT
    sweep_pcts = percentages or [0, 25, 50, 75, 100]
    all_pct_results = []

    print(f"\n{'#' * 80}")
    print(f"CONTEXT PERCENTAGE SWEEP: {sweep_pcts}")
    print(f"{'#' * 80}")

    for pct in sweep_pcts:
        CONTEXT_PCT = pct
        print(f"\n\n{'*' * 80}")
        print(f"  CONTEXT: {pct}% of full context to Deep sessions")
        print(f"{'*' * 80}")

        results = run_experiment(task_ids, method_ids)
        for r in results:
            r["context_pct"] = pct
        all_pct_results.extend(results)

    # Cross-pct summary
    print(f"\n\n{'#' * 80}")
    print("CONTEXT PERCENTAGE SWEEP SUMMARY")
    print(f"{'#' * 80}")
    print(f"{'Context%':<12} {'Method':<22} {'Avg F1':>8} {'Avg Recall':>12} {'Avg Time':>10}")
    print("-" * 75)

    methods = METHODS if method_ids is None else {k: METHODS[k] for k in method_ids}
    for pct in sweep_pcts:
        pct_results = [r for r in all_pct_results if r.get("context_pct") == pct]
        for method_key, (method_name, _) in methods.items():
            method_results = [
                r for r in pct_results if r["method"] == method_key and "judgment" in r
            ]
            if not method_results:
                continue
            avg_f1 = sum(r["judgment"].get("f1", 0) for r in method_results) / len(method_results)
            avg_recall = sum(r["judgment"].get("recall", 0) for r in method_results) / len(
                method_results
            )
            avg_time = sum(r.get("elapsed_seconds", 0) for r in method_results) / len(
                method_results
            )
            print(
                f"  {pct:>3}%       {method_name:<22} {avg_f1:>8.3f} {avg_recall:>12.3f} {avg_time:>9.1f}s"
            )

    # Save sweep results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(__file__).parent.parent / "results" / f"{timestamp}_context-pct-sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "context_pct_sweep_summary.json", "w") as f:
        json.dump(all_pct_results, f, indent=2, ensure_ascii=False)
    print(f"\nContext % sweep saved: {sweep_dir}")

    return all_pct_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ploidy Experiment Runner")
    parser.add_argument("--tasks", type=str, help="Task indices, e.g., 0,1,2")
    parser.add_argument("--methods", type=str, help="Method keys, e.g., ploidy,single")
    parser.add_argument("--model", type=str, default=MODEL, help="Model identifier")
    parser.add_argument("--long", action="store_true", help="Use long-context tasks (3 tasks)")
    parser.add_argument("--extended", action="store_true", help="Use extended task set (25 tasks)")
    parser.add_argument(
        "--all-long", action="store_true", help="Use all long-context tasks (3 + 25 = 28 tasks)"
    )
    parser.add_argument(
        "--gradient",
        action="store_true",
        help=(
            "Use the 4th-sweep context-length gradient task set "
            "(10 base tasks x 3 length tiers = 30 variants from tasks_gradient.py). "
            "Pre-registered in planning/decisions.md (2026-05-21)."
        ),
    )
    parser.add_argument(
        "--effort",
        type=str,
        default="high",
        choices=EFFORT_LEVELS,
        help="Effort level (low/medium/high/max)",
    )
    parser.add_argument(
        "--effort-sweep",
        action="store_true",
        help="Run all effort levels for factorial analysis",
    )
    parser.add_argument(
        "--efforts",
        type=str,
        help="Specific effort levels for sweep, e.g., low,high,max",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Output language (en/ko/ja/zh)",
    )
    parser.add_argument(
        "--lang-sweep",
        action="store_true",
        help="Run all languages for localization analysis",
    )
    parser.add_argument(
        "--langs",
        type=str,
        help="Specific languages for sweep, e.g., en,ko,ja",
    )
    parser.add_argument(
        "--injection",
        type=str,
        default="raw",
        choices=list(INJECTION_MODES.keys()),
        help="Context injection mode (raw/system_prompt/memory/skills/claude_md)",
    )
    parser.add_argument(
        "--injection-sweep",
        action="store_true",
        help="Run all injection modes for context delivery analysis",
    )
    parser.add_argument(
        "--injections",
        type=str,
        help="Specific injection modes for sweep, e.g., raw,memory,skills",
    )
    parser.add_argument(
        "--deep-n",
        type=int,
        default=1,
        help="Number of Deep sessions (stochastic sampling, default: 1)",
    )
    parser.add_argument(
        "--fresh-n",
        type=int,
        default=1,
        help="Number of Fresh sessions (stochastic sampling, default: 1)",
    )
    parser.add_argument(
        "--ploidy",
        type=int,
        default=None,
        help="Set session count per role (1=haploid, 2=diploid, 3=triploid). Overrides --deep-n and --fresh-n.",
    )
    parser.add_argument(
        "--ploidy-sweep",
        action="store_true",
        help="Run ploidy levels 1n through 4n for stochastic sampling analysis",
    )
    parser.add_argument(
        "--ploidy-levels",
        type=str,
        help="Specific ploidy levels for sweep, e.g., 1,3,4",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="claude",
        choices=list(_BACKENDS.keys()),
        help="LLM backend: claude (CLI, free with subscription) or openai (API, supports OpenRouter/Ollama via OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--context-pct",
        type=int,
        default=100,
        help="Percentage of context given to Deep sessions (0-100, default: 100)",
    )
    parser.add_argument(
        "--context-pct-sweep",
        action="store_true",
        help="Run context percentage sweep for dose-response analysis",
    )
    parser.add_argument(
        "--context-pcts",
        type=str,
        help="Specific context percentages for sweep, e.g., 0,25,50,75,100",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume into an existing results directory (skip completed task-method pairs)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for API calls (default: 0.0 for reproducibility)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max output tokens (default: 8192)",
    )
    args = parser.parse_args()
    BACKEND = args.backend
    TEMPERATURE = args.temperature
    MAX_TOKENS = args.max_tokens
    MODEL = (
        args.model if args.model != "claude-opus-4-7" else BACKEND_DEFAULTS.get(BACKEND, args.model)
    )
    JUDGE_MODEL = MODEL  # Use same model for judging by default
    EFFORT = args.effort
    LANGUAGE = args.lang
    INJECTION_MODE = args.injection
    CONTEXT_PCT = args.context_pct
    if args.ploidy is not None:
        DEEP_N = args.ploidy
        FRESH_N = args.ploidy
    else:
        DEEP_N = args.deep_n
        FRESH_N = args.fresh_n

    if args.long:
        from tasks_longcontext import LONG_CONTEXT_TASKS

        TASKS.clear()
        TASKS.extend(LONG_CONTEXT_TASKS)

    if hasattr(args, "extended") and args.extended:
        from tasks_extended import EXTENDED_TASKS

        TASKS.clear()
        TASKS.extend(EXTENDED_TASKS)

    if hasattr(args, "all_long") and args.all_long:
        from tasks_extended import EXTENDED_TASKS
        from tasks_longcontext import LONG_CONTEXT_TASKS

        TASKS.clear()
        TASKS.extend(LONG_CONTEXT_TASKS)
        TASKS.extend(EXTENDED_TASKS)

    if hasattr(args, "gradient") and args.gradient:
        from tasks_gradient import GRADIENT_TASKS

        TASKS.clear()
        TASKS.extend(GRADIENT_TASKS)

    task_ids = [int(x) for x in args.tasks.split(",")] if args.tasks else None
    method_ids = args.methods.split(",") if args.methods else None

    if args.context_pct_sweep:
        sweep_pcts = [int(x) for x in args.context_pcts.split(",")] if args.context_pcts else None
        run_context_pct_sweep(task_ids, method_ids, sweep_pcts)
    elif args.effort_sweep:
        sweep_efforts = args.efforts.split(",") if args.efforts else None
        run_effort_sweep(task_ids, method_ids, sweep_efforts)
    elif args.lang_sweep:
        sweep_langs = args.langs.split(",") if args.langs else None
        run_language_sweep(task_ids, method_ids, sweep_langs)
    elif args.injection_sweep:
        sweep_modes = args.injections.split(",") if args.injections else None
        run_injection_sweep(task_ids, method_ids, sweep_modes)
    elif args.ploidy_sweep:
        sweep_levels = (
            [int(x) for x in args.ploidy_levels.split(",")] if args.ploidy_levels else None
        )
        run_ploidy_sweep(task_ids, method_ids, sweep_levels)
    else:
        resume_dir = Path(args.resume) if args.resume else None
        run_experiment(task_ids, method_ids, resume_dir=resume_dir)
