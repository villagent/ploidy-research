"""Terminal CLI for Ploidy — one-shot debate from the shell.

Targets the running server's ``POST /v1/debate/stream`` endpoint so
nothing new lives here that is not also available to a browser or a
Discord bot. The CLI exists because scripting / cron / "just ask
from zsh" is a use case Claude Code cannot cover.

Usage::

    ploidy-ask --context-file architecture.md "should we rewrite in Rust?"
    ploidy-ask --context-file a.md --context-file b.md --deep-n 2 "..."
    PLOIDY_URL=https://ploidy.example.com ploidy-ask --context-file context.md "..."

Exits 0 on ``completed``, non-zero on ``error`` or a transport
failure. Streams progress ticks to stderr and prints the final
``rendered_markdown`` to stdout so you can pipe it into ``bat``,
``glow``, or a file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


_EMOJI = {
    "phase_started": "📍",
    "positions_generated": "🧠",
    "challenges_generated": "⚔️",
    "completed": "✅",
    "result": "🎯",
    "error": "❌",
}


def _describe(event_type: str, data: dict[str, Any]) -> str:
    """Render one progress event as a single human-readable line."""
    if event_type == "phase_started":
        return f"phase: {data.get('phase')}"
    if event_type == "positions_generated":
        return f"{data.get('side')} positions × {data.get('count')}"
    if event_type == "challenges_generated":
        return (
            f"challenges exchanged "
            f"(deep={data.get('deep_action')}, fresh={data.get('fresh_action')})"
        )
    if event_type == "completed":
        conf = data.get("confidence")
        points = data.get("points")
        conf_str = f"{int(conf * 100)}%" if isinstance(conf, (int, float)) else "?"
        return f"done — confidence {conf_str}, {points} points"
    if event_type == "error":
        return f"{data.get('kind', 'Error')}: {data.get('message')}"
    return event_type


def _iter_sse_frames(stream: Iterator[bytes]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse a chunked byte stream into ``(event_type, data)`` tuples.

    SSE frames are terminated by a blank line. We accumulate chunks in a
    string buffer and yield each frame as it completes.
    """
    buffer = ""
    for chunk in stream:
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            event_type, payload = _parse_frame(frame)
            if event_type is None:
                continue
            yield event_type, payload


def _parse_frame(frame: str) -> tuple[str | None, dict[str, Any]]:
    event_type: str | None = None
    data_line: str | None = None
    for line in frame.splitlines():
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_line = line[len("data:") :].strip()
    if not event_type or data_line is None:
        return None, {}
    try:
        parsed = json.loads(data_line)
    except json.JSONDecodeError:
        return None, {}
    # Each frame's JSON is ``{"type": X, "data": {...}}``. Return the
    # nested ``data`` dict so callers do not have to unwrap.
    return event_type, parsed.get("data", {}) if isinstance(parsed, dict) else {}


def _stream_debate(url: str, body: dict[str, Any], token: str | None) -> tuple[str | None, int]:
    """Drive one debate run against the SSE endpoint.

    Returns ``(rendered_markdown, exit_code)``. Exit code is 0 on
    ``result`` (with markdown if present), 2 on a server-side ``error``
    frame, and 3 on a transport failure.
    """
    if httpx is None:
        print(
            "ploidy-ask requires httpx. Install with: pip install ploidy[cli]",
            file=sys.stderr,
        )
        return None, 3

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.stream(
            "POST",
            url,
            json=body,
            headers=headers,
            timeout=httpx.Timeout(None, connect=10.0),
        ) as resp:
            if resp.status_code != 200:
                print(
                    f"❌ HTTP {resp.status_code}: {resp.read().decode('utf-8', 'replace')}",
                    file=sys.stderr,
                )
                return None, 3
            rendered: str | None = None
            exit_code = 0
            for event_type, data in _iter_sse_frames(resp.iter_bytes()):
                emoji = _EMOJI.get(event_type, "·")
                print(f"{emoji} {_describe(event_type, data)}", file=sys.stderr)
                if event_type == "result":
                    rendered = data.get("rendered_markdown") or data.get("synthesis")
                elif event_type == "error":
                    exit_code = 2
            return rendered, exit_code
    except httpx.HTTPError as exc:
        print(f"❌ transport error: {exc}", file=sys.stderr)
        return None, 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ploidy-ask",
        description="Run a Ploidy debate against a running server and stream progress.",
    )
    parser.add_argument("prompt", help="The decision question to debate.")
    parser.add_argument(
        "--url",
        default=os.environ.get("PLOIDY_URL", "http://127.0.0.1:8765"),
        help="Base URL of the Ploidy server (default: $PLOIDY_URL or localhost).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("PLOIDY_API_TOKEN"),
        help="Bearer token when the server configures PLOIDY_TOKENS.",
    )
    parser.add_argument("--deep-n", type=int, default=1)
    parser.add_argument("--fresh-n", type=int, default=1)
    parser.add_argument(
        "--effort",
        choices=("low", "medium", "high", "max"),
        default="high",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Output language code (en/ko/ja/zh).",
    )
    parser.add_argument(
        "--context-file",
        action="append",
        required=True,
        metavar="PATH",
        help=(
            "Text file visible only to Deep. Repeat for multiple files; "
            "use '-' to read one context document from stdin."
        ),
    )
    args = parser.parse_args(argv)

    context_documents: list[str] = []
    context_sources: list[str] = []
    for filename in args.context_file:
        if filename == "-":
            content = sys.stdin.read()
            source = "stdin"
        else:
            path = Path(filename)
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                parser.error(f"cannot read --context-file {filename!r} as UTF-8 text: {exc}")
            # Source labels are persisted and may be returned by a remote
            # service; keep local usernames/directories out of that surface.
            source = f"file:{path.name}"
        if not content.strip():
            parser.error(f"--context-file {filename!r} is empty")
        context_documents.append(content)
        context_sources.append(source)

    endpoint = args.url.rstrip("/") + "/v1/debate/stream"
    body = {
        "prompt": args.prompt,
        "context_documents": context_documents,
        "context_sources": context_sources,
        "deep_n": args.deep_n,
        "fresh_n": args.fresh_n,
        "effort": args.effort,
        "language": args.language,
    }
    rendered, code = _stream_debate(endpoint, body, args.token)
    if rendered:
        print(rendered)
    return code


if __name__ == "__main__":
    sys.exit(main())
