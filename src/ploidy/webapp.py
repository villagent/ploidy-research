# ruff: noqa: E501
"""Single-page web UI for the live debate stream.

Served from the same FastMCP server so operators only need one
deployment. Pointing a browser at the root URL gives a text input,
an in-memory-only bearer-token field, and a progress panel driven by
the existing ``POST /v1/debate/stream`` SSE endpoint.

Intentionally zero-build: vanilla HTML + ``fetch`` stream parsing +
The output block treats the server's ``rendered_markdown`` as text rather
than HTML, because model output must never reach an ``innerHTML`` sink.
"""

from __future__ import annotations

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ploidy — live debate</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
         system-ui, sans-serif; background: var(--bg); color: var(--text);
         line-height: 1.55; }
  .wrap { max-width: 860px; margin: 0 auto; padding: 2rem 1rem 4rem; }
  header { display: flex; align-items: baseline; gap: .6rem; margin-bottom: 1.2rem; }
  header h1 { font-size: 1.4rem; }
  header h1 span { color: var(--accent); }
  header small { color: var(--muted); }
  form { background: var(--surface); border: 1px solid var(--border);
         border-radius: 8px; padding: 1rem; }
  label { display: block; color: var(--muted); font-size: .85rem;
          margin-top: .6rem; margin-bottom: .25rem; }
  label:first-child { margin-top: 0; }
  textarea, input[type=password] {
    width: 100%; background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px; padding: .6rem;
    font-family: inherit; font-size: .95rem;
  }
  textarea { min-height: 110px; resize: vertical; }
  .opts { display: flex; gap: .6rem; margin-top: .6rem; align-items: flex-end; }
  .opts > div { flex: 1; }
  .opts label { margin-top: 0; }
  select {
    width: 100%; background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px; padding: .5rem;
    font-family: inherit; font-size: .95rem;
  }
  button {
    margin-top: 1rem; background: var(--accent); color: #0b1118;
    border: 0; border-radius: 6px; padding: .6rem 1.2rem;
    font-weight: 600; cursor: pointer; font-size: .95rem;
  }
  button:disabled { opacity: .5; cursor: wait; }
  .panel { background: var(--surface); border: 1px solid var(--border);
           border-radius: 8px; padding: 1rem; margin-top: 1rem; }
  .panel h2 { font-size: .95rem; color: var(--muted);
              text-transform: uppercase; letter-spacing: .05em;
              margin-bottom: .6rem; }
  #progress .tick { padding: .3rem 0; color: var(--muted); font-size: .9rem;
                    font-family: ui-monospace, Menlo, monospace; }
  #progress .tick.done { color: var(--green); }
  #progress .tick.err { color: var(--red); }
  #result details { margin: .6rem 0; padding: .5rem .8rem;
                    background: var(--bg); border: 1px solid var(--border);
                    border-radius: 6px; }
  #result details summary { cursor: pointer; color: var(--muted); }
  #result details[open] summary { color: var(--text); margin-bottom: .4rem; }
  #result h2 { margin: .8rem 0 .4rem; font-size: 1.1rem; }
  #result h3, #result h4 { margin: .6rem 0 .3rem; }
  #result ul { margin-left: 1.4rem; }
  #result code { background: var(--bg); padding: 1px 5px; border-radius: 3px;
                 font-size: .9em; }
  #result pre { background: var(--bg); border: 1px solid var(--border);
                padding: .6rem; border-radius: 6px; overflow-x: auto; }
  #result hr { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }
  .hidden { display: none; }
  footer { text-align: center; padding: 1.6rem; color: var(--muted);
           font-size: .8rem; }
  footer a { color: var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1><span>Ploidy</span> live debate</h1>
    <small>Deep × Fresh · streaming</small>
  </header>

  <form id="debate-form">
    <label for="prompt">Decision question</label>
    <textarea id="prompt" name="prompt"
      placeholder="Should we rewrite the ingestion pipeline in Rust?"
      required></textarea>

    <label for="context">Deep-only project context</label>
    <textarea id="context" name="context"
      placeholder="Paste architecture notes, code, incidents, or decisions visible only to Deep."
      required></textarea>

    <div class="opts">
      <div>
        <label for="deep_n">Deep n</label>
        <select id="deep_n" name="deep_n">
          <option value="1" selected>1</option>
          <option value="2">2</option>
          <option value="3">3</option>
        </select>
      </div>
      <div>
        <label for="fresh_n">Fresh n</label>
        <select id="fresh_n" name="fresh_n">
          <option value="1" selected>1</option>
          <option value="2">2</option>
          <option value="3">3</option>
        </select>
      </div>
      <div>
        <label for="effort">Effort</label>
        <select id="effort" name="effort">
          <option value="low">low</option>
          <option value="medium">medium</option>
          <option value="high" selected>high</option>
          <option value="max">max</option>
        </select>
      </div>
    </div>

    <label for="token">Bearer token <small>(optional, never stored)</small></label>
    <input id="token" name="token" type="password" autocomplete="off"
           placeholder="Only needed if the server configures PLOIDY_TOKENS">

    <button id="submit" type="submit">Run debate</button>
  </form>

  <section id="progress-panel" class="panel hidden">
    <h2>Progress</h2>
    <div id="progress"></div>
  </section>

  <section id="result-panel" class="panel hidden">
    <h2>Result</h2>
    <div id="result"></div>
  </section>
</div>

<footer>
  Same endpoint as the MCP <code>debate(mode='auto')</code> tool —
  see <a href="/healthz">/healthz</a>, <a href="/metrics">/metrics</a>
  for ops.
</footer>

<script>
(() => {
  const form = document.getElementById('debate-form');
  const progressPanel = document.getElementById('progress-panel');
  const resultPanel = document.getElementById('result-panel');
  const progress = document.getElementById('progress');
  const result = document.getElementById('result');
  const submit = document.getElementById('submit');
  const tokenInput = document.getElementById('token');

  const EMOJI = {
    phase_started: '📍',
    positions_generated: '🧠',
    challenges_generated: '⚔️',
    completed: '✅',
    result: '🎯',
    error: '❌',
  };

  function tick(type, msg, cls = '') {
    const line = document.createElement('div');
    line.className = 'tick ' + cls;
    line.textContent = `${EMOJI[type] || '·'} ${msg}`;
    progress.appendChild(line);
    progress.scrollTop = progress.scrollHeight;
  }

  function describe(type, data) {
    if (type === 'phase_started') return `Phase: ${data.phase}`;
    if (type === 'positions_generated')
      return `${data.side} positions ready (${data.count})`;
    if (type === 'challenges_generated')
      return `Challenges exchanged — deep=${data.deep_action}, fresh=${data.fresh_action}`;
    if (type === 'completed')
      return `Debate complete (confidence ${(data.confidence * 100).toFixed(0)}%, ${data.points} points)`;
    if (type === 'result') return 'Rendered output received';
    if (type === 'error') return `Error: ${data.message || data.kind}`;
    return type;
  }

  async function run(body, token) {
    progressPanel.classList.remove('hidden');
    resultPanel.classList.add('hidden');
    progress.replaceChildren();
    result.replaceChildren();

    const headers = { 'Content-Type': 'application/json' };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const resp = await fetch('/v1/debate/stream', {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      tick('error', `HTTP ${resp.status}: ${await resp.text()}`, 'err');
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by a blank line.
      const frames = buffer.split('\n\n');
      buffer = frames.pop() || '';
      for (const frame of frames) {
        if (!frame.trim()) continue;
        const eventLine = frame.match(/^event:\s*(\S+)/m);
        const dataLine = frame.match(/^data:\s*(.+)$/m);
        if (!eventLine || !dataLine) continue;
        const type = eventLine[1];
        let payload;
        try {
          payload = JSON.parse(dataLine[1]);
        } catch {
          continue;
        }
        const data = payload.data || {};
        const isError = type === 'error';
        const isEnd = type === 'result';
        tick(type, describe(type, data), isError ? 'err' : (isEnd ? 'done' : ''));

        if (isEnd) {
          const md = data.rendered_markdown || '_(no rendered_markdown field — server is pre-v0.4.1)_';
          const pre = document.createElement('pre');
          pre.textContent = md;
          result.replaceChildren(pre);
          resultPanel.classList.remove('hidden');
        }
      }
    }
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = form.prompt.value.trim();
    const context = form.context.value.trim();
    if (!prompt || !context) return;

    const token = tokenInput.value.trim();

    submit.disabled = true;
    submit.textContent = 'Running…';
    try {
      await run(
        {
          prompt,
          context_documents: [context],
          context_sources: ['web:deep-context'],
          deep_n: Number(form.deep_n.value),
          fresh_n: Number(form.fresh_n.value),
          effort: form.effort.value,
        },
        token,
      );
    } catch (err) {
      tick('error', err.message || String(err), 'err');
    } finally {
      submit.disabled = false;
      submit.textContent = 'Run debate';
    }
  });
})();
</script>
</body>
</html>
"""


def index_html() -> str:
    """Return the single-page debate UI as a static HTML string."""
    return _HTML
