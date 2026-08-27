#!/usr/bin/env python3
"""Per-session re-judging for AD1 ploidy(2n) Opus cells.

Spec-v3 §2.2 primary DV `fresh_exclusive_count_raw` needs Deep/Fresh
session-level recall, not the converged-output recall the cell JSONs
currently carry. Re-judges each Deep/Fresh Position section against GT
using gemini-2.5-pro CLI (free tier, valid pre-2026-06-18 sunset).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tasks_adversarial import ADVERSARIAL_TASKS

GT_BY_TASK = {t.id: t.ground_truth for t in ADVERSARIAL_TASKS}
SECTION_RE = re.compile(r"^=== ([^=\n]+?) ===\s*\n", re.MULTILINE)


def split_sections(output):
    parts = SECTION_RE.split(output)
    result = {}
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            result[parts[i].strip()] = parts[i + 1]
    return result


def position_role(label):
    m = re.match(r"(Deep|Fresh) Position (\d+)/", label)
    if not m:
        return (None, None)
    return (m.group(1).lower(), int(m.group(2)) - 1)


JUDGE_PROMPT = """You are evaluating whether a reviewer found specific issues.

GROUND TRUTH issues (numbered 1-N):
{gt_block}

REVIEWER OUTPUT:
{output}

For EACH ground truth issue, decide:
- FOUND: the reviewer clearly identified this issue
- PARTIAL: the reviewer mentioned a related concern but not the exact issue
- MISSED: the reviewer did not address this issue

Return ONLY a JSON object like:
{{"verdicts": [{{"gt": 1, "verdict": "FOUND"}}, {{"gt": 2, "verdict": "MISSED"}}]}}

No explanation, no markdown — just the JSON object.
"""


async def judge_one(sem, session_text, gt_items, task_id, role, idx):
    async with sem:
        gt_block = "\n".join(f"{i + 1}. {g}" for i, g in enumerate(gt_items))
        prompt = JUDGE_PROMPT.format(gt_block=gt_block, output=session_text[:30000])
        proc = await asyncio.create_subprocess_exec(
            "gemini", "-m", "gemini-2.5-pro", "-p", prompt,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return {"task_id": task_id, "role": role, "idx": idx,
                    "error": f"gemini rc={proc.returncode}"}
        out = stdout.decode(errors="replace")
        m = re.search(r"\{[^{}]*\"verdicts\".*?\}", out, re.DOTALL)
        if not m:
            return {"task_id": task_id, "role": role, "idx": idx, "error": "no_json"}
        try:
            obj = json.loads(m.group(0))
            per_gt = {}
            for v in obj.get("verdicts", []):
                gt_num = v.get("gt")
                verdict = v.get("verdict", "MISSED").upper()
                if gt_num is not None:
                    per_gt[gt_num] = verdict
            return {"task_id": task_id, "role": role, "idx": idx, "per_gt": per_gt}
        except json.JSONDecodeError as e:
            return {"task_id": task_id, "role": role, "idx": idx, "error": f"parse: {e}"}


async def process_cell(sem, cell_path, out_fp, progress):
    try:
        cell = json.loads(cell_path.read_text())
    except Exception:
        progress["error"] += 1
        return
    tid = cell.get("task_id") or cell.get("task_name", "")
    gt = GT_BY_TASK.get(tid, [])
    if not gt:
        progress["skip"] += 1
        return
    output = cell.get("output", "")
    if not output:
        progress["skip"] += 1
        return

    sections = split_sections(output)
    sessions = []
    for label, text in sections.items():
        role, idx = position_role(label)
        if role is not None and idx is not None:
            sessions.append((role, idx, text))
    if not sessions:
        progress["skip"] += 1
        return

    results = await asyncio.gather(
        *(judge_one(sem, t, gt, tid, r, i) for r, i, t in sessions)
    )

    n_gt = len(gt)
    deep_found = [False] * n_gt
    fresh_found = [False] * n_gt
    for r in results:
        if not r or "per_gt" not in r:
            continue
        role = r["role"]
        for gt_num, verdict in r["per_gt"].items():
            if 1 <= gt_num <= n_gt and verdict in ("FOUND", "PARTIAL"):
                if role == "deep":
                    deep_found[gt_num - 1] = True
                elif role == "fresh":
                    fresh_found[gt_num - 1] = True

    deep_excl = sum(1 for i in range(n_gt) if deep_found[i] and not fresh_found[i])
    fresh_excl = sum(1 for i in range(n_gt) if fresh_found[i] and not deep_found[i])
    both = sum(1 for i in range(n_gt) if deep_found[i] and fresh_found[i])
    neither = sum(1 for i in range(n_gt) if not deep_found[i] and not fresh_found[i])

    record = {
        "cell_path": str(cell_path), "task_id": tid, "n_gt": n_gt,
        "deep_excl_raw": deep_excl, "fresh_excl_raw": fresh_excl,
        "both": both, "neither": neither, "session_results": results,
    }
    out_fp.write(json.dumps(record) + "\n")
    out_fp.flush()
    progress["done"] += 1
    if progress["done"] % 5 == 0:
        print(f"  [{progress['done']}/{progress['total']}] {Path(cell_path).name[:50]} "
              f"D={deep_excl} F={fresh_excl} both={both} miss={neither}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("experiments/results"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    candidates = []
    for d in sorted(args.results_root.iterdir()):
        if not d.is_dir(): continue
        for fp in d.glob("*.json"):
            if "summary" in fp.name or "secondary" in fp.name: continue
            try: j = json.loads(fp.read_text())
            except Exception: continue
            if "error" in j: continue
            tid = j.get("task_id") or j.get("task_name", "")
            if not tid.startswith("adv_"): continue
            if j.get("model") != "claude-opus-4-7": continue
            if j.get("method") != "ploidy": continue
            if (j.get("deep_n"), j.get("fresh_n")) != (2, 2): continue
            if not j.get("output"): continue
            candidates.append(fp)

    print(f"Found {len(candidates)} candidate AD1 ploidy(2n) Opus cells")
    if args.limit:
        candidates = candidates[: args.limit]
        print(f"Limited to {len(candidates)}")

    sem = asyncio.Semaphore(args.parallel)
    progress = {"done": 0, "error": 0, "skip": 0, "total": len(candidates)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out_fp:
        await asyncio.gather(*(process_cell(sem, p, out_fp, progress) for p in candidates))
    print(f"\nFinal: {progress}")


if __name__ == "__main__":
    asyncio.run(main())
