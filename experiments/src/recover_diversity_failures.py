"""Recover failed cells from the March 2026 Codex diversity runs.

The script never mutates the original result directories. It reads the two
legacy ``summary.json`` files, classifies their failed rows, and writes a new
recovery run directory under ``experiments/results``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import run_diversity_experiment as diversity

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "experiments" / "results"

DEFAULT_SUMMARIES = [
    RESULTS_ROOT / "20260328_022104_diversity_claude-vs-codex" / "summary.json",
    RESULTS_ROOT / "20260328_022112_diversity_gemini-vs-codex" / "summary.json",
]


def classify_error(error: str) -> dict[str, str | bool]:
    """Classify a legacy failure and whether a one-shot recovery is sensible."""
    low = error.lower()
    if "codex cli error" in low and "usage limit" in low:
        return {
            "component": "codex",
            "reason": "codex_usage_limit",
            "recoverability": "one_shot_current_quota_dependent",
            "retry_if_repeats": False,
        }
    if "codex cli error" in low:
        return {
            "component": "codex",
            "reason": "codex_cli_error",
            "recoverability": "one_shot_with_current_cli",
            "retry_if_repeats": False,
        }
    if "claude cli error" in low:
        return {
            "component": "claude",
            "reason": "claude_cli_error",
            "recoverability": "provider_dependent",
            "retry_if_repeats": False,
        }
    if "gemini" in low and "timed out" in low:
        return {
            "component": "gemini",
            "reason": "gemini_timeout",
            "recoverability": "provider_dependent_timeout_risk",
            "retry_if_repeats": False,
        }
    if "timed out" in low:
        return {
            "component": "unknown",
            "reason": "timeout",
            "recoverability": "provider_dependent_timeout_risk",
            "retry_if_repeats": False,
        }
    return {
        "component": "unknown",
        "reason": "other",
        "recoverability": "inspect_manually",
        "retry_if_repeats": False,
    }


def parse_primary_secondary(summary_path: Path) -> tuple[str, str]:
    """Parse ``diversity_primary-vs-secondary`` from a legacy result path."""
    suffix = summary_path.parent.name.split("_diversity_", 1)[1]
    primary, secondary = suffix.split("-vs-", 1)
    return primary, secondary


def load_failed_cells(summary_paths: Iterable[Path]) -> list[dict]:
    """Load failed rows from the supplied legacy diversity summaries."""
    failed: list[dict] = []
    for summary_path in summary_paths:
        primary, secondary = parse_primary_secondary(summary_path)
        rows = json.loads(summary_path.read_text())
        for source_index, row in enumerate(rows):
            if "error" not in row:
                continue
            failed.append(
                {
                    "source_summary": str(summary_path),
                    "source_run": summary_path.parent.name,
                    "source_index": source_index,
                    "primary": primary,
                    "secondary": secondary,
                    "task_id": row["task_id"],
                    "condition": row["condition"],
                    "repeat": row.get("repeat", 1),
                    "legacy_error": row["error"],
                    **classify_error(row["error"]),
                }
            )
    return failed


def condition_name(condition: str, primary: str, secondary: str) -> str:
    """Return the same display name used by the legacy diversity runner."""
    names = {
        "D_single_primary": f"Single ({primary})",
        "D_single_secondary": f"Single ({secondary})",
        "A_cross_model_symmetric": (
            f"Cross-Model Symmetric ({primary} vs {secondary}, same ctx)"
        ),
        "B_same_model_asymmetric_primary": (
            f"Same-Model Asymmetric ({primary} deep vs {primary} fresh)"
        ),
        "B_same_model_asymmetric_secondary": (
            f"Same-Model Asymmetric ({secondary} deep vs {secondary} fresh)"
        ),
        "C_cross_model_asymmetric": (
            f"Cross-Model Asymmetric ({primary} deep vs {secondary} fresh)"
        ),
        "C_cross_model_asymmetric_reversed": (
            f"Cross-Model Asymmetric ({secondary} deep vs {primary} fresh)"
        ),
    }
    return names[condition]


def build_condition_fn(cell: dict) -> Callable[[diversity.Task], str]:
    """Build the callable for a failed diversity cell."""
    primary = cell["primary"]
    secondary = cell["secondary"]
    condition = cell["condition"]
    fns: dict[str, Callable[[diversity.Task], str]] = {
        "D_single_primary": lambda task: diversity.method_single(task, primary),
        "D_single_secondary": lambda task: diversity.method_single(task, secondary),
        "A_cross_model_symmetric": (
            lambda task: diversity.method_cross_model_symmetric(task, primary, secondary)
        ),
        "B_same_model_asymmetric_primary": (
            lambda task: diversity.method_same_model_asymmetric(task, primary)
        ),
        "B_same_model_asymmetric_secondary": (
            lambda task: diversity.method_same_model_asymmetric(task, secondary)
        ),
        "C_cross_model_asymmetric": (
            lambda task: diversity.method_cross_model_asymmetric(task, primary, secondary)
        ),
        "C_cross_model_asymmetric_reversed": (
            lambda task: diversity.method_cross_model_asymmetric(task, secondary, primary)
        ),
    }
    return fns[condition]


def choose_smoke_cells(failed: list[dict], limit: int) -> list[dict]:
    """Choose low-cost Codex-side failures first for smoke recovery."""
    priority_conditions = ["D_single_secondary", "D_single_primary"]

    def sort_key(cell: dict) -> tuple[int, int, str, int]:
        component_rank = 0 if cell["component"] == "codex" else 1
        try:
            condition_rank = priority_conditions.index(cell["condition"])
        except ValueError:
            condition_rank = 9
        return (component_rank, condition_rank, cell["source_run"], cell["source_index"])

    return sorted(failed, key=sort_key)[:limit]


def load_merge_patch_keys(paths: Iterable[Path]) -> set[tuple[str, int]]:
    """Load recovered ``(source_summary, source_index)`` keys from merge patches."""
    keys: set[tuple[str, int]] = set()
    for path in paths:
        for item in json.loads(path.read_text()):
            keys.add((item["source_summary"], item["source_index"]))
    return keys


def filter_failed_cells(
    failed: list[dict],
    only_component: str | None = None,
    only_reason: str | None = None,
    exclude_keys: set[tuple[str, int]] | None = None,
) -> list[dict]:
    """Filter failed cells by component/reason and previously recovered keys."""
    excluded = exclude_keys or set()
    selected = []
    for cell in failed:
        key = (cell["source_summary"], cell["source_index"])
        if key in excluded:
            continue
        if only_component and cell["component"] != only_component:
            continue
        if only_reason and cell["reason"] != only_reason:
            continue
        selected.append(cell)
    return selected


def current_failure_status(error: str) -> str:
    """Classify a current recovery failure for no-retry handling."""
    low = error.lower()
    if "codex cli error" in low and "usage limit" in low:
        return "current_codex_usage_limit_no_retry"
    if "usage limit" in low:
        return "current_usage_limit_no_retry"
    return "current_error_no_retry"


def score_judgment(task: diversity.Task, judgment: dict) -> dict[str, int | float]:
    """Convert a judge response to the legacy summary metrics."""
    if "scores" not in judgment:
        return {
            "found": 0,
            "partial": 0,
            "missed": 0,
            "total_gt": len(task.ground_truth),
            "bonus_findings": 0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
        }
    found = sum(1 for score in judgment["scores"] if score["verdict"] == "FOUND")
    partial = sum(1 for score in judgment["scores"] if score["verdict"] == "PARTIAL")
    missed = sum(1 for score in judgment["scores"] if score["verdict"] == "MISSED")
    total = len(task.ground_truth)
    bonus = judgment.get("bonus_findings", 0)
    recall = (found + 0.5 * partial) / total
    precision = (found + 0.5 * partial) / max(found + partial + bonus, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    return {
        "found": found,
        "partial": partial,
        "missed": missed,
        "total_gt": total,
        "bonus_findings": bonus,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def run_cell(cell: dict, out_dir: Path) -> dict:
    """Run one failed diversity cell and write its full recovery artifact."""
    tasks_by_id = {task.id: task for task in diversity.TASKS}
    task = tasks_by_id[cell["task_id"]]
    cond_name = condition_name(cell["condition"], cell["primary"], cell["secondary"])
    cond_fn = build_condition_fn(cell)
    artifact_name = (
        f"{cell['source_run']}__idx-{cell['source_index']:02d}"
        f"__{cell['task_id']}__{cell['condition']}.json"
    )
    artifact_path = out_dir / artifact_name

    diversity.reset_token_tracker()
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    try:
        output = cond_fn(task)
        judgment = diversity.judge_result(task, cond_name, output)
        elapsed = time.time() - t0
        metrics = score_judgment(task, judgment)
        tokens = diversity.get_token_usage()
        row = {
            "repeat": cell.get("repeat", 1),
            "task_id": task.id,
            "task_name": task.name,
            "condition": cell["condition"],
            "condition_name": cond_name,
            "primary_backend": cell["primary"],
            "secondary_backend": cell["secondary"],
            **metrics,
            "elapsed_seconds": elapsed,
            "token_usage": {
                "prompt_tokens": tokens["prompt_tokens"],
                "completion_tokens": tokens["completion_tokens"],
                "total_tokens": tokens["total_tokens"],
                "llm_calls": tokens["calls"],
                "estimated": tokens["estimated"],
            },
        }
        record = {
            "status": "ok",
            "started_at": started,
            "elapsed_seconds": elapsed,
            "source": cell,
            "merge_row": row,
            "task": asdict(task),
            "output": output,
            "judgment": judgment,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        record = {
            "status": current_failure_status(str(exc)),
            "started_at": started,
            "elapsed_seconds": elapsed,
            "source": cell,
            "error": str(exc),
        }

    artifact_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return record


def write_classification(
    out_dir: Path,
    failed: list[dict],
    eligible: list[dict],
    selected: list[dict],
    excluded_keys: set[tuple[str, int]],
) -> None:
    """Write failure inventory and selected recovery cells."""
    counts: dict[str, int] = {}
    for cell in failed:
        key = f"{cell['component']}:{cell['reason']}"
        counts[key] = counts.get(key, 0) + 1
    (out_dir / "failed_cells.json").write_text(
        json.dumps(failed, indent=2, ensure_ascii=False)
    )
    (out_dir / "classification_summary.json").write_text(
        json.dumps(
            {
                "total_failed_cells": len(failed),
                "total_eligible_cells": len(eligible),
                "excluded_recovered_cells": len(excluded_keys),
                "counts": counts,
                "selected_cells": selected,
                "selected_smoke_cells": selected,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    """Run smoke recovery for selected failed diversity cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        default=None,
        help="Legacy summary.json to recover. Defaults to the two Codex diversity runs.",
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=2,
        help="Number of failed cells to attempt. Use 2 for smoke.",
    )
    parser.add_argument(
        "--only-component",
        type=str,
        choices=["codex", "claude", "gemini", "unknown"],
        default=None,
        help="Attempt only failures attributed to this component.",
    )
    parser.add_argument(
        "--only-reason",
        type=str,
        default=None,
        help="Attempt only failures with this classified reason.",
    )
    parser.add_argument(
        "--exclude-merge-patch",
        action="append",
        type=Path,
        default=None,
        help="Do not re-attempt cells already present in this merge_patch.json.",
    )
    parser.add_argument(
        "--stop-on-codex-limit",
        action="store_true",
        help="Stop after recording a current Codex usage-limit failure.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="smoke",
        help="Label used in the default recovery run directory name.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Recovery output directory. Defaults to a new experiments/results run dir.",
    )
    args = parser.parse_args()

    summaries = args.summary or DEFAULT_SUMMARIES
    failed = load_failed_cells(summaries)
    excluded_keys = load_merge_patch_keys(args.exclude_merge_patch or [])
    eligible = filter_failed_cells(
        failed,
        only_component=args.only_component,
        only_reason=args.only_reason,
        exclude_keys=excluded_keys,
    )
    selected = choose_smoke_cells(eligible, args.smoke)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        RESULTS_ROOT / f"{timestamp}_diversity_recovery_{args.label}-{len(selected)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    write_classification(out_dir, failed, eligible, selected, excluded_keys)
    records = []
    stopped_early = False
    for index, cell in enumerate(selected, 1):
        print(
            f"[{index}/{len(selected)}] {cell['source_run']} "
            f"idx={cell['source_index']} {cell['task_id']}::{cell['condition']} "
            f"legacy={cell['reason']}",
            flush=True,
        )
        record = run_cell(cell, out_dir)
        records.append(record)
        print(f"  -> {record['status']} ({record['elapsed_seconds']:.1f}s)", flush=True)
        if args.stop_on_codex_limit and record["status"] == "current_codex_usage_limit_no_retry":
            stopped_early = True
            print("Stopping: current Codex usage limit recorded.", flush=True)
            break

    merge_patch = [
        {
            "source_summary": record["source"]["source_summary"],
            "source_index": record["source"]["source_index"],
            "replacement_row": record["merge_row"],
        }
        for record in records
        if record["status"] == "ok"
    ]
    (out_dir / "summary.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    (out_dir / "merge_patch.json").write_text(
        json.dumps(merge_patch, indent=2, ensure_ascii=False)
    )
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "attempted": len(records),
                "succeeded": sum(1 for record in records if record["status"] == "ok"),
                "failed": sum(1 for record in records if record["status"] != "ok"),
                "stopped_early": stopped_early,
                "merge_patch_rows": len(merge_patch),
                "only_component": args.only_component,
                "only_reason": args.only_reason,
                "excluded_recovered_cells": len(excluded_keys),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Recovery output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
