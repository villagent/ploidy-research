"""Run Tier 3 Codex repeats for the Opus 4.7 matched subset.

This runner extends the completed Tier 1 matched subset without changing its
meaning or rewriting its artifacts:

* repeat 1 is read from ``20260617_081145_codex_matched_subset`` only;
* repeats 2 and 3 are new Codex subject cells;
* Claude reference cells are never regenerated;
* Gemini CLI and Antigravity are never invoked;
* scoring remains the fixed Claude Opus 4.7 judge for same-scoring parity.

The resulting artifact is therefore a Codex subject Tier 3 with fixed Claude
judge scoring, not a pure no-Claude-call run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import run_codex_matched_subset as base
from run_codex_matched_subset import (
    CODEX_MODEL,
    DEEP_N,
    EFFORT,
    FRESH_N,
    INJECTION_MODE,
    JUDGE_BACKEND,
    JUDGE_MODEL,
    LANGUAGE,
    METHODS,
    REFERENCE_MODEL,
    RESULTS_ROOT,
    CliFailure,
    MatchedCodexRunner,
    codex_version,
    load_json,
    metrics_from_cell,
    now_iso,
    sha256_text,
    task_registry,
    write_json,
)


BASELINE_TIER1_DIR = RESULTS_ROOT / "20260617_081145_codex_matched_subset"
CLEAN_PARITY_DIR = RESULTS_ROOT / "20260617_060030_clean_cross_model_parity"
DEFAULT_REPEATS = (2, 3)
BASELINE_REPEAT = 1
ALL_REPEATS = (1, 2, 3)
INSTABILITY_F1_RANGE_THRESHOLD = 0.25
INSTABILITY_RECALL_RANGE_THRESHOLD = 0.25


def mean(values: list[float]) -> float | None:
    """Return a rounded mean, or ``None`` for empty input."""
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def stddev(values: list[float]) -> float | None:
    """Return rounded sample standard deviation, or zero for a singleton."""
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return round(statistics.stdev(values), 4)


def median(values: list[float]) -> float | None:
    """Return a rounded median, or ``None`` for empty input."""
    if not values:
        return None
    return round(float(statistics.median(values)), 4)


class CodexMatchedTier3Runner(MatchedCodexRunner):
    """Run repeat 2/3 Codex subject cells over the Tier 1 matched task set."""

    def __init__(
        self,
        run_dir: Path,
        repeats: tuple[int, ...] = DEFAULT_REPEATS,
        stop_on_capacity: bool = True,
    ) -> None:
        self.run_dir = run_dir
        self.raw_dir = run_dir / "raw_calls"
        self.tmp_root = run_dir / "tmp"
        self.repeats = repeats
        self.stop_on_capacity = stop_on_capacity
        self.cli_versions = {"codex": codex_version()}
        self.call_lock = threading.Lock()
        self.current_cell: dict[str, Any] | None = None
        self.current_call_index = 0
        self.cells: list[dict[str, Any]] = []
        self.stop_reason: dict[str, Any] | None = None
        self.registry = task_registry()

        self.baseline_tasks = self.load_baseline_tasks()
        self.baseline_summary = self.load_baseline_summary()
        self.baseline_rows = self.load_baseline_rows()
        self.task_items = [self.registry[row["task_id"]] for row in self.baseline_tasks]
        self.references = {
            row["task_id"]: row["claude_reference"] for row in self.baseline_tasks
        }
        self.prior_codex: dict[str, dict[str, dict[str, Any]]] = {}
        self.todo_cells = [
            (repeat, item, method)
            for repeat in self.repeats
            for item in self.task_items
            for method in METHODS
        ]

    def load_baseline_tasks(self) -> list[dict[str, Any]]:
        """Load the exact Tier 1 matched task manifest."""
        path = BASELINE_TIER1_DIR / "MATCHED_TASKS.json"
        data = load_json(path)
        if not isinstance(data, list) or len(data) != 33:
            raise RuntimeError(f"Expected 33 matched tasks in {path}")
        missing = [row.get("task_id") for row in data if row.get("task_id") not in self.registry]
        if missing:
            raise RuntimeError(f"Matched task ids not present in local registry: {missing}")
        return data

    def load_baseline_summary(self) -> dict[str, Any]:
        """Load the completed Tier 1 summary."""
        path = BASELINE_TIER1_DIR / "summary.json"
        data = load_json(path)
        if not isinstance(data, dict):
            raise RuntimeError(f"Could not read Tier 1 baseline summary: {path}")
        coverage = data.get("coverage", {})
        if coverage.get("combined_covered_cells") != 99:
            raise RuntimeError("Tier 1 baseline is not 99/99 complete")
        return data

    def load_baseline_rows(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Return repeat-1 Codex rows keyed by task and method."""
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.baseline_summary.get("comparison_rows", []):
            if row.get("codex_status") != "success":
                continue
            key = (row["task_id"], row["method"])
            rows[key] = {
                "repeat_index": BASELINE_REPEAT,
                "task_id": row["task_id"],
                "method": row["method"],
                "codex_path": row["codex_path"],
                "codex_source": row.get("codex_source"),
                "codex_f1": row["codex_f1"],
                "codex_recall": row["codex_recall"],
                "claude_reference_path": row["claude_reference_path"],
                "claude_f1": row["claude_f1"],
                "claude_recall": row["claude_recall"],
                "status": "success",
            }
        expected = len(self.baseline_tasks) * len(METHODS)
        if len(rows) != expected:
            raise RuntimeError(f"Tier 1 baseline rows are incomplete: {len(rows)}/{expected}")
        return rows

    def prepare(self, resume: bool = False) -> None:
        """Create run directories and write the immutable Tier 3 spec."""
        if self.run_dir.exists() and not resume:
            raise FileExistsError(f"Refusing to overwrite existing run dir: {self.run_dir}")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.load_existing_cells()
        self.write_run_spec()
        self.write_summary()

    def load_existing_cells(self) -> None:
        """Load repeat 2/3 cells already produced in this run directory."""
        self.cells = []
        if not self.run_dir.exists():
            return
        for path in sorted(self.run_dir.glob("codex__r*.json")):
            cell = load_json(path)
            if isinstance(cell, dict):
                self.cells.append(cell)

    def write_run_spec(self) -> None:
        """Write Tier 3 run spec and a baseline-aware matched task manifest."""
        matched_tasks = []
        for row in self.baseline_tasks:
            task_id = row["task_id"]
            baseline_repeat_1 = {
                method: self.baseline_rows[(task_id, method)] for method in METHODS
            }
            matched_tasks.append(
                {
                    **row,
                    "tier3_baseline_repeat_1": baseline_repeat_1,
                    "tier3_new_repeats": list(self.repeats),
                    "tier3_new_todo_methods": list(METHODS),
                }
            )
        write_json(self.run_dir / "MATCHED_TASKS.json", matched_tasks)
        spec = {
            "created_at": now_iso(),
            "purpose": "codex_matched_subset_tier3",
            "full_corpus_replication": False,
            "matched_subset_replication": True,
            "tier": 3,
            "caveat": "Codex subject Tier 3 with fixed Claude judge scoring; not pure no-Claude-call run",
            "baseline_repeat_1": {
                "repeat_index": BASELINE_REPEAT,
                "run_dir": str(BASELINE_TIER1_DIR),
                "policy": "read-only reference only; no duplicate Codex subject calls",
                "coverage_cells": len(self.baseline_rows),
            },
            "reference_policy": {
                "reference_model": REFERENCE_MODEL,
                "reference_source": str(BASELINE_TIER1_DIR / "MATCHED_TASKS.json"),
                "reference_filter": {
                    "effort": EFFORT,
                    "language": LANGUAGE,
                    "injection_mode": INJECTION_MODE,
                    "deep_n": DEEP_N,
                    "fresh_n": FRESH_N,
                    "methods": list(METHODS),
                    "exclude_dirs_with_contaminated_marker": True,
                },
                "cell_selection_rule": "reuse the exact Tier 1 matched task/reference manifest",
            },
            "codex_execution_policy": {
                "subject_backend": "codex",
                "subject_model": CODEX_MODEL,
                "provider_arm": "openai_codex_cli",
                "new_repeats_to_attempt": list(self.repeats),
                "new_subject_cells_to_attempt": len(self.todo_cells),
                "forbidden_subject_arms": [
                    "anthropic_claude_cli",
                    "google_gemini_cli_legacy",
                    "google_antigravity",
                ],
                "read_only_reference_dirs": [str(BASELINE_TIER1_DIR), str(CLEAN_PARITY_DIR)],
                "existing_tier1_cells_are_not_rerun": True,
            },
            "parity_policy": {
                "same_input": True,
                "same_role": True,
                "same_scoring": True,
                "same_prompt_text": {
                    "single": True,
                    "ccr": "partial: downstream prompts include the model's prior output",
                    "ploidy": "partial: downstream prompts include model-output-conditioned positions/challenges",
                },
            },
            "scoring": {
                "backend": JUDGE_BACKEND,
                "model": JUDGE_MODEL,
                "allowed_claude_calls": "fixed judge scoring calls only",
                "prompt_source": "experiments/src/run_experiment.py::judge_result prompt text, copied in run_codex_matched_subset.py",
                "metric_formula_source": "experiments/src/run_experiment.py::_run_one_method",
            },
            "matrix": {
                "matched_tasks": len(self.task_items),
                "methods": list(METHODS),
                "repeats": list(ALL_REPEATS),
                "baseline_repeat_1_cells": len(self.baseline_rows),
                "new_repeat_cells_target": len(self.todo_cells),
                "total_tier3_target_cells": len(self.task_items) * len(METHODS) * len(ALL_REPEATS),
            },
            "instability_flag_thresholds": {
                "f1_range_gte": INSTABILITY_F1_RANGE_THRESHOLD,
                "recall_range_gte": INSTABILITY_RECALL_RANGE_THRESHOLD,
            },
            "cli_versions": self.cli_versions,
        }
        write_json(self.run_dir / "RUN_SPEC.json", spec)
        lines = [
            "# Codex Matched Tier 3",
            "",
            "Codex subject Tier 3 with fixed Claude judge scoring; not pure no-Claude-call run.",
            "",
            f"- baseline repeat 1: {len(self.baseline_rows)}/99 cells from `{BASELINE_TIER1_DIR}`",
            f"- new repeats: {', '.join(str(r) for r in self.repeats)}",
            f"- new Codex subject cells to attempt: {len(self.todo_cells)}",
            "- Gemini CLI and Antigravity are not invoked.",
            "- Existing Claude subject/reference cells are not regenerated.",
        ]
        (self.run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def result_path(self, repeat: int, task_id: str, method_id: str) -> Path:
        """Return the per-cell result path for a repeat/task/method."""
        return self.run_dir / f"codex__r{repeat:02d}__{task_id}__{method_id}.json"

    def cell_key(self, cell: dict[str, Any]) -> tuple[int, str, str]:
        """Return a repeat/task/method key for a cell."""
        return (int(cell["repeat_index"]), cell["task_id"], cell["method"])

    def successful_current_keys(self) -> set[tuple[int, str, str]]:
        """Return successful repeat 2/3 keys already present in this run."""
        return {
            self.cell_key(cell)
            for cell in self.cells
            if cell.get("status") == "success"
        }

    def replace_cell_in_memory(self, result: dict[str, Any]) -> None:
        """Replace any previous in-memory cell for this key, then append result."""
        key = self.cell_key(result)
        self.cells = [cell for cell in self.cells if self.cell_key(cell) != key]
        self.cells.append(result)

    def run_cell(self, repeat: int, item: base.TaskItem, method_id: str) -> dict[str, Any]:
        """Run one Codex repeat cell and persist the cell JSON."""
        result_path = self.result_path(repeat, item.task_id, method_id)
        if result_path.exists():
            existing = load_json(result_path)
            if isinstance(existing, dict) and existing.get("status") == "success":
                return existing
            if isinstance(existing, dict):
                self.archive_failed_attempt(result_path)
        self.configure_experiment_module()
        method_name, method_fn = base.exp.METHODS[method_id]
        cell_slug = f"codex__r{repeat:02d}__{item.task_id}__{method_id}"
        self.current_cell = {"cell_slug": cell_slug, "calls": []}
        self.current_call_index = 0
        base.exp.reset_token_tracker()
        started = time.time()
        baseline = self.baseline_rows[(item.task_id, method_id)]
        base_record = {
            "phase": "codex_matched_tier3",
            "tier": 3,
            "repeat_index": repeat,
            "baseline_repeat_index": BASELINE_REPEAT,
            "cell_slug": cell_slug,
            "model_key": "codex",
            "subject_backend": "codex",
            "subject_model": CODEX_MODEL,
            "subject_label": "Codex CLI default",
            "requested_model": "CLI default",
            "task_id": item.task_id,
            "task_name": item.task.name,
            "task_source": item.source,
            "task_source_index": item.source_index,
            "method": method_id,
            "method_name": method_name,
            "effort": EFFORT,
            "language": LANGUAGE,
            "injection_mode": INJECTION_MODE,
            "deep_n": DEEP_N,
            "fresh_n": FRESH_N,
            "judge_backend": JUDGE_BACKEND,
            "judge_model": JUDGE_MODEL,
            "started_at": now_iso(),
            "provider_arm": "openai_codex_cli",
            "matched_reference_path": self.references[item.task_id][method_id]["path"],
            "baseline_repeat_1_codex_path": baseline["codex_path"],
            "baseline_repeat_1_codex_source": baseline.get("codex_source"),
            "same_input": True,
            "same_role": True,
            "same_scoring": True,
            "same_prompt_text": True if method_id == "single" else "partial",
            "caveat": "Codex subject Tier 3 with fixed Claude judge scoring; not pure no-Claude-call run",
        }
        try:
            print(f"RUN r{repeat} {item.task_id}::{method_id}", flush=True)
            output = method_fn(item.task)
            judgment = self.score_output(item.task, output)
            metrics = self.calculate_metrics(item.task, judgment)
            elapsed = round(time.time() - started, 1)
            result = {
                **base_record,
                "status": "success",
                "elapsed_seconds": elapsed,
                "task": asdict(item.task),
                "output": output,
                "output_sha256": sha256_text(output),
                **metrics,
                "token_usage": base.exp.get_token_usage(),
                "judgment": judgment,
                "calls": self.current_cell.get("calls", []),
            }
            print(
                f"OK  r{repeat} {item.task_id}::{method_id} "
                f"f1={metrics['f1']:.3f} recall={metrics['recall']:.3f}",
                flush=True,
            )
        except CliFailure as exc:
            elapsed = round(time.time() - started, 1)
            result = {
                **base_record,
                "status": "failure",
                "elapsed_seconds": elapsed,
                "failure_class": exc.failure_class,
                "error": exc.message,
                "task": asdict(item.task),
                "calls": self.current_cell.get("calls", []),
            }
            print(
                f"FAIL r{repeat} {item.task_id}::{method_id} "
                f"{exc.failure_class}: {exc.message[:160]}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.time() - started, 1)
            result = {
                **base_record,
                "status": "failure",
                "elapsed_seconds": elapsed,
                "failure_class": "codex_runner_error",
                "error": repr(exc),
                "task": asdict(item.task),
                "calls": self.current_cell.get("calls", []),
            }
            print(
                f"FAIL r{repeat} {item.task_id}::{method_id} runner_error: {repr(exc)[:160]}",
                flush=True,
            )
        finally:
            self.current_cell = None
        write_json(result_path, result)
        self.replace_cell_in_memory(result)
        self.write_summary()
        return result

    def archive_failed_attempt(self, result_path: Path) -> None:
        """Preserve an existing failed cell before retrying it on resume."""
        attempt_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = self.run_dir / "failed_attempts"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{result_path.stem}__{attempt_tag}.json"
        result_path.replace(archive_path)
        raw_path = self.raw_dir / result_path.stem
        if raw_path.exists():
            raw_archive_dir = self.run_dir / "raw_calls_failed_attempts"
            raw_archive_dir.mkdir(parents=True, exist_ok=True)
            raw_path.replace(raw_archive_dir / f"{result_path.stem}__{attempt_tag}")

    def run(self, max_cells: int | None = None) -> None:
        """Run remaining repeat 2/3 cells, stopping on quota/capacity/timeout."""
        completed = self.successful_current_keys()
        runnable = [
            (repeat, item, method)
            for repeat, item, method in self.todo_cells
            if (repeat, item.task_id, method) not in completed
        ]
        if max_cells is not None:
            runnable = runnable[:max_cells]
        print(
            f"Tier 3: {len(self.task_items)} tasks | {len(METHODS)} methods | "
            f"baseline repeat 1 = {len(self.baseline_rows)} cells | "
            f"{len(runnable)} new cells to run now",
            flush=True,
        )
        for index, (repeat, item, method) in enumerate(runnable, start=1):
            print(f"[{index}/{len(runnable)}] r{repeat} {item.task_id}::{method}", flush=True)
            cell = self.run_cell(repeat, item, method)
            if cell.get("status") != "success" and self.stop_on_capacity and self.should_stop_after_failure(cell):
                self.stop_reason = {
                    "phase": "codex_matched_tier3",
                    "failure_class": cell.get("failure_class"),
                    "cell_slug": cell.get("cell_slug"),
                    "repeat_index": cell.get("repeat_index"),
                    "reason": "Clear quota/capacity/timeout; stopping Tier 3 run.",
                }
                self.write_summary()
                return
        self.write_summary()

    def new_success_rows(self) -> list[dict[str, Any]]:
        """Return successful repeat 2/3 rows in comparison-friendly form."""
        rows = []
        for cell in self.cells:
            if cell.get("status") != "success":
                continue
            ref = self.references[cell["task_id"]][cell["method"]]["metrics"]
            rows.append(
                {
                    "repeat_index": cell["repeat_index"],
                    "task_id": cell["task_id"],
                    "method": cell["method"],
                    "status": "success",
                    "codex_path": str(self.result_path(cell["repeat_index"], cell["task_id"], cell["method"])),
                    "codex_source": "current_tier3",
                    "codex_f1": cell["f1"],
                    "codex_recall": cell["recall"],
                    "claude_f1": ref["f1"],
                    "claude_recall": ref["recall"],
                    "claude_reference_path": self.references[cell["task_id"]][cell["method"]]["path"],
                }
            )
        return rows

    def all_success_rows(self) -> list[dict[str, Any]]:
        """Return baseline repeat 1 plus successful repeat 2/3 rows."""
        rows = list(self.baseline_rows.values())
        rows.extend(self.new_success_rows())
        for row in rows:
            row["codex_minus_claude_f1"] = round(row["codex_f1"] - row["claude_f1"], 4)
            row["codex_minus_claude_recall"] = round(
                row["codex_recall"] - row["claude_recall"], 4
            )
        return sorted(rows, key=lambda r: (r["repeat_index"], r["task_id"], r["method"]))

    def build_coverage(self) -> dict[str, Any]:
        """Build baseline, new repeat, and total Tier 3 coverage counts."""
        successes = [cell for cell in self.cells if cell.get("status") == "success"]
        failures = [cell for cell in self.cells if cell.get("status") != "success"]
        attempted_by_repeat: dict[str, dict[str, int]] = {}
        for repeat in self.repeats:
            cells = [cell for cell in self.cells if cell.get("repeat_index") == repeat]
            attempted_by_repeat[str(repeat)] = {
                "attempted": len(cells),
                "succeeded": sum(1 for cell in cells if cell.get("status") == "success"),
                "failed": sum(1 for cell in cells if cell.get("status") != "success"),
            }
        expected_new = len(self.task_items) * len(METHODS) * len(self.repeats)
        total_target = len(self.task_items) * len(METHODS) * len(ALL_REPEATS)
        success_keys = {
            (cell["repeat_index"], cell["task_id"], cell["method"]) for cell in successes
        }
        missing_new = [
            {"repeat_index": repeat, "task_id": item.task_id, "method": method}
            for repeat in self.repeats
            for item in self.task_items
            for method in METHODS
            if (repeat, item.task_id, method) not in success_keys
        ]
        return {
            "matched_tasks": len(self.task_items),
            "methods": list(METHODS),
            "baseline_repeat_1": {
                "covered_cells": len(self.baseline_rows),
                "target_cells": len(self.task_items) * len(METHODS),
                "coverage": f"{len(self.baseline_rows)}/{len(self.task_items) * len(METHODS)}",
            },
            "new_repeats": attempted_by_repeat,
            "new_attempted_cells": len(self.cells),
            "new_succeeded_cells": len(successes),
            "new_failed_cells": len(failures),
            "new_target_cells": expected_new,
            "total_tier3_covered_cells": len(self.baseline_rows) + len(successes),
            "total_tier3_target_cells": total_target,
            "total_tier3_coverage": f"{len(self.baseline_rows) + len(successes)}/{total_target}",
            "missing_new_cells": missing_new,
        }

    def build_failure_distribution(self) -> dict[str, int]:
        """Return failure_class distribution for repeat 2/3 cells."""
        counter = Counter(
            cell.get("failure_class", "unknown_failure")
            for cell in self.cells
            if cell.get("status") != "success"
        )
        return dict(sorted(counter.items()))

    def build_method_metrics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate Claude reference and repeated Codex metrics by method."""
        out: dict[str, Any] = {}
        for method in METHODS:
            claude_refs = [
                self.references[item.task_id][method]["metrics"] for item in self.task_items
            ]
            method_rows = [row for row in rows if row["method"] == method]
            out[method] = {
                "claude_reference": {
                    "n": len(claude_refs),
                    "mean_f1": mean([row["f1"] for row in claude_refs]),
                    "mean_recall": mean([row["recall"] for row in claude_refs]),
                },
                "codex_repeated": {
                    "n": len(method_rows),
                    "mean_f1": mean([row["codex_f1"] for row in method_rows]),
                    "mean_recall": mean([row["codex_recall"] for row in method_rows]),
                },
            }
            if out[method]["codex_repeated"]["mean_f1"] is not None:
                out[method]["delta_codex_minus_claude"] = {
                    "delta_f1": round(
                        out[method]["codex_repeated"]["mean_f1"]
                        - out[method]["claude_reference"]["mean_f1"],
                        4,
                    ),
                    "delta_recall": round(
                        out[method]["codex_repeated"]["mean_recall"]
                        - out[method]["claude_reference"]["mean_recall"],
                        4,
                    ),
                }
            else:
                out[method]["delta_codex_minus_claude"] = {
                    "delta_f1": None,
                    "delta_recall": None,
                }
        return out

    def build_repeat_metrics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate Codex metrics by repeat and method."""
        by_repeat: dict[str, dict[str, Any]] = {}
        for repeat in ALL_REPEATS:
            by_repeat[str(repeat)] = {}
            for method in METHODS:
                sub = [
                    row
                    for row in rows
                    if row["repeat_index"] == repeat and row["method"] == method
                ]
                by_repeat[str(repeat)][method] = {
                    "n": len(sub),
                    "mean_f1": mean([row["codex_f1"] for row in sub]),
                    "mean_recall": mean([row["codex_recall"] for row in sub]),
                    "min_f1": min([row["codex_f1"] for row in sub], default=None),
                    "median_f1": median([row["codex_f1"] for row in sub]),
                    "max_f1": max([row["codex_f1"] for row in sub], default=None),
                    "min_recall": min([row["codex_recall"] for row in sub], default=None),
                    "median_recall": median([row["codex_recall"] for row in sub]),
                    "max_recall": max([row["codex_recall"] for row in sub], default=None),
                }
        return by_repeat

    def build_repeat_variance(self, repeat_metrics: dict[str, Any]) -> dict[str, Any]:
        """Summarize variability across repeat-level method means."""
        out: dict[str, Any] = {}
        for method in METHODS:
            f1_means = [
                repeat_metrics[str(repeat)][method]["mean_f1"]
                for repeat in ALL_REPEATS
                if repeat_metrics[str(repeat)][method]["mean_f1"] is not None
            ]
            recall_means = [
                repeat_metrics[str(repeat)][method]["mean_recall"]
                for repeat in ALL_REPEATS
                if repeat_metrics[str(repeat)][method]["mean_recall"] is not None
            ]
            out[method] = {
                "repeat_mean_f1": {
                    "n_repeats": len(f1_means),
                    "mean": mean(f1_means),
                    "std": stddev(f1_means),
                    "min": min(f1_means, default=None),
                    "median": median(f1_means),
                    "max": max(f1_means, default=None),
                },
                "repeat_mean_recall": {
                    "n_repeats": len(recall_means),
                    "mean": mean(recall_means),
                    "std": stddev(recall_means),
                    "min": min(recall_means, default=None),
                    "median": median(recall_means),
                    "max": max(recall_means, default=None),
                },
            }
        return out

    def build_instability_flags(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flag task-method rows with large repeat-to-repeat swings or missing repeats."""
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["task_id"], row["method"])].append(row)
        flags = []
        for item in self.task_items:
            for method in METHODS:
                sub = grouped.get((item.task_id, method), [])
                repeats_present = sorted(row["repeat_index"] for row in sub)
                f1s = [row["codex_f1"] for row in sub]
                recalls = [row["codex_recall"] for row in sub]
                f1_range = round(max(f1s) - min(f1s), 4) if f1s else None
                recall_range = round(max(recalls) - min(recalls), 4) if recalls else None
                reasons = []
                if repeats_present != list(ALL_REPEATS):
                    reasons.append("missing_repeat")
                if f1_range is not None and f1_range >= INSTABILITY_F1_RANGE_THRESHOLD:
                    reasons.append("f1_range")
                if recall_range is not None and recall_range >= INSTABILITY_RECALL_RANGE_THRESHOLD:
                    reasons.append("recall_range")
                if reasons:
                    flags.append(
                        {
                            "task_id": item.task_id,
                            "method": method,
                            "repeats_present": repeats_present,
                            "f1_range": f1_range,
                            "recall_range": recall_range,
                            "reasons": reasons,
                            "values": [
                                {
                                    "repeat_index": row["repeat_index"],
                                    "f1": row["codex_f1"],
                                    "recall": row["codex_recall"],
                                }
                                for row in sorted(sub, key=lambda r: r["repeat_index"])
                            ],
                        }
                    )
        return flags

    def build_tier1_vs_tier3_conclusion(
        self,
        method_metrics: dict[str, Any],
        repeat_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare Tier 1 repeat-1 deltas with current repeated Tier 3 deltas."""
        out: dict[str, Any] = {}
        for method in METHODS:
            tier1_f1 = repeat_metrics["1"][method]["mean_f1"]
            tier1_recall = repeat_metrics["1"][method]["mean_recall"]
            claude = method_metrics[method]["claude_reference"]
            tier1_delta_f1 = round(tier1_f1 - claude["mean_f1"], 4)
            tier1_delta_recall = round(tier1_recall - claude["mean_recall"], 4)
            tier3_delta = method_metrics[method]["delta_codex_minus_claude"]
            f1_change = None
            recall_change = None
            if tier3_delta["delta_f1"] is not None:
                f1_change = round(tier3_delta["delta_f1"] - tier1_delta_f1, 4)
                recall_change = round(tier3_delta["delta_recall"] - tier1_delta_recall, 4)
            out[method] = {
                "tier1_repeat1_delta_f1": tier1_delta_f1,
                "tier1_repeat1_delta_recall": tier1_delta_recall,
                "tier3_repeated_delta_f1": tier3_delta["delta_f1"],
                "tier3_repeated_delta_recall": tier3_delta["delta_recall"],
                "change_from_tier1_delta_f1": f1_change,
                "change_from_tier1_delta_recall": recall_change,
                "direction": self.direction_label(tier1_delta_f1, tier3_delta["delta_f1"]),
            }
        return out

    def direction_label(self, tier1_delta: float, tier3_delta: float | None) -> str:
        """Return a compact maintained/weakened/reversed label."""
        if tier3_delta is None:
            return "incomplete"
        if tier1_delta == 0:
            return "new_nonzero" if tier3_delta != 0 else "maintained_tie"
        if (tier1_delta < 0 < tier3_delta) or (tier1_delta > 0 > tier3_delta):
            return "reversed"
        if abs(tier3_delta) < abs(tier1_delta):
            return "weakened"
        if abs(tier3_delta) > abs(tier1_delta):
            return "strengthened"
        return "maintained"

    def build_claim_boundaries(self, coverage: dict[str, Any]) -> dict[str, list[str]]:
        """Return paper-safe and forbidden claims for the current Tier 3 state."""
        return {
            "paper_safe_claims": [
                "This is an Opus 4.7 matched subset Tier 3 replication, not a full corpus replication.",
                "The task set is the 33-task Tier 1 matched set from MATCHED_TASKS.json.",
                f"Baseline repeat 1 is read-only and covers {coverage['baseline_repeat_1']['coverage']} cells.",
                "Repeats 2/3 are Codex subject cells using provider_arm=openai_codex_cli.",
                "Codex subject Tier 3 with fixed Claude judge scoring; not pure no-Claude-call run.",
                "same_input=true, same_role=true, same_scoring=true.",
                "single has same_prompt_text=true; ccr/ploidy are partial because downstream prompts depend on prior model outputs.",
            ],
            "forbidden_claims": [
                "Do not claim full Claude 8k+ corpus replication.",
                "Do not call this Codex-only; scoring uses fixed Claude judge calls.",
                "Do not claim strict byte-identical prompts for CCR or Ploidy downstream prompts.",
                "Do not generalize to the full corpus or other provider families from this matched subset.",
                "Do not merge these rows with Gemini legacy, Antigravity, or Claude subject arms.",
                "Do not claim new Claude subject/reference cells were generated.",
            ],
        }

    def write_summary(self) -> None:
        """Write summary JSON and a compact Markdown report."""
        rows = self.all_success_rows()
        coverage = self.build_coverage()
        method_metrics = self.build_method_metrics(rows)
        repeat_metrics = self.build_repeat_metrics(rows)
        repeat_variance = self.build_repeat_variance(repeat_metrics)
        summary = {
            "run_dir": str(self.run_dir),
            "updated_at": now_iso(),
            "status": "stopped" if self.stop_reason else "running_or_complete",
            "stop_reason": self.stop_reason,
            "caveat": "Codex subject Tier 3 with fixed Claude judge scoring; not pure no-Claude-call run",
            "coverage": coverage,
            "failure_class_distribution": self.build_failure_distribution(),
            "method_metrics": method_metrics,
            "repeat_metrics": repeat_metrics,
            "repeat_variance": repeat_variance,
            "tier1_vs_tier3_conclusion": self.build_tier1_vs_tier3_conclusion(
                method_metrics,
                repeat_metrics,
            ),
            "instability_flags": self.build_instability_flags(rows),
            "comparison_rows": rows,
            "claim_boundaries": self.build_claim_boundaries(coverage),
            "parity_policy": {
                "same_input": True,
                "same_role": True,
                "same_scoring": True,
                "same_prompt_text": {
                    "single": True,
                    "ccr": "partial",
                    "ploidy": "partial",
                },
            },
            "cells": self.cells,
        }
        if (
            self.stop_reason is None
            and coverage["total_tier3_covered_cells"] == coverage["total_tier3_target_cells"]
        ):
            summary["status"] = "complete"
        write_json(self.run_dir / "summary.json", summary)
        self.write_report(summary)

    def write_report(self, summary: dict[str, Any]) -> None:
        """Write a human-readable Tier 3 report."""
        cov = summary["coverage"]
        lines = [
            "# Codex Matched Tier 3 Report",
            "",
            summary["caveat"],
            "",
            f"Status: {summary['status']}",
            f"Stop reason: {summary['stop_reason']}",
            f"Baseline repeat 1: {cov['baseline_repeat_1']['coverage']}",
            (
                f"New repeats: {cov['new_succeeded_cells']} succeeded / "
                f"{cov['new_failed_cells']} failed / {cov['new_attempted_cells']} attempted "
                f"(target {cov['new_target_cells']})"
            ),
            f"Total Tier 3 coverage: {cov['total_tier3_coverage']}",
            "",
            "## Method Means",
            "",
            "| method | Claude F1 | Claude Recall | Codex F1 | Codex Recall | Delta F1 | Delta Recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method in METHODS:
            row = summary["method_metrics"][method]
            delta = row["delta_codex_minus_claude"]
            lines.append(
                f"| {method} | {row['claude_reference']['mean_f1']} | "
                f"{row['claude_reference']['mean_recall']} | "
                f"{row['codex_repeated']['mean_f1']} | "
                f"{row['codex_repeated']['mean_recall']} | "
                f"{delta['delta_f1']} | {delta['delta_recall']} |"
            )
        lines.extend(
            [
                "",
                "## Repeat Variance",
                "",
                "| method | F1 std | F1 min/median/max | Recall std | Recall min/median/max |",
                "|---|---:|---|---:|---|",
            ]
        )
        for method in METHODS:
            row = summary["repeat_variance"][method]
            f1 = row["repeat_mean_f1"]
            recall = row["repeat_mean_recall"]
            lines.append(
                f"| {method} | {f1['std']} | {f1['min']}/{f1['median']}/{f1['max']} | "
                f"{recall['std']} | {recall['min']}/{recall['median']}/{recall['max']} |"
            )
        lines.extend(
            [
                "",
                "## Tier 1 vs Tier 3",
                "",
                "| method | Tier 1 Delta F1 | Tier 3 Delta F1 | Direction | Tier 1 Delta Recall | Tier 3 Delta Recall |",
                "|---|---:|---:|---|---:|---:|",
            ]
        )
        for method in METHODS:
            row = summary["tier1_vs_tier3_conclusion"][method]
            lines.append(
                f"| {method} | {row['tier1_repeat1_delta_f1']} | "
                f"{row['tier3_repeated_delta_f1']} | {row['direction']} | "
                f"{row['tier1_repeat1_delta_recall']} | {row['tier3_repeated_delta_recall']} |"
            )
        lines.extend(
            [
                "",
                "## Failure Classes",
                "",
                "```json",
                json.dumps(summary["failure_class_distribution"], indent=2),
                "```",
                "",
                "## Instability Flags",
                "",
                f"Flagged task-methods: {len(summary['instability_flags'])}",
                "",
                "## Paper-Safe Claims",
                "",
            ]
        )
        lines.extend(f"- {claim}" for claim in summary["claim_boundaries"]["paper_safe_claims"])
        lines.extend(["", "## Forbidden Claims", ""])
        lines.extend(f"- {claim}" for claim in summary["claim_boundaries"]["forbidden_claims"])
        if summary.get("stop_reason"):
            lines.extend(["", "## Stop Reason", "", "```json", json.dumps(summary["stop_reason"], indent=2), "```"])
        (self.run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Codex matched-subset Tier 3 repeats")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory. Defaults to experiments/results/<timestamp>_codex_matched_tier3",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run dir and skip successful repeat cells.",
    )
    parser.add_argument(
        "--continue-on-capacity",
        action="store_true",
        help="Continue after quota/capacity/timeout instead of stopping.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional execution cap for smoke tests or controlled resumes.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        nargs="+",
        default=list(DEFAULT_REPEATS),
        help="Repeat indexes to run. Defaults to 2 3.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write RUN_SPEC/MATCHED_TASKS/summary/report without model calls.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the Tier 3 matched Codex repeat experiment."""
    args = parse_args()
    repeats = tuple(args.repeats)
    if any(repeat <= BASELINE_REPEAT for repeat in repeats):
        raise SystemExit("This runner only executes repeats greater than baseline repeat 1")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or RESULTS_ROOT / f"{timestamp}_codex_matched_tier3"
    runner = CodexMatchedTier3Runner(
        run_dir=run_dir,
        repeats=repeats,
        stop_on_capacity=not args.continue_on_capacity,
    )
    runner.prepare(resume=args.resume)
    if args.plan_only:
        print(f"RUN_DIR={run_dir}", flush=True)
        return 0
    runner.run(max_cells=args.max_cells)
    print(f"RUN_DIR={run_dir}", flush=True)
    if runner.stop_reason:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
