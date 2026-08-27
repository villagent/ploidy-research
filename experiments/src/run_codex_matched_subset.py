"""Run the Codex matched-subset expansion against Opus 4.7 references.

This runner is deliberately narrower than ``run_clean_cross_model_parity.py``:

* reference cells are existing Claude Opus 4.7 results only;
* subject cells are Codex CLI only;
* Gemini CLI and Antigravity are never invoked;
* the prior clean-parity Codex cells are counted as completed coverage but are
  not re-run;
* all new artifacts are written to a fresh matched-subset run directory.

The scoring prompt and metric formula are copied from the clean parity runner
and ``run_experiment.py`` so the Codex cells remain comparable to the existing
Claude reference cells. In this file, "Codex only" means the subject-model arm;
the judge backend remains fixed to Claude Opus 4.7 for same-scoring parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
PRIOR_CLEAN_PARITY_DIR = RESULTS_ROOT / "20260617_060030_clean_cross_model_parity"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_experiment as exp  # noqa: E402
from task_model import Task  # noqa: E402
from tasks_gradient import GRADIENT_TASKS  # noqa: E402
from tasks_longcontext import LONG_CONTEXT_TASKS  # noqa: E402


METHODS = ("single", "ccr", "ploidy")
REFERENCE_MODEL = "claude-opus-4-7"
CODEX_MODEL = "codex-default"
JUDGE_BACKEND = "claude"
JUDGE_MODEL = "claude-opus-4-7"
EFFORT = "high"
LANGUAGE = "en"
INJECTION_MODE = "raw"
DEEP_N = 1
FRESH_N = 1
TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class TaskItem:
    """Task registry entry used by the matched-subset runner."""

    task_id: str
    source: str
    source_index: int
    task: Task


@dataclass
class CliFailure(Exception):
    """Structured CLI failure with a stable failure class."""

    backend: str
    failure_class: str
    message: str
    call_record: dict[str, Any] | None = None

    def __str__(self) -> str:
        """Return the human-readable failure message."""
        return self.message


def sha256_text(text: str) -> str:
    """Return a SHA-256 hex digest for prompt and response identity checks."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    """Return a local ISO timestamp with second precision."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> Any | None:
    """Load JSON, returning ``None`` for non-JSON or unreadable files."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def write_json(path: Path, data: Any) -> None:
    """Write stable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def model_of(cell: dict[str, Any]) -> str | None:
    """Return the subject model field across old and clean-parity schemas."""
    return cell.get("model") or cell.get("subject_model")


def task_id_of(cell: dict[str, Any]) -> str | None:
    """Return the task id field across old and clean-parity schemas."""
    task = cell.get("task")
    return cell.get("task_id") or (task.get("id") if isinstance(task, dict) else None)


def cell_status_is_success(cell: dict[str, Any]) -> bool:
    """Return true for old cells with absent status and new explicit successes."""
    return cell.get("status") in (None, "success")


def is_reference_condition(cell: dict[str, Any]) -> bool:
    """Return true when a Claude cell matches the fixed reference condition."""
    return (
        model_of(cell) == REFERENCE_MODEL
        and cell_status_is_success(cell)
        and cell.get("method") in METHODS
        and cell.get("effort") == EFFORT
        and cell.get("language") == LANGUAGE
        and cell.get("injection_mode") == INJECTION_MODE
        and int(cell.get("deep_n", 1)) == DEEP_N
        and int(cell.get("fresh_n", 1)) == FRESH_N
    )


def metrics_from_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Extract or recompute F1/recall metrics from a result cell."""
    found = int(cell.get("found", 0) or 0)
    partial = int(cell.get("partial", 0) or 0)
    missed = int(cell.get("missed", 0) or 0)
    total = int(cell.get("total_gt", 0) or 0)
    bonus = int(cell.get("bonus_findings", 0) or 0)
    recall = cell.get("recall")
    if recall is None and total:
        recall = (found + 0.5 * partial) / total
    precision = cell.get("precision")
    if precision is None:
        precision = (found + 0.5 * partial) / max(found + partial + bonus, 1)
    f1 = cell.get("f1")
    if f1 is None:
        f1 = 2 * precision * recall / max(precision + recall, 0.001)
    return {
        "found": found,
        "partial": partial,
        "missed": missed,
        "total_gt": total,
        "bonus_findings": bonus,
        "precision": round(float(precision), 4),
        "recall": round(float(recall or 0.0), 4),
        "f1": round(float(f1 or 0.0), 4),
    }


def task_registry() -> dict[str, TaskItem]:
    """Return every task id addressable by this matched-subset run."""
    out: dict[str, TaskItem] = {}
    for index, task in enumerate(LONG_CONTEXT_TASKS):
        out[task.id] = TaskItem(task.id, "longcontext", index, task)
    for index, task in enumerate(GRADIENT_TASKS):
        out[task.id] = TaskItem(task.id, "gradient", index, task)
    return out


def reference_candidates(results_root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Collect existing Opus 4.7 reference candidates by task and method."""
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path in sorted(results_root.rglob("*.json")):
        if (path.parent / ".contaminated.json").exists():
            continue
        cell = load_json(path)
        if not isinstance(cell, dict) or not is_reference_condition(cell):
            continue
        task_id = task_id_of(cell)
        method = cell.get("method")
        if not task_id or method not in METHODS:
            continue
        row = {
            "path": str(path),
            "task_id": task_id,
            "method": method,
            "metrics": metrics_from_cell(cell),
            "source_run_dir": str(path.parent),
        }
        by_task.setdefault(task_id, {}).setdefault(method, []).append(row)
    return by_task


def choose_references(
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    registry: dict[str, TaskItem],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Choose one latest non-contaminated reference per matched task-method."""
    chosen: dict[str, dict[str, dict[str, Any]]] = {}
    for task_id in sorted(candidates):
        if task_id not in registry:
            continue
        by_method = candidates[task_id]
        if not all(by_method.get(method) for method in METHODS):
            continue
        chosen[task_id] = {method: sorted(by_method[method], key=lambda r: r["path"])[-1] for method in METHODS}
    return chosen


def prior_clean_codex_cells(
    references: dict[str, dict[str, dict[str, Any]]],
    prior_dir: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load prior clean-parity Codex cells that count as completed coverage."""
    prior: dict[str, dict[str, dict[str, Any]]] = {}
    for task_id in references:
        for method in METHODS:
            path = prior_dir / f"codex__{task_id}__{method}.json"
            cell = load_json(path)
            if (
                isinstance(cell, dict)
                and model_of(cell) == CODEX_MODEL
                and cell.get("status") == "success"
                and cell.get("method") == method
            ):
                prior.setdefault(task_id, {})[method] = {
                    "path": str(path),
                    "task_id": task_id,
                    "method": method,
                    "metrics": metrics_from_cell(cell),
                    "source_run_dir": str(path.parent),
                    "source": "prior_clean_cross_model_parity",
                }
    return prior


def codex_version() -> dict[str, Any]:
    """Capture Codex CLI version without invoking other provider CLIs."""
    try:
        result = subprocess.run(["codex", "--version"], capture_output=True, text=True, timeout=30)
        return {
            "command": ["codex", "--version"],
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"command": ["codex", "--version"], "error": repr(exc)}


def classify_failure(
    backend: str,
    returncode: int | None,
    stdout: str,
    stderr: str,
    response: str,
    timed_out: bool,
) -> str | None:
    """Map raw CLI failure text to stable failure classes."""
    text = f"{stdout}\n{stderr}\n{response}".lower()
    if timed_out:
        return f"{backend}_timeout"
    if returncode == 0:
        if backend == "claude":
            quota_banners = (
                "claude usage limit reached",
                "you've reached your usage limit",
                "you have reached your usage limit",
                "usage limit. resets",
                "limit reached. resets",
            )
            if any(phrase in response[:1000].lower() for phrase in quota_banners):
                return "claude_quota"
        return None
    quota_phrases = (
        "usage limit",
        "rate limit",
        "rate_limit",
        "quota",
        "too many requests",
        "429",
        "you've reached your",
        "hit your limit",
        "usage_limit",
    )
    capacity_phrases = (
        "at capacity",
        "server capacity",
        "over capacity",
        "overloaded",
        "service unavailable",
        "503",
        "502",
        "resource exhausted",
    )
    if any(phrase in text for phrase in quota_phrases):
        return f"{backend}_quota"
    if any(phrase in text for phrase in capacity_phrases):
        return f"{backend}_capacity"
    if returncode is not None and returncode != 0:
        return f"{backend}_cli_error"
    return None


def classify_prompt(prompt: str, role: str) -> str:
    """Best-effort call-stage label for prompt parity auditing."""
    if role == "judge":
        return "judge_scoring"
    if "A colleague produced this analysis:" in prompt:
        return "ccr_fresh_review"
    if "debate was held. Synthesize into a final verdict." in prompt:
        return "ploidy_convergence"
    if "Now, a reviewer with NO project context found:" in prompt:
        return "ploidy_deep_challenge"
    if "Now, a reviewer with deep project context found:" in prompt:
        return "ploidy_fresh_challenge"
    if "You have NO background context about this system. Review based purely" in prompt:
        return "fresh_position"
    if "For each issue, classify your confidence as HIGH, MEDIUM, or LOW." in prompt:
        return "deep_position_with_confidence"
    if "List every bug, risk, or issue you can find. Be specific and technical." in prompt:
        return "deep_review"
    return "subject_other"


class MatchedCodexRunner:
    """Run new Codex cells for the Opus 4.7 matched subset."""

    def __init__(self, run_dir: Path, stop_on_capacity: bool = True) -> None:
        self.run_dir = run_dir
        self.raw_dir = run_dir / "raw_calls"
        self.tmp_root = run_dir / "tmp"
        self.stop_on_capacity = stop_on_capacity
        self.cli_versions = {"codex": codex_version()}
        self.call_lock = threading.Lock()
        self.current_cell: dict[str, Any] | None = None
        self.current_call_index = 0
        self.cells: list[dict[str, Any]] = []
        self.stop_reason: dict[str, Any] | None = None
        self.registry = task_registry()
        self.references = choose_references(reference_candidates(RESULTS_ROOT), self.registry)
        self.prior_codex = prior_clean_codex_cells(self.references, PRIOR_CLEAN_PARITY_DIR)
        self.task_items = [self.registry[task_id] for task_id in sorted(self.references)]
        self.todo_cells = [
            (item, method)
            for item in self.task_items
            for method in METHODS
            if method not in self.prior_codex.get(item.task_id, {})
        ]

    def prepare(self, resume: bool = False) -> None:
        """Create run directories and write the run specification."""
        if self.run_dir.exists() and not resume:
            raise FileExistsError(f"Refusing to overwrite existing run dir: {self.run_dir}")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.load_existing_cells()
        self.write_run_spec()
        self.write_summary()

    def load_existing_cells(self) -> None:
        """Load already-produced cells when resuming an interrupted run."""
        self.cells = []
        if not self.run_dir.exists():
            return
        for path in sorted(self.run_dir.glob("codex__*.json")):
            cell = load_json(path)
            if isinstance(cell, dict):
                self.cells.append(cell)

    def write_run_spec(self) -> None:
        """Write the matched task list and immutable run spec."""
        matched_tasks = []
        for item in self.task_items:
            refs = self.references[item.task_id]
            prior = self.prior_codex.get(item.task_id, {})
            matched_tasks.append(
                {
                    "task_id": item.task_id,
                    "task_name": item.task.name,
                    "source": item.source,
                    "source_index": item.source_index,
                    "context_sha256": sha256_text(item.task.context),
                    "prompt_sha256": sha256_text(item.task.prompt),
                    "ground_truth_sha256": sha256_text(
                        json.dumps(item.task.ground_truth, ensure_ascii=False)
                    ),
                    "claude_reference": refs,
                    "prior_codex_clean_parity": prior,
                    "todo_methods": [method for method in METHODS if method not in prior],
                }
            )
        write_json(self.run_dir / "MATCHED_TASKS.json", matched_tasks)
        spec = {
            "created_at": now_iso(),
            "purpose": "codex_matched_subset_expansion",
            "full_corpus_replication": False,
            "matched_replication": True,
            "reference_policy": {
                "reference_model": REFERENCE_MODEL,
                "reference_filter": {
                    "effort": EFFORT,
                    "language": LANGUAGE,
                    "injection_mode": INJECTION_MODE,
                    "deep_n": DEEP_N,
                    "fresh_n": FRESH_N,
                    "methods": list(METHODS),
                    "exclude_dirs_with_contaminated_marker": True,
                },
                "task_inclusion_rule": "task has successful Opus 4.7 single, ccr, and ploidy cells",
                "cell_selection_rule": "latest path-sorted non-contaminated reference per task-method",
            },
            "codex_execution_policy": {
                "subject_backend": "codex",
                "subject_model": CODEX_MODEL,
                "new_subject_provider_arms": ["openai_codex_cli"],
                "forbidden_subject_arms": [
                    "anthropic_claude_cli",
                    "google_gemini_cli_legacy",
                    "google_antigravity",
                ],
                "prior_clean_parity_dir": str(PRIOR_CLEAN_PARITY_DIR),
                "prior_clean_parity_cells_counted": sum(
                    1 for task_id in self.prior_codex for _ in self.prior_codex[task_id]
                ),
                "existing_non_clean_codex_cells_ignored": True,
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
                "prompt_source": "experiments/src/run_experiment.py::judge_result prompt text, copied verbatim",
                "metric_formula_source": "experiments/src/run_experiment.py::_run_one_method",
                "note": "Judge calls are scoring calls, not new Claude subject-arm cells.",
            },
            "matrix": {
                "matched_tasks": len(self.task_items),
                "methods": list(METHODS),
                "total_target_cells": len(self.task_items) * len(METHODS),
                "prior_clean_parity_codex_cells": sum(
                    len(methods) for methods in self.prior_codex.values()
                ),
                "new_cells_to_attempt": len(self.todo_cells),
            },
            "cli_versions": self.cli_versions,
        }
        write_json(self.run_dir / "RUN_SPEC.json", spec)
        lines = [
            "# Codex Matched-Subset Expansion",
            "",
            "This is a matched replication, not a full Claude corpus replication.",
            "",
            f"- matched tasks: {len(self.task_items)}",
            f"- target cells: {len(self.task_items) * len(METHODS)}",
            f"- prior clean-parity Codex cells counted: {spec['matrix']['prior_clean_parity_codex_cells']}",
            f"- new Codex cells to attempt: {len(self.todo_cells)}",
            "",
            "Gemini CLI and Antigravity are not invoked by this runner.",
        ]
        (self.run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def invoke_cli(
        self,
        backend: str,
        prompt: str,
        requested_model: str | None,
        role: str,
        system_prompt: str | None = None,
    ) -> str:
        """Invoke Codex subject or Claude judge CLI and record raw artifacts."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        if self.current_cell is None:
            raise RuntimeError("No current cell set for CLI invocation")
        with self.call_lock:
            self.current_call_index += 1
            call_index = self.current_call_index
        cell_slug = self.current_cell["cell_slug"]
        call_dir = self.raw_dir / cell_slug / f"{call_index:02d}_{role}_{backend}"
        call_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = call_dir / "prompt.txt"
        stdout_path = call_dir / "stdout.txt"
        stderr_path = call_dir / "stderr.txt"
        response_path = call_dir / "response.txt"
        prompt_path.write_text(full_prompt, encoding="utf-8")

        neutral_cwd = Path(tempfile.mkdtemp(prefix=f"{backend}-", dir=self.tmp_root))
        outfile_path: Path | None = None
        input_text: str | None = None
        if backend == "codex":
            outfile_path = call_dir / "codex_last_message.txt"
            cmd = [
                "codex",
                "exec",
                "--output-last-message",
                str(outfile_path),
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--cd",
                str(neutral_cwd),
                "-",
            ]
            if requested_model and requested_model != CODEX_MODEL:
                cmd[2:2] = ["-m", requested_model]
            input_text = full_prompt
        elif backend == "claude":
            cmd = ["claude", "--print"]
            if requested_model:
                cmd.extend(["--model", requested_model])
            if system_prompt:
                cmd.extend(["--system-prompt", system_prompt])
            cmd.append(prompt)
        else:
            raise ValueError(f"Backend is forbidden in this runner: {backend}")

        started = time.time()
        timed_out = False
        try:
            result = subprocess.run(
                cmd,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=str(neutral_cwd),
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            returncode: int | None = result.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            returncode = None
        elapsed = round(time.time() - started, 3)
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        if backend == "codex" and outfile_path is not None and outfile_path.exists():
            response = outfile_path.read_text(encoding="utf-8").strip()
        else:
            response = stdout.strip()
        response_path.write_text(response, encoding="utf-8")

        failure_class = classify_failure(backend, returncode, stdout, stderr, response, timed_out)
        call_record = {
            "call_index": call_index,
            "role": role,
            "prompt_kind": classify_prompt(prompt, role),
            "backend": backend,
            "requested_model": requested_model or "CLI default",
            "resolved_model": self.resolved_model_note(backend, requested_model),
            "command": self.sanitize_command_for_record(cmd, backend),
            "cwd": str(neutral_cwd),
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_seconds": elapsed,
            "prompt_sha256": sha256_text(full_prompt),
            "response_sha256": sha256_text(response),
            "prompt_path": str(prompt_path.relative_to(self.run_dir)),
            "stdout_path": str(stdout_path.relative_to(self.run_dir)),
            "stderr_path": str(stderr_path.relative_to(self.run_dir)),
            "response_path": str(response_path.relative_to(self.run_dir)),
            "failure_class": failure_class,
        }
        write_json(call_dir / "call.json", call_record)
        self.current_cell.setdefault("calls", []).append(call_record)
        if failure_class:
            message = stderr.strip() or stdout.strip() or response[:2000] or f"{backend} call failed"
            raise CliFailure(backend, failure_class, message[:2000], call_record)

        exp._track_tokens(exp._estimate_tokens(full_prompt), exp._estimate_tokens(response))
        return response

    def resolved_model_note(self, backend: str, requested_model: str | None) -> str:
        """Return model-resolution metadata when the CLI does not expose it."""
        if backend == "codex":
            version = self.cli_versions.get("codex", {}).get("stdout") or "version unavailable"
            if requested_model and requested_model != CODEX_MODEL:
                return f"requested explicitly ({requested_model}); CLI version: {version}"
            return f"CLI default; resolved model not reported by Codex CLI; CLI version: {version}"
        if requested_model:
            return f"requested explicitly ({requested_model})"
        return "CLI default"

    def sanitize_command_for_record(self, cmd: list[str], backend: str) -> list[str]:
        """Return command metadata without embedding long prompt text."""
        recorded = list(cmd)
        if backend == "claude" and recorded:
            recorded[-1] = "<prompt>"
            if "--system-prompt" in recorded:
                idx = recorded.index("--system-prompt")
                if idx + 1 < len(recorded):
                    recorded[idx + 1] = "<system_prompt>"
        return recorded

    def call_subject_llm(
        self,
        prompt: str,
        model: str | None = None,
        effort: str | None = None,
        lang: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Replacement for ``run_experiment.call_llm`` during subject calls."""
        del model, effort
        actual_lang = lang or LANGUAGE
        if actual_lang != "en" and actual_lang in exp.LANGUAGES:
            prompt = f"{prompt}\n\n{exp.LANGUAGES[actual_lang]}"
        return self.invoke_cli(
            backend="codex",
            prompt=prompt,
            requested_model=None,
            role="subject",
            system_prompt=system_prompt,
        )

    def score_output(self, task: Task, output: str) -> dict[str, Any]:
        """Score one cell with the canonical judge prompt and fixed judge."""
        gt_list = "\n".join(f"  {i + 1}. {gt}" for i, gt in enumerate(task.ground_truth))
        prompt = (
            "You are an expert judge evaluating a code review / architecture analysis.\n\n"
            f"GROUND TRUTH issues (known correct answers):\n{gt_list}\n\n"
            f"REVIEWER OUTPUT:\n{output}\n\n"
            "For EACH ground truth issue, determine:\n"
            "- FOUND: clearly identified (even if worded differently)\n"
            "- PARTIAL: hinted at but not fully articulated\n"
            "- MISSED: not identified\n\n"
            "Also count additional valid issues NOT in ground truth (bonus findings).\n\n"
            "Respond in this EXACT JSON format and nothing else:\n"
            '{"scores": [{"ground_truth_index": 1, "verdict": "FOUND", "evidence": "..."}'
            ', ...], "bonus_findings": 0, "summary": "..."}'
        )
        judgment = self.invoke_cli(
            backend=JUDGE_BACKEND,
            prompt=prompt,
            requested_model=JUDGE_MODEL,
            role="judge",
        )
        try:
            json_start = judgment.index("{")
            json_end = judgment.rindex("}") + 1
            return json.loads(judgment[json_start:json_end])
        except (ValueError, json.JSONDecodeError) as exc:
            raise CliFailure(JUDGE_BACKEND, "judge_parse_error", str(exc)) from exc

    def calculate_metrics(self, task: Task, judgment: dict[str, Any]) -> dict[str, Any]:
        """Calculate metrics using the canonical run_experiment formula."""
        found = sum(1 for s in judgment.get("scores", []) if s.get("verdict") == "FOUND")
        partial = sum(1 for s in judgment.get("scores", []) if s.get("verdict") == "PARTIAL")
        missed = sum(1 for s in judgment.get("scores", []) if s.get("verdict") == "MISSED")
        total = len(task.ground_truth)
        recall = (found + 0.5 * partial) / total if total else 0.0
        bonus = judgment.get("bonus_findings", 0)
        precision = (found + 0.5 * partial) / max(found + partial + bonus, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)
        return {
            "found": found,
            "partial": partial,
            "missed": missed,
            "total_gt": total,
            "bonus_findings": bonus,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        }

    def configure_experiment_module(self) -> None:
        """Set run_experiment globals to the matched-subset condition."""
        exp.TASKS.clear()
        exp.TASKS.extend(item.task for item in self.task_items)
        exp.EFFORT = EFFORT
        exp.LANGUAGE = LANGUAGE
        exp.INJECTION_MODE = INJECTION_MODE
        exp.DEEP_N = DEEP_N
        exp.FRESH_N = FRESH_N
        exp.CONTEXT_PCT = 100
        exp.call_llm = self.call_subject_llm

    def run_cell(self, item: TaskItem, method_id: str) -> dict[str, Any]:
        """Run one Codex task-method cell and persist the cell JSON."""
        result_path = self.run_dir / f"codex__{item.task_id}__{method_id}.json"
        if result_path.exists():
            existing = load_json(result_path)
            if isinstance(existing, dict) and existing.get("status") == "success":
                return existing
            if isinstance(existing, dict):
                self.archive_failed_attempt(result_path)
        self.configure_experiment_module()
        method_name, method_fn = exp.METHODS[method_id]
        cell_slug = f"codex__{item.task_id}__{method_id}"
        self.current_cell = {"cell_slug": cell_slug, "calls": []}
        self.current_call_index = 0
        exp.reset_token_tracker()
        started = time.time()
        base = {
            "phase": "matched_subset",
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
            "same_input": True,
            "same_role": True,
            "same_scoring": True,
            "same_prompt_text": True if method_id == "single" else "partial",
        }
        try:
            print(f"RUN {cell_slug}", flush=True)
            output = method_fn(item.task)
            judgment = self.score_output(item.task, output)
            metrics = self.calculate_metrics(item.task, judgment)
            elapsed = round(time.time() - started, 1)
            result = {
                **base,
                "status": "success",
                "elapsed_seconds": elapsed,
                "task": asdict(item.task),
                "output": output,
                "output_sha256": sha256_text(output),
                **metrics,
                "token_usage": exp.get_token_usage(),
                "judgment": judgment,
                "calls": self.current_cell.get("calls", []),
            }
            print(f"OK  {cell_slug} f1={metrics['f1']:.3f} recall={metrics['recall']:.3f}", flush=True)
        except CliFailure as exc:
            elapsed = round(time.time() - started, 1)
            result = {
                **base,
                "status": "failure",
                "elapsed_seconds": elapsed,
                "failure_class": exc.failure_class,
                "error": exc.message,
                "task": asdict(item.task),
                "calls": self.current_cell.get("calls", []),
            }
            print(f"FAIL {cell_slug} {exc.failure_class}: {exc.message[:160]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.time() - started, 1)
            result = {
                **base,
                "status": "failure",
                "elapsed_seconds": elapsed,
                "failure_class": "codex_runner_error",
                "error": repr(exc),
                "task": asdict(item.task),
                "calls": self.current_cell.get("calls", []),
            }
            print(f"FAIL {cell_slug} runner_error: {repr(exc)[:160]}", flush=True)
        finally:
            self.current_cell = None
        write_json(result_path, result)
        self.cells.append(result)
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

    def should_stop_after_failure(self, cell: dict[str, Any]) -> bool:
        """Return true when quota, capacity, or timeout should stop the run."""
        failure_class = cell.get("failure_class", "")
        return any(x in failure_class for x in ("quota", "capacity", "timeout"))

    def successful_current_keys(self) -> set[tuple[str, str]]:
        """Return current-run successful cell keys."""
        return {
            (cell["task_id"], cell["method"])
            for cell in self.cells
            if cell.get("status") == "success"
        }

    def run(self, max_cells: int | None = None) -> None:
        """Run all remaining matched cells, stopping on clear capacity failures."""
        completed = self.successful_current_keys()
        runnable = [
            (item, method)
            for item, method in self.todo_cells
            if (item.task_id, method) not in completed
        ]
        if max_cells is not None:
            runnable = runnable[:max_cells]
        print(
            f"Matched subset: {len(self.task_items)} tasks | "
            f"{len(self.task_items) * len(METHODS)} target cells | "
            f"{sum(len(v) for v in self.prior_codex.values())} prior clean-parity cells | "
            f"{len(runnable)} cells to run now",
            flush=True,
        )
        for index, (item, method) in enumerate(runnable, start=1):
            print(f"[{index}/{len(runnable)}] {item.task_id}::{method}", flush=True)
            cell = self.run_cell(item, method)
            if cell.get("status") != "success" and self.stop_on_capacity and self.should_stop_after_failure(cell):
                self.stop_reason = {
                    "phase": "matched_subset",
                    "failure_class": cell.get("failure_class"),
                    "cell_slug": cell.get("cell_slug"),
                    "reason": "Clear quota/capacity/timeout; stopping matched-subset run.",
                }
                self.write_summary()
                return
        self.write_summary()

    def current_codex_rows(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return successful current-run Codex rows by task and method."""
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for cell in self.cells:
            if cell.get("status") != "success":
                continue
            out.setdefault(cell["task_id"], {})[cell["method"]] = {
                "path": str(self.run_dir / f"{cell['cell_slug']}.json"),
                "task_id": cell["task_id"],
                "method": cell["method"],
                "metrics": metrics_from_cell(cell),
                "source_run_dir": str(self.run_dir),
                "source": "current_matched_subset",
            }
        return out

    def combined_codex_rows(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return prior clean-parity plus current successful Codex rows."""
        combined = {task_id: dict(methods) for task_id, methods in self.prior_codex.items()}
        for task_id, methods in self.current_codex_rows().items():
            combined.setdefault(task_id, {}).update(methods)
        return combined

    def build_comparison_rows(self) -> list[dict[str, Any]]:
        """Build Claude reference vs Codex matched comparison rows."""
        rows: list[dict[str, Any]] = []
        codex = self.combined_codex_rows()
        for item in self.task_items:
            for method in METHODS:
                ref = self.references[item.task_id][method]
                cx = codex.get(item.task_id, {}).get(method)
                row: dict[str, Any] = {
                    "task_id": item.task_id,
                    "method": method,
                    "claude_reference_path": ref["path"],
                    "claude_f1": ref["metrics"]["f1"],
                    "claude_recall": ref["metrics"]["recall"],
                    "codex_status": "success" if cx else "missing",
                    "codex_path": cx["path"] if cx else None,
                    "codex_source": cx.get("source") if cx else None,
                    "codex_f1": cx["metrics"]["f1"] if cx else None,
                    "codex_recall": cx["metrics"]["recall"] if cx else None,
                }
                if cx:
                    row["codex_minus_claude_f1"] = round(
                        cx["metrics"]["f1"] - ref["metrics"]["f1"], 4
                    )
                    row["codex_minus_claude_recall"] = round(
                        cx["metrics"]["recall"] - ref["metrics"]["recall"], 4
                    )
                rows.append(row)
        return rows

    def build_method_deltas(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Aggregate Codex minus Claude deltas by method."""
        out: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            sub = [r for r in rows if r["method"] == method and r["codex_status"] == "success"]
            if not sub:
                out[method] = {"n": 0, "mean_delta_f1": None, "mean_delta_recall": None}
                continue
            out[method] = {
                "n": len(sub),
                "mean_delta_f1": round(
                    sum(r["codex_minus_claude_f1"] for r in sub) / len(sub), 4
                ),
                "mean_delta_recall": round(
                    sum(r["codex_minus_claude_recall"] for r in sub) / len(sub), 4
                ),
            }
        return out

    def build_claim_boundaries(self, comparison_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
        """Return paper-claim guidance derived from current coverage."""
        complete_tasks = [
            item.task_id
            for item in self.task_items
            if all(
                r["codex_status"] == "success"
                for r in comparison_rows
                if r["task_id"] == item.task_id
            )
        ]
        return {
            "can_claim": [
                "This is a matched replication over tasks where Opus 4.7 already has single, ccr, and ploidy cells.",
                "It does not attempt to reproduce the full 8,238-cell Claude corpus.",
                f"Codex matched coverage is {len(complete_tasks)}/{len(self.task_items)} tasks at the time of this report.",
                "Single uses identical initial prompt text; CCR and Ploidy are prompt-template matched but output-conditioned after the first subject response.",
            ],
            "cannot_claim": [
                "Do not claim full-corpus cross-model replication.",
                "Do not claim independent model-family validation beyond Codex CLI default.",
                "Do not treat n=1 per matched cell as a stable effect-size estimate.",
                "Do not merge Codex CLI results with Gemini CLI, Antigravity, or Claude provider arms.",
            ],
        }

    def build_coverage(self, comparison_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Build task and cell coverage counts."""
        current_success = [cell for cell in self.cells if cell.get("status") == "success"]
        current_fail = [cell for cell in self.cells if cell.get("status") != "success"]
        combined = self.combined_codex_rows()
        complete_tasks = [
            item.task_id
            for item in self.task_items
            if all(method in combined.get(item.task_id, {}) for method in METHODS)
        ]
        covered_cells = sum(
            1 for item in self.task_items for method in METHODS if method in combined.get(item.task_id, {})
        )
        return {
            "matched_tasks": len(self.task_items),
            "target_cells": len(self.task_items) * len(METHODS),
            "prior_clean_parity_complete_tasks": sum(
                1 for item in self.task_items if all(method in self.prior_codex.get(item.task_id, {}) for method in METHODS)
            ),
            "prior_clean_parity_cells": sum(len(methods) for methods in self.prior_codex.values()),
            "current_attempted_cells": len(self.cells),
            "current_succeeded_cells": len(current_success),
            "current_failed_cells": len(current_fail),
            "combined_covered_cells": covered_cells,
            "combined_complete_tasks": len(complete_tasks),
            "combined_complete_task_ids": complete_tasks,
            "missing_cells": [
                {"task_id": item.task_id, "method": method}
                for item in self.task_items
                for method in METHODS
                if method not in combined.get(item.task_id, {})
            ],
            "comparison_success_rows": sum(
                1 for row in comparison_rows if row["codex_status"] == "success"
            ),
        }

    def write_summary(self) -> None:
        """Write summary JSON and a compact Markdown report."""
        comparison_rows = self.build_comparison_rows()
        coverage = self.build_coverage(comparison_rows)
        method_deltas = self.build_method_deltas(comparison_rows)
        summary = {
            "run_dir": str(self.run_dir),
            "updated_at": now_iso(),
            "status": "stopped" if self.stop_reason else "running_or_complete",
            "stop_reason": self.stop_reason,
            "coverage": coverage,
            "method_deltas_codex_minus_claude": method_deltas,
            "comparison_rows": comparison_rows,
            "claim_boundaries": self.build_claim_boundaries(comparison_rows),
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
            and coverage["combined_covered_cells"] == coverage["target_cells"]
        ):
            summary["status"] = "complete"
        write_json(self.run_dir / "summary.json", summary)
        self.write_report(summary)

    def write_report(self, summary: dict[str, Any]) -> None:
        """Write a human-readable matched-subset report."""
        cov = summary["coverage"]
        lines = [
            "# Codex Matched-Subset Report",
            "",
            f"Status: {summary['status']}",
            f"Coverage: {cov['combined_complete_tasks']}/{cov['matched_tasks']} tasks, "
            f"{cov['combined_covered_cells']}/{cov['target_cells']} cells",
            f"Current run: {cov['current_succeeded_cells']} succeeded / "
            f"{cov['current_failed_cells']} failed / {cov['current_attempted_cells']} attempted",
            "",
            "## Method Deltas",
            "",
            "| method | n | mean Codex-Claude F1 | mean Codex-Claude recall |",
            "|---|---:|---:|---:|",
        ]
        for method, row in summary["method_deltas_codex_minus_claude"].items():
            lines.append(
                f"| {method} | {row['n']} | {row['mean_delta_f1']} | {row['mean_delta_recall']} |"
            )
        lines.extend(
            [
                "",
                "## Missing Cells",
                "",
                "| task | method |",
                "|---|---|",
            ]
        )
        for row in cov["missing_cells"][:120]:
            lines.append(f"| {row['task_id']} | {row['method']} |")
        if not cov["missing_cells"]:
            lines.append("| - | - |")
        lines.extend(
            [
                "",
                "## Claim Boundaries",
                "",
                "Can claim:",
            ]
        )
        lines.extend(f"- {claim}" for claim in summary["claim_boundaries"]["can_claim"])
        lines.append("")
        lines.append("Cannot claim:")
        lines.extend(f"- {claim}" for claim in summary["claim_boundaries"]["cannot_claim"])
        if summary.get("stop_reason"):
            lines.extend(["", "## Stop Reason", "", "```json", json.dumps(summary["stop_reason"], indent=2), "```"])
        (self.run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Codex matched-subset expansion")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory. Defaults to experiments/results/<timestamp>_codex_matched_subset",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run dir and skip successful current-run cells.",
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
        help="Optional execution cap for controlled resumes.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the matched task list and summary without model calls.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the Codex matched-subset expansion."""
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or RESULTS_ROOT / f"{timestamp}_codex_matched_subset"
    runner = MatchedCodexRunner(run_dir=run_dir, stop_on_capacity=not args.continue_on_capacity)
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
