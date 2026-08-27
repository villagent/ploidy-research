"""Run a clean cross-model parity experiment.

This wrapper reuses the canonical task/method prompt builders from
``run_experiment.py`` but fixes two issues that make the old cross-model and
diversity-recovery artifacts insufficient for strict parity:

* subject model calls and scoring calls are separated;
* every CLI invocation stores raw stdout/stderr plus a prompt hash.

The default bounded matrix is intentionally small:

* tasks: three original long-context tasks from the gradient long tier
* methods: single, ccr, ploidy
* models: Claude, Gemini, Codex
* repeats: one
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_experiment as exp  # noqa: E402
from tasks_gradient import GRADIENT_TASKS  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    """A subject model/backend to run in the parity matrix."""

    key: str
    backend: str
    label: str
    requested_model: str | None
    metadata_model: str


@dataclass
class CliFailure(Exception):
    """Structured CLI failure with a stable class for summary tables."""

    backend: str
    failure_class: str
    message: str
    call_record: dict[str, Any] | None = None

    def __str__(self) -> str:
        """Return the human-readable failure message."""
        return self.message


DEFAULT_TASK_INDICES = (11, 14, 17)
DEFAULT_METHODS = ("single", "ccr", "ploidy")
DEFAULT_MODELS = (
    ModelSpec("claude", "claude", "Claude Opus 4.7", "claude-opus-4-7", "claude-opus-4-7"),
    ModelSpec("gemini", "gemini", "Gemini 2.5 Pro", "gemini-2.5-pro", "gemini-2.5-pro"),
    ModelSpec("codex", "codex", "Codex CLI default", None, "codex-default"),
)
JUDGE_BACKEND = "claude"
JUDGE_MODEL = "claude-opus-4-7"
EFFORT = "high"
LANGUAGE = "en"
INJECTION_MODE = "raw"
DEEP_N = 1
FRESH_N = 1
TIMEOUT_SECONDS = 600


def sha256_text(text: str) -> str:
    """Return a SHA-256 hex digest for prompt and output identity checks."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    """Return a local ISO-like timestamp with second precision."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def get_cli_versions() -> dict[str, dict[str, Any]]:
    """Capture CLI versions required by the run contract."""
    commands = {
        "codex": ["codex", "--version"],
        "gemini": ["gemini", "--version"],
        "claude": ["claude", "--version"],
    }
    out: dict[str, dict[str, Any]] = {}
    for name, cmd in commands.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out[name] = {
                "command": cmd,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except Exception as exc:  # noqa: BLE001
            out[name] = {"command": cmd, "error": repr(exc)}
    return out


def classify_failure(backend: str, returncode: int | None, stdout: str, stderr: str, timed_out: bool) -> str:
    """Map raw CLI failure text to backend-specific failure classes."""
    text = f"{stdout}\n{stderr}".lower()
    if timed_out:
        return f"{backend}_timeout"
    if backend == "gemini" and "modelnotfound" in text:
        return "gemini_model_unavailable"
    quota_phrases = (
        "usage limit",
        "rate limit",
        "rate_limit",
        "quota",
        "too many requests",
        "429",
    )
    capacity_phrases = (
        "capacity",
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
    return f"{backend}_unknown_failure"


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


class CleanParityRunner:
    """Run the bounded parity matrix and persist all raw artifacts."""

    def __init__(self, run_dir: Path, stop_on_capacity: bool = True) -> None:
        self.run_dir = run_dir
        self.raw_dir = run_dir / "raw_calls"
        self.tmp_root = run_dir / "tmp"
        self.stop_on_capacity = stop_on_capacity
        self.cli_versions = get_cli_versions()
        self.call_lock = threading.Lock()
        self.current_cell: dict[str, Any] | None = None
        self.current_call_index = 0
        self.cells: list[dict[str, Any]] = []
        self.stop_reason: dict[str, Any] | None = None

    def prepare(self) -> None:
        """Create run directories and write the pre-run specification."""
        if self.run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run dir: {self.run_dir}")
        self.raw_dir.mkdir(parents=True)
        self.tmp_root.mkdir(parents=True)
        self.write_run_spec()

    def write_json(self, path: Path, data: Any) -> None:
        """Write JSON with stable formatting."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_run_spec(self) -> None:
        """Write the bounded matrix before any model cell runs."""
        tasks = [GRADIENT_TASKS[i] for i in DEFAULT_TASK_INDICES]
        spec = {
            "created_at": now_iso(),
            "purpose": "clean_cross_model_parity",
            "not_recovery": True,
            "bounded_matrix": {
                "tasks": [
                    {
                        "index": idx,
                        "id": task.id,
                        "name": task.name,
                        "context_sha256": sha256_text(task.context),
                        "prompt_sha256": sha256_text(task.prompt),
                        "ground_truth_sha256": sha256_text(json.dumps(task.ground_truth, ensure_ascii=False)),
                    }
                    for idx, task in zip(DEFAULT_TASK_INDICES, tasks, strict=True)
                ],
                "methods": list(DEFAULT_METHODS),
                "models": [asdict(m) for m in DEFAULT_MODELS],
                "repeats": 1,
                "effort": EFFORT,
                "language": LANGUAGE,
                "injection_mode": INJECTION_MODE,
                "deep_n": DEEP_N,
                "fresh_n": FRESH_N,
            },
            "smoke_cell": {
                "model": "codex",
                "task_index": DEFAULT_TASK_INDICES[0],
                "task_id": tasks[0].id,
                "method": "single",
                "counts_as_matrix_cell": True,
            },
            "scoring": {
                "backend": JUDGE_BACKEND,
                "model": JUDGE_MODEL,
                "prompt_source": "experiments/src/run_experiment.py::judge_result prompt text, copied verbatim",
                "metric_formula_source": "experiments/src/run_experiment.py::_run_one_method",
            },
            "cli_versions": self.cli_versions,
            "codex_policy": {
                "forbidden": ["--full-auto"],
                "used": [
                    "codex exec",
                    "--output-last-message <file>",
                    "--ephemeral",
                    "--sandbox read-only",
                    "--ignore-rules",
                    "--skip-git-repo-check",
                    "--cd <temp>",
                ],
                "model_metadata": "requested_model is CLI default; Codex CLI does not report resolved model in exec output.",
            },
        }
        self.write_json(self.run_dir / "RUN_SPEC.json", spec)
        readme = [
            "# Clean Cross-Model Parity Run",
            "",
            "This run is a fresh bounded parity matrix, not a diversity recovery pass.",
            "",
            "Matrix:",
            f"- tasks: {', '.join(t.id for t in tasks)}",
            f"- methods: {', '.join(DEFAULT_METHODS)}",
            f"- models: {', '.join(m.key for m in DEFAULT_MODELS)}",
            "- repeats: 1",
            "",
            "Scoring is fixed to Claude Opus 4.7 for every subject model.",
        ]
        (self.run_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    def invoke_cli(
        self,
        backend: str,
        prompt: str,
        requested_model: str | None,
        role: str,
        system_prompt: str | None = None,
    ) -> str:
        """Invoke a provider CLI and record raw prompt/stdout/stderr."""
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
        if backend == "claude":
            cmd = ["claude", "--print"]
            if requested_model:
                cmd.extend(["--model", requested_model])
            if system_prompt:
                cmd.extend(["--system-prompt", system_prompt])
            cmd.append(prompt)
        elif backend == "gemini":
            cmd = ["gemini", "-p", full_prompt]
            if requested_model:
                cmd.extend(["-m", requested_model])
        elif backend == "codex":
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
            if requested_model:
                cmd[2:2] = ["-m", requested_model]
            input_text = full_prompt
        else:
            raise ValueError(f"Unknown backend: {backend}")

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

        failure_class: str | None = None
        if timed_out or returncode != 0:
            failure_class = classify_failure(backend, returncode, stdout, stderr, timed_out)
        if backend == "claude" and response:
            low = response.lower()
            if any(
                phrase in low
                for phrase in (
                    "claude usage limit reached",
                    "you've reached your usage limit",
                    "you have reached your usage limit",
                    "usage limit. resets",
                    "limit reached. resets",
                )
            ):
                failure_class = "claude_quota"
        if backend == "gemini" and response:
            low = response.lower()
            if "resource exhausted" in low or "modelnotfound" in low:
                failure_class = classify_failure(backend, returncode, stdout, stderr + response, timed_out)

        recorded_cmd = self.sanitize_command_for_record(cmd, backend)
        call_record = {
            "call_index": call_index,
            "role": role,
            "prompt_kind": classify_prompt(prompt, role),
            "backend": backend,
            "requested_model": requested_model or "CLI default",
            "resolved_model": self.resolved_model_note(backend, requested_model),
            "command": recorded_cmd,
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
        self.write_json(call_dir / "call.json", call_record)
        self.current_cell.setdefault("calls", []).append(call_record)
        if failure_class:
            message = stderr.strip() or stdout.strip() or f"{backend} call failed"
            raise CliFailure(backend, failure_class, message[:2000], call_record)

        exp._track_tokens(exp._estimate_tokens(full_prompt), exp._estimate_tokens(response))
        return response

    def resolved_model_note(self, backend: str, requested_model: str | None) -> str:
        """Return model-resolution metadata when the CLI exposes no exact value."""
        version = self.cli_versions.get(backend, {}).get("stdout") or "version unavailable"
        if requested_model:
            return f"requested explicitly ({requested_model}); CLI version: {version}"
        return f"CLI default; resolved model not reported by {backend} CLI; CLI version: {version}"

    def sanitize_command_for_record(self, cmd: list[str], backend: str) -> list[str]:
        """Return command metadata without embedding long prompt text."""
        recorded = list(cmd)
        if backend == "claude" and recorded:
            recorded[-1] = "<prompt>"
            if "--system-prompt" in recorded:
                idx = recorded.index("--system-prompt")
                if idx + 1 < len(recorded):
                    recorded[idx + 1] = "<system_prompt>"
        if backend == "gemini" and "-p" in recorded:
            idx = recorded.index("-p")
            if idx + 1 < len(recorded):
                recorded[idx + 1] = "<prompt>"
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
        if self.current_cell is None:
            raise RuntimeError("Subject call outside a cell")
        model_spec: ModelSpec = self.current_cell["model_spec"]
        return self.invoke_cli(
            backend=model_spec.backend,
            prompt=prompt,
            requested_model=model_spec.requested_model,
            role="subject",
            system_prompt=system_prompt,
        )

    def score_output(self, task: Any, method_name: str, output: str) -> dict[str, Any]:
        """Score one cell with the canonical judge prompt and fixed judge model."""
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

    def calculate_metrics(self, task: Any, judgment: dict[str, Any]) -> dict[str, Any]:
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
        """Set run_experiment globals to the bounded parity condition."""
        exp.TASKS.clear()
        exp.TASKS.extend(GRADIENT_TASKS)
        exp.EFFORT = EFFORT
        exp.LANGUAGE = LANGUAGE
        exp.INJECTION_MODE = INJECTION_MODE
        exp.DEEP_N = DEEP_N
        exp.FRESH_N = FRESH_N
        exp.CONTEXT_PCT = 100
        exp.call_llm = self.call_subject_llm

    def run_cell(self, model_spec: ModelSpec, task_index: int, method_id: str, phase: str) -> dict[str, Any]:
        """Run one model/task/method cell and persist the cell JSON."""
        self.configure_experiment_module()
        task = GRADIENT_TASKS[task_index]
        method_name, method_fn = exp.METHODS[method_id]
        cell_slug = f"{model_spec.key}__{task.id}__{method_id}"
        result_path = self.run_dir / f"{cell_slug}.json"
        self.current_cell = {
            "cell_slug": cell_slug,
            "model_spec": model_spec,
            "calls": [],
        }
        self.current_call_index = 0
        exp.reset_token_tracker()
        started = time.time()
        base = {
            "phase": phase,
            "cell_slug": cell_slug,
            "model_key": model_spec.key,
            "subject_backend": model_spec.backend,
            "subject_model": model_spec.metadata_model,
            "subject_label": model_spec.label,
            "requested_model": model_spec.requested_model or "CLI default",
            "task_index": task_index,
            "task_id": task.id,
            "task_name": task.name,
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
        }
        try:
            print(f"RUN {cell_slug}", flush=True)
            output = method_fn(task)
            judgment = self.score_output(task, method_name, output)
            metrics = self.calculate_metrics(task, judgment)
            elapsed = round(time.time() - started, 1)
            result = {
                **base,
                "status": "success",
                "elapsed_seconds": elapsed,
                "task": asdict(task),
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
                "task": asdict(task),
                "calls": self.current_cell.get("calls", []),
            }
            print(f"FAIL {cell_slug} {exc.failure_class}: {exc.message[:160]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.time() - started, 1)
            result = {
                **base,
                "status": "failure",
                "elapsed_seconds": elapsed,
                "failure_class": f"{model_spec.backend}_runner_error",
                "error": repr(exc),
                "task": asdict(task),
                "calls": self.current_cell.get("calls", []),
            }
            print(f"FAIL {cell_slug} runner_error: {repr(exc)[:160]}", flush=True)
        finally:
            self.current_cell = None
        self.write_json(result_path, result)
        self.cells.append(result)
        self.write_summary()
        return result

    def should_stop_after_failure(self, cell: dict[str, Any]) -> bool:
        """Return true when a clear quota/capacity/timeout should stop the run."""
        failure_class = cell.get("failure_class", "")
        return any(x in failure_class for x in ("quota", "capacity", "timeout"))

    def matrix_cells(self) -> list[tuple[ModelSpec, int, str]]:
        """Return the bounded matrix in deterministic order."""
        return [
            (model, task_index, method)
            for model in DEFAULT_MODELS
            for task_index in DEFAULT_TASK_INDICES
            for method in DEFAULT_METHODS
        ]

    def run(self) -> None:
        """Run smoke first, then the remaining bounded matrix."""
        smoke = (DEFAULT_MODELS[2], DEFAULT_TASK_INDICES[0], "single")
        smoke_result = self.run_cell(*smoke, phase="smoke")
        if smoke_result["status"] != "success":
            self.stop_reason = {
                "phase": "smoke",
                "failure_class": smoke_result.get("failure_class"),
                "cell_slug": smoke_result.get("cell_slug"),
                "reason": "Smoke cell failed; bounded matrix not started.",
            }
            self.write_summary()
            return

        smoke_key = (smoke[0].key, smoke[1], smoke[2])
        for model_spec, task_index, method_id in self.matrix_cells():
            key = (model_spec.key, task_index, method_id)
            if key == smoke_key:
                continue
            cell = self.run_cell(model_spec, task_index, method_id, phase="matrix")
            if cell["status"] != "success" and self.stop_on_capacity and self.should_stop_after_failure(cell):
                self.stop_reason = {
                    "phase": "matrix",
                    "failure_class": cell.get("failure_class"),
                    "cell_slug": cell.get("cell_slug"),
                    "reason": "Clear quota/capacity/timeout; stopping bounded run.",
                }
                self.write_summary()
                return
        self.write_summary()

    def build_success_table(self) -> list[dict[str, Any]]:
        """Build model x method x task success/failure rows."""
        return [
            {
                "model": c["model_key"],
                "method": c["method"],
                "task_id": c["task_id"],
                "status": c["status"],
                "failure_class": c.get("failure_class"),
                "f1": c.get("f1"),
                "recall": c.get("recall"),
            }
            for c in self.cells
        ]

    def build_score_comparison(self) -> list[dict[str, Any]]:
        """Build score comparison rows across successful cells."""
        rows: list[dict[str, Any]] = []
        for task_index in DEFAULT_TASK_INDICES:
            task_id = GRADIENT_TASKS[task_index].id
            for method in DEFAULT_METHODS:
                row: dict[str, Any] = {"task_id": task_id, "method": method}
                for model in DEFAULT_MODELS:
                    match = next(
                        (
                            c
                            for c in self.cells
                            if c["model_key"] == model.key
                            and c["task_index"] == task_index
                            and c["method"] == method
                        ),
                        None,
                    )
                    row[f"{model.key}_status"] = match.get("status") if match else "not_run"
                    row[f"{model.key}_f1"] = match.get("f1") if match else None
                    row[f"{model.key}_recall"] = match.get("recall") if match else None
                rows.append(row)
        return rows

    def build_prompt_parity_table(self) -> list[dict[str, Any]]:
        """Summarize strict prompt-hash parity by task/method/call kind."""
        buckets: dict[tuple[str, str, str, str], dict[str, set[str]]] = {}
        for cell in self.cells:
            for call in cell.get("calls", []):
                if call["role"] != "subject":
                    continue
                key = (cell["task_id"], cell["method"], call["role"], call["prompt_kind"])
                buckets.setdefault(key, {}).setdefault(cell["model_key"], set()).add(call["prompt_sha256"])
        rows: list[dict[str, Any]] = []
        for (task_id, method, role, prompt_kind), by_model in sorted(buckets.items()):
            observed = {model: sorted(hashes) for model, hashes in sorted(by_model.items())}
            union = {h for hashes in by_model.values() for h in hashes}
            strict_same = len(union) == 1 and len(by_model) == len(DEFAULT_MODELS)
            rows.append(
                {
                    "task_id": task_id,
                    "method": method,
                    "role": role,
                    "prompt_kind": prompt_kind,
                    "strict_same_prompt_text": strict_same,
                    "hashes_by_model": observed,
                    "note": (
                        "Identical text across all models."
                        if strict_same
                        else "May be output-conditioned or not all models completed this call."
                    ),
                }
            )
        return rows

    def build_parity_audit(self) -> dict[str, Any]:
        """Build the requested same_input/same_prompt/same_role/same_scoring audit."""
        return {
            "same_input": {
                "status": "met_for_run_cells",
                "basis": "All models use identical Task objects from tasks_gradient.py at indices 11,14,17.",
            },
            "same_prompt_text": {
                "status": "partially_met",
                "basis": (
                    "Initial subject prompts are template-identical. Strict full-cell prompt text is not "
                    "identical for CCR/Ploidy after the first model response, because later prompts include "
                    "that model's own prior output. Judge prompts also include reviewer output, so their "
                    "templates are identical but hashes differ by subject output."
                ),
            },
            "same_role": {
                "status": "met_at_template_level",
                "basis": "Each model is run through the same method function and Deep/Fresh role text.",
            },
            "same_scoring": {
                "status": "met",
                "basis": f"Every cell is scored by fixed judge {JUDGE_BACKEND}:{JUDGE_MODEL} with the same scoring prompt template and metric formula.",
            },
            "prompt_parity_table": self.build_prompt_parity_table(),
        }

    def write_summary(self) -> None:
        """Write summary.json and a compact Markdown report."""
        successful = [c for c in self.cells if c.get("status") == "success"]
        failed = [c for c in self.cells if c.get("status") != "success"]
        summary = {
            "run_dir": str(self.run_dir),
            "updated_at": now_iso(),
            "status": "stopped" if self.stop_reason else "running_or_complete",
            "stop_reason": self.stop_reason,
            "cell_count": len(self.cells),
            "success_count": len(successful),
            "failure_count": len(failed),
            "cli_versions": self.cli_versions,
            "success_failure_table": self.build_success_table(),
            "score_comparison": self.build_score_comparison(),
            "parity_audit": self.build_parity_audit(),
            "cells": self.cells,
        }
        if self.stop_reason is None and len(self.cells) == len(self.matrix_cells()):
            summary["status"] = "complete"
        self.write_json(self.run_dir / "summary.json", summary)
        self.write_report(summary)

    def write_report(self, summary: dict[str, Any]) -> None:
        """Write a human-readable report for the run."""
        lines = [
            "# Clean Cross-Model Parity Report",
            "",
            f"Status: {summary['status']}",
            f"Cells: {summary['success_count']} success / {summary['failure_count']} failure / {summary['cell_count']} attempted",
            "",
            "## Success / Failure",
            "",
            "| model | method | task | status | failure_class | F1 | recall |",
            "|---|---|---|---|---|---:|---:|",
        ]
        for row in summary["success_failure_table"]:
            lines.append(
                "| {model} | {method} | {task_id} | {status} | {failure_class} | {f1} | {recall} |".format(
                    **{k: "" if v is None else v for k, v in row.items()}
                )
            )
        lines.extend(
            [
                "",
                "## Score Comparison",
                "",
                "| task | method | claude F1/R | gemini F1/R | codex F1/R |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in summary["score_comparison"]:
            lines.append(
                "| {task_id} | {method} | {claude_f1}/{claude_recall} | {gemini_f1}/{gemini_recall} | {codex_f1}/{codex_recall} |".format(
                    **{k: "" if v is None else v for k, v in row.items()}
                )
            )
        audit = summary["parity_audit"]
        lines.extend(
            [
                "",
                "## Parity Audit",
                "",
                "| condition | status | basis |",
                "|---|---|---|",
            ]
        )
        for key in ("same_input", "same_prompt_text", "same_role", "same_scoring"):
            item = audit[key]
            lines.append(f"| {key} | {item['status']} | {item['basis']} |")
        if summary.get("stop_reason"):
            lines.extend(["", "## Stop Reason", "", "```json", json.dumps(summary["stop_reason"], indent=2), "```"])
        (self.run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the clean parity runner."""
    parser = argparse.ArgumentParser(description="Run clean cross-model parity bounded matrix")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit run directory. Defaults to experiments/results/<timestamp>_clean_cross_model_parity",
    )
    parser.add_argument(
        "--continue-on-capacity",
        action="store_true",
        help="Continue after quota/capacity/timeout instead of stopping the bounded run.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the clean cross-model parity experiment."""
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or RESULTS_ROOT / f"{timestamp}_clean_cross_model_parity"
    runner = CleanParityRunner(run_dir=run_dir, stop_on_capacity=not args.continue_on_capacity)
    runner.prepare()
    runner.run()
    print(f"RUN_DIR={run_dir}", flush=True)
    if runner.stop_reason:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
