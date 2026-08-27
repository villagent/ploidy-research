"""
Experiment 5: Species Diversity vs Cultural Diversity
=====================================================
Paper 1 (Context as Lifespan) 핵심 실험.

가설: context diversity (문화적 다양성) > model diversity (종 다양성)

조건:
  A) cross_model_symmetric  — 다른 모델, 같은 context (종 다양성)
     Opus(deep) vs Gemini(deep): 둘 다 full context
  B) same_model_asymmetric  — 같은 모델, 다른 context (문화 다양성)
     Opus(deep) vs Opus(fresh): context 비대칭
  C) cross_model_asymmetric — 다른 모델, 다른 context (종 + 문화)
     Opus(deep) vs Gemini(fresh): 모델도 다르고 context도 다름
  D) single                 — baseline
     Opus(deep) 단독

기대 결과: B ≥ A (context diversity가 model diversity보다 중요)
          C ≥ B이나 marginal gain 작음 (종 다양성 추가 기여 미미)

Usage:
    python experiments/run_diversity_experiment.py
    python experiments/run_diversity_experiment.py --tasks 0,1,2
    python experiments/run_diversity_experiment.py --secondary gemini  # 2차 모델 변경
    python experiments/run_diversity_experiment.py --repeats 3          # 반복 횟수
"""

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ─── Token Tracking ─────────────────────────────────────────────────────────

_token_tracker = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "calls": 0,
    "estimated": True,
}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _track_tokens(prompt_tokens: int, completion_tokens: int, exact: bool = False):
    _token_tracker["prompt_tokens"] += prompt_tokens
    _token_tracker["completion_tokens"] += completion_tokens
    _token_tracker["total_tokens"] += prompt_tokens + completion_tokens
    _token_tracker["calls"] += 1
    if exact:
        _token_tracker["estimated"] = False


def reset_token_tracker():
    _token_tracker.update(
        prompt_tokens=0, completion_tokens=0, total_tokens=0, calls=0, estimated=True
    )


def get_token_usage() -> dict:
    return dict(_token_tracker)


# ─── Backends ────────────────────────────────────────────────────────────────


def call_claude(
    prompt: str, model: str = "claude-opus-4-6", effort: str = "high", system_prompt: str = None
) -> str:
    cmd = ["claude", "--print", "--model", model]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    if effort and effort != "high":
        cmd.extend(["--effort", effort])
    cmd.append(prompt)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI error: {result.stderr.strip()}")
    output = result.stdout.strip()
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    _track_tokens(_estimate_tokens(full_prompt), _estimate_tokens(output))
    return output


def call_gemini(
    prompt: str, model: str = "gemini-3.1-pro", effort: str = "high", system_prompt: str = None
) -> str:
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    cmd = ["gemini", "-p", full_prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"gemini CLI error: {result.stderr.strip()}")
    output = result.stdout.strip()
    _track_tokens(_estimate_tokens(full_prompt), _estimate_tokens(output))
    return output


def call_codex(
    prompt: str, model: str = "codex-default", effort: str = "high", system_prompt: str = None
) -> str:
    import os
    import tempfile

    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    outfile = tempfile.mktemp(suffix=".txt")
    cmd = ["codex", "exec", "--output-last-message", outfile, full_prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"codex CLI error: {result.stderr.strip()}")
    try:
        with open(outfile) as f:
            output = f.read().strip()
        _track_tokens(_estimate_tokens(full_prompt), _estimate_tokens(output))
        return output
    finally:
        os.unlink(outfile) if os.path.exists(outfile) else None


BACKENDS = {
    "claude": call_claude,
    "gemini": call_gemini,
    "codex": call_codex,
}

BACKEND_MODELS = {
    "claude": "claude-opus-4-6",
    "gemini": "gemini-3.1-pro",
    "codex": "codex-default",
}


# ─── Tasks (long-context only — short에서는 차이 안 남) ─────────────────────


@dataclass
class Task:
    id: str
    name: str
    context: str
    prompt: str
    ground_truth: list[str]


TASKS = [
    Task(
        id="long_db_migration",
        name="DB Migration Strategy (biased context)",
        context=(
            "We've been running PostgreSQL 11 for 4 years. The team spent 3 months "
            "evaluating and chose pg_upgrade for the migration to PostgreSQL 16. "
            "We already have scripts written and tested in staging. The CTO approved "
            "the approach last quarter. Two team members have pg_upgrade expertise. "
            "However, we're also moving from a monolith to microservices next quarter, "
            "which will require splitting the database. Some tables have 500M+ rows "
            "with complex foreign keys. We use logical replication for the read replicas."
        ),
        prompt=(
            "Review our PostgreSQL 11→16 migration plan using pg_upgrade. "
            "The migration window is 2 hours on a Saturday. Database is 2TB with "
            "500M+ row tables. We use logical replication for read replicas. "
            "What issues or risks do you see?"
        ),
        ground_truth=[
            "pg_upgrade requires downtime — 2TB with 500M rows likely exceeds 2-hour window",
            "Logical replication subscribers must be re-synced after pg_upgrade (breaks replication)",
            "If splitting DB for microservices next quarter, migrating monolith first is potentially wasted effort — consider logical replication migration instead",
            "Foreign key constraints on 500M+ row tables make pg_upgrade extension installations risky",
            "Sunk cost bias: 3 months of pg_upgrade prep + CTO approval creates anchoring toward suboptimal approach",
        ],
    ),
    Task(
        id="long_auth_overhaul",
        name="Auth System Overhaul (biased context)",
        context=(
            "Our SAML-based SSO has been running for 5 years across 12 enterprise clients. "
            "We built it on a well-known SAML library (ruby-saml) and our security team "
            "audited it last year with no critical findings. We're planning to add OIDC "
            "support alongside SAML. The PM wants both protocols supported simultaneously "
            "for backward compatibility. Our auth team of 3 engineers has deep SAML expertise "
            "but limited OIDC experience. We've had zero auth-related security incidents."
        ),
        prompt=(
            "Review our plan to add OIDC support alongside our existing SAML SSO system. "
            "We want to maintain backward compatibility for our 12 enterprise SAML clients "
            "while offering OIDC as a new option. The auth team has 3 engineers with deep "
            "SAML expertise. What risks or issues do you see?"
        ),
        ground_truth=[
            "Dual protocol increases attack surface — session management must handle both token types",
            "ruby-saml has had critical CVEs (e.g., CVE-2024-45409) — 'no critical findings' may reflect audit scope, not actual security",
            "OIDC token refresh + SAML assertion lifetime mismatch can create session inconsistency",
            "3 engineers with 'deep SAML expertise but limited OIDC' is a knowledge asymmetry risk for the OIDC implementation",
            "Zero incidents ≠ zero vulnerabilities — survivorship bias in security assessment",
        ],
    ),
    Task(
        id="long_microservice_split",
        name="Microservice Split (biased context)",
        context=(
            "Our monolith handles 50K RPM with p99 latency of 120ms. The team has been "
            "advocating for microservices for 18 months. We've identified 6 bounded contexts "
            "and plan to extract the User, Notification, and Payment services first. "
            "We use PostgreSQL with 40+ tables and complex joins across contexts. "
            "The DevOps team set up Kubernetes last quarter. Our monitoring uses Datadog "
            "with custom dashboards. The CTO greenlit the migration after seeing success "
            "stories from similar-sized companies. We have 8 backend engineers."
        ),
        prompt=(
            "Review our plan to split a monolith (50K RPM, p99=120ms) into microservices. "
            "Starting with User, Notification, and Payment services. We have 40+ PostgreSQL "
            "tables with complex cross-context joins. 8 backend engineers, Kubernetes ready. "
            "What risks or issues do you see?"
        ),
        ground_truth=[
            "Cross-context joins will become distributed queries — likely increasing p99 latency significantly",
            "8 engineers / 6 services = understaffed for microservice operational overhead (on-call, deployment pipelines, service mesh)",
            "Payment service extraction carries PCI compliance risk — audit scope changes when payment logic moves",
            "50K RPM with 120ms p99 is already good — migration motivation may be trend-following rather than necessity",
            "Data consistency across User/Payment requires distributed transactions or saga pattern — complexity spike",
            "Notification service is often the easiest to extract — starting with Payment simultaneously is high risk",
        ],
    ),
]


# ─── Judge ──────────────────────────────────────────────────────────────────


def judge_result(task: Task, method_name: str, output: str) -> dict:
    """Judge output against ground truth using Claude Opus as unified judge."""
    gt_list = "\n".join(f"  {i + 1}. {g}" for i, g in enumerate(task.ground_truth))
    prompt = (
        f"You are a precise technical judge. Compare this review output against known ground truth.\n\n"
        f"=== GROUND TRUTH ISSUES ===\n{gt_list}\n\n"
        f"=== REVIEW OUTPUT ({method_name}) ===\n{output}\n\n"
        f"For EACH ground truth issue, respond with exactly one of:\n"
        f"  FOUND — the review clearly identifies this issue\n"
        f"  PARTIAL — the review touches on this but incompletely\n"
        f"  MISSED — the review does not mention this issue\n\n"
        f"Also count how many additional findings the review made beyond the ground truth.\n\n"
        f"Respond in this exact JSON format:\n"
        f'{{"scores": [{{"issue": "...", "verdict": "FOUND|PARTIAL|MISSED", "evidence": "..."}}], '
        f'"bonus_findings": <int>}}'
    )
    raw = call_claude(prompt, model="claude-opus-4-6", effort="high")
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"raw_judge": raw, "parse_error": True}


# ─── Methods ────────────────────────────────────────────────────────────────


def method_single(task: Task, backend: str = "claude") -> str:
    """D) Baseline: single model, full context."""
    fn = BACKENDS[backend]
    model = BACKEND_MODELS[backend]
    return fn(
        f"Context about this system:\n{task.context}\n\n{task.prompt}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        model=model,
    )


def method_cross_model_symmetric(task: Task, primary: str, secondary: str) -> str:
    """A) 종 다양성: 다른 모델, 같은 context. 둘 다 full context로 debate."""
    fn_p = BACKENDS[primary]
    fn_s = BACKENDS[secondary]
    model_p = BACKEND_MODELS[primary]
    model_s = BACKEND_MODELS[secondary]

    full_prompt = (
        f"Context about this system:\n{task.context}\n\n{task.prompt}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW."
    )

    # POSITION: 둘 다 full context
    pos_p = fn_p(full_prompt, model=model_p)
    pos_s = fn_s(full_prompt, model=model_s)

    # CHALLENGE: 서로의 결과를 리뷰
    challenge_p = fn_p(
        f"You reviewed a system and found:\n\n{pos_p}\n\n"
        f"Another reviewer (different model, SAME context as you) found:\n\n{pos_s}\n\n"
        f"For EACH of their points, respond with AGREE/CHALLENGE/SYNTHESIZE.\n"
        f"Also list anything you found that they missed.",
        model=model_p,
    )
    challenge_s = fn_s(
        f"You reviewed a system and found:\n\n{pos_s}\n\n"
        f"Another reviewer (different model, SAME context as you) found:\n\n{pos_p}\n\n"
        f"For EACH of their points, respond with AGREE/CHALLENGE/SYNTHESIZE.\n"
        f"Also list anything you found that they missed.",
        model=model_s,
    )

    # CONVERGENCE: 통합 (Claude Opus as judge)
    return call_claude(
        f"Two different AI models reviewed the same system with IDENTICAL context.\n\n"
        f"=== Model A ({primary}) Position ===\n{pos_p}\n\n"
        f"=== Model B ({secondary}) Position ===\n{pos_s}\n\n"
        f"=== Model A challenges B ===\n{challenge_p}\n\n"
        f"=== Model B challenges A ===\n{challenge_s}\n\n"
        f"Produce a final list of ALL confirmed issues with severity.",
        model="claude-opus-4-6",
    )


def method_same_model_asymmetric(task: Task, backend: str = "claude") -> str:
    """B) 문화 다양성: 같은 모델, 다른 context. Deep vs Fresh."""
    fn = BACKENDS[backend]
    model = BACKEND_MODELS[backend]

    # POSITION: Deep (full context) vs Fresh (no context)
    pos_deep = fn(
        f"Context about this system:\n{task.context}\n\n{task.prompt}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        model=model,
    )
    pos_fresh = fn(
        f"{task.prompt}\n\n"
        f"You have NO background context about this system. Review based purely on "
        f"the code/question itself.\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        model=model,
    )

    # CHALLENGE
    challenge_deep = fn(
        f"You are an experienced reviewer with full project context. You found:\n\n{pos_deep}\n\n"
        f"A reviewer with NO project context found:\n\n{pos_fresh}\n\n"
        f"For EACH of their points, respond with AGREE/CHALLENGE/SYNTHESIZE.\n"
        f"Also list anything you found that they missed.",
        model=model,
    )
    challenge_fresh = fn(
        f"You are a fresh reviewer with NO project context. You found:\n\n{pos_fresh}\n\n"
        f"A reviewer with deep project context found:\n\n{pos_deep}\n\n"
        f"For EACH of their points, respond with AGREE/CHALLENGE/SYNTHESIZE.\n"
        f"Also list anything you found that they missed.",
        model=model,
    )

    # CONVERGENCE
    return fn(
        f"A debate was held between Deep (full context) and Fresh (no context) sessions.\n\n"
        f"=== Deep Position ===\n{pos_deep}\n\n"
        f"=== Fresh Position ===\n{pos_fresh}\n\n"
        f"=== Deep challenges Fresh ===\n{challenge_deep}\n\n"
        f"=== Fresh challenges Deep ===\n{challenge_fresh}\n\n"
        f"Produce a final list of ALL confirmed issues with severity.",
        model=model,
    )


def method_cross_model_asymmetric(task: Task, primary: str, secondary: str) -> str:
    """C) 종 + 문화: 다른 모델, 다른 context. Primary=Deep, Secondary=Fresh."""
    fn_p = BACKENDS[primary]
    fn_s = BACKENDS[secondary]
    model_p = BACKEND_MODELS[primary]
    model_s = BACKEND_MODELS[secondary]

    # POSITION: Primary=Deep (full context), Secondary=Fresh (no context)
    pos_deep = fn_p(
        f"Context about this system:\n{task.context}\n\n{task.prompt}\n\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        model=model_p,
    )
    pos_fresh = fn_s(
        f"{task.prompt}\n\n"
        f"You have NO background context about this system. Review based purely on "
        f"the code/question itself.\n"
        f"List every bug, risk, or issue you can find. Be specific and technical.\n"
        f"For each issue, classify your confidence as HIGH, MEDIUM, or LOW.",
        model=model_s,
    )

    # CHALLENGE
    challenge_deep = fn_p(
        f"You are an experienced reviewer with full project context. You found:\n\n{pos_deep}\n\n"
        f"A completely different AI model with NO project context found:\n\n{pos_fresh}\n\n"
        f"For EACH of their points, respond with AGREE/CHALLENGE/SYNTHESIZE.\n"
        f"Also list anything you found that they missed.",
        model=model_p,
    )
    challenge_fresh = fn_s(
        f"You are a fresh reviewer with NO project context. You found:\n\n{pos_fresh}\n\n"
        f"A completely different AI model with deep project context found:\n\n{pos_deep}\n\n"
        f"For EACH of their points, respond with AGREE/CHALLENGE/SYNTHESIZE.\n"
        f"Also list anything you found that they missed.",
        model=model_s,
    )

    # CONVERGENCE (unified judge: Claude Opus)
    return call_claude(
        f"A cross-model asymmetric debate was held.\n"
        f"Deep session ({primary}, full context) vs Fresh session ({secondary}, no context).\n\n"
        f"=== Deep ({primary}) Position ===\n{pos_deep}\n\n"
        f"=== Fresh ({secondary}) Position ===\n{pos_fresh}\n\n"
        f"=== Deep challenges Fresh ===\n{challenge_deep}\n\n"
        f"=== Fresh challenges Deep ===\n{challenge_fresh}\n\n"
        f"Produce a final list of ALL confirmed issues with severity.",
        model="claude-opus-4-6",
    )


# ─── Runner ──────────────────────────────────────────────────────────────────


def run_diversity_experiment(
    task_ids: list[int] | None = None,
    primary: str = "claude",
    secondary: str = "gemini",
    repeats: int = 1,
):
    tasks = TASKS if task_ids is None else [TASKS[i] for i in task_ids]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = (
        Path(__file__).parent / "results" / f"{timestamp}_diversity_{primary}-vs-{secondary}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    conditions = {
        "D_single_primary": (
            f"Single ({primary})",
            lambda t: method_single(t, primary),
        ),
        "D_single_secondary": (
            f"Single ({secondary})",
            lambda t: method_single(t, secondary),
        ),
        "A_cross_model_symmetric": (
            f"Cross-Model Symmetric ({primary} vs {secondary}, same ctx)",
            lambda t: method_cross_model_symmetric(t, primary, secondary),
        ),
        "B_same_model_asymmetric_primary": (
            f"Same-Model Asymmetric ({primary} deep vs {primary} fresh)",
            lambda t: method_same_model_asymmetric(t, primary),
        ),
        "B_same_model_asymmetric_secondary": (
            f"Same-Model Asymmetric ({secondary} deep vs {secondary} fresh)",
            lambda t: method_same_model_asymmetric(t, secondary),
        ),
        "C_cross_model_asymmetric": (
            f"Cross-Model Asymmetric ({primary} deep vs {secondary} fresh)",
            lambda t: method_cross_model_asymmetric(t, primary, secondary),
        ),
        "C_cross_model_asymmetric_reversed": (
            f"Cross-Model Asymmetric ({secondary} deep vs {primary} fresh)",
            lambda t: method_cross_model_asymmetric(t, secondary, primary),
        ),
    }

    all_results = []

    for rep in range(repeats):
        print(f"\n{'#' * 70}")
        print(f"  REPEAT {rep + 1}/{repeats}")
        print(f"{'#' * 70}")

        for task in tasks:
            print(f"\n{'=' * 60}")
            print(f"Task: {task.name} ({task.id})")
            print(f"Ground truth: {len(task.ground_truth)} issues")
            print(f"{'=' * 60}")

            for cond_id, (cond_name, cond_fn) in conditions.items():
                print(f"\n  [{cond_name}] running...", end=" ", flush=True)
                t0 = time.time()
                reset_token_tracker()

                try:
                    output = cond_fn(task)
                    elapsed = time.time() - t0
                    print(f"done ({elapsed:.0f}s)")

                    print(f"  [{cond_name}] judging...", end=" ", flush=True)
                    judgment = judge_result(task, cond_name, output)
                    print("done")

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
                        token_str = f"~{tokens['total_tokens']}tok/{tokens['calls']}calls"
                        print(
                            f"  -> {found}/{total} found, {partial} partial, {missed} missed | F1={f1:.3f} | {token_str}"
                        )
                    else:
                        found = partial = missed = 0
                        f1 = recall = precision = 0.0
                        bonus = 0
                        print("  -> judge parse error")

                    tokens = get_token_usage()
                    result = {
                        "repeat": rep + 1,
                        "task_id": task.id,
                        "task_name": task.name,
                        "condition": cond_id,
                        "condition_name": cond_name,
                        "primary_backend": primary,
                        "secondary_backend": secondary,
                        "found": found,
                        "partial": partial,
                        "missed": missed,
                        "total_gt": total,
                        "bonus_findings": bonus,
                        "recall": recall,
                        "precision": precision,
                        "f1": f1,
                        "elapsed_seconds": elapsed,
                        "token_usage": {
                            "prompt_tokens": tokens["prompt_tokens"],
                            "completion_tokens": tokens["completion_tokens"],
                            "total_tokens": tokens["total_tokens"],
                            "llm_calls": tokens["calls"],
                            "estimated": tokens["estimated"],
                        },
                    }
                    all_results.append(result)

                except Exception as e:
                    print(f"ERROR: {e}")
                    all_results.append(
                        {
                            "repeat": rep + 1,
                            "task_id": task.id,
                            "condition": cond_id,
                            "error": str(e),
                        }
                    )

    # Save results
    with open(results_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary table
    print(f"\n\n{'=' * 80}")
    print("DIVERSITY EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Condition':<45} {'Recall%':>8} {'F1':>8} {'Found':>6} {'Missed':>7} {'~Tokens':>8} {'Calls':>6}"
    )
    print("-" * 90)

    for cond_id in conditions:
        cond_results = [
            r for r in all_results if r.get("condition") == cond_id and "error" not in r
        ]
        if cond_results:
            avg_recall = sum(r["recall"] for r in cond_results) / len(cond_results)
            avg_f1 = sum(r["f1"] for r in cond_results) / len(cond_results)
            avg_found = sum(r["found"] for r in cond_results) / len(cond_results)
            avg_missed = sum(r["missed"] for r in cond_results) / len(cond_results)
            avg_tokens = sum(
                r.get("token_usage", {}).get("total_tokens", 0) for r in cond_results
            ) / len(cond_results)
            avg_calls = sum(
                r.get("token_usage", {}).get("llm_calls", 0) for r in cond_results
            ) / len(cond_results)
            name = conditions[cond_id][0][:44]
            print(
                f"{name:<45} {avg_recall * 100:>7.1f}% {avg_f1:>8.3f} {avg_found:>6.1f} {avg_missed:>7.1f} {avg_tokens:>8.0f} {avg_calls:>6.1f}"
            )

    print(f"\nResults saved: {results_dir}")
    print("\nHypothesis check:")
    print("  H1: B (same-model asymmetric) >= A (cross-model symmetric)?")
    print("  H2: C (cross-model asymmetric) >= B? Marginal gain?")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 5: Species vs Cultural Diversity")
    parser.add_argument("--tasks", type=str, help="Task indices, e.g., 0,1,2")
    parser.add_argument(
        "--primary",
        type=str,
        default="claude",
        choices=list(BACKENDS.keys()),
        help="Primary model backend",
    )
    parser.add_argument(
        "--secondary",
        type=str,
        default="gemini",
        choices=list(BACKENDS.keys()),
        help="Secondary model backend",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repetitions")

    args = parser.parse_args()
    task_ids = [int(x) for x in args.tasks.split(",")] if args.tasks else None

    run_diversity_experiment(
        task_ids=task_ids,
        primary=args.primary,
        secondary=args.secondary,
        repeats=args.repeats,
    )
