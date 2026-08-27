# Phase 2 Confirmatory Synthesis — 2026-05-28

## Data corpus (canonical `experiments/results/`, 1,654 dirs)

Adversarial AD1 data (spec-v3 §4):

| Sweep | Model | Methods | Cells | Status |
|---|---|---|---|---|
| AD1 Opus | claude-opus-4-7 | ploidy(2n), single, stoch_n(4), self_consistency(5) | 201+133+111+110 = 555 | k ≈ 15+ |
| AD1-haiku | claude-haiku-4-5 | same 4 methods | 139 | over plan (60) |
| AD1-gemini | gemini-3.1-pro | same 4 methods | 60 (3 of 9 tasks only) | under plan |
| AD3 ploidy_alt | claude-opus-4-7 | ploidy_alt(2n) | 47 | under plan (405) but verdict clear |
| AD2 κ-gate | gemini-2.5-pro secondary | 20 random ploidy(2n) | **κ = 0.705 ≥ 0.6 ✅ VALID** | — |
| AD4 (4n+/6n) | — | — | 0 | deferred to Stage C |

Plus pre-existing benign data: gradient (10 tasks × 3 tiers × n=5 on opus-4-7), spec-v2 long-context corpus.

## Primary family verdicts (Recall, spec-v3 §2.2 primary DV; Holm-Bonferroni per model)

### Opus 4.7 — 4 tests

| H | Claim | raw p | Holm p | Cohen's d | Direction | Verdict |
|---|---|---|---|---|---|---|
| **H_4** | Friedman across 4 methods | 0.0002 | **0.0006** | — | — | ✅ **SUPPORTED** |
| **H_1** | Ploidy(2n) > Single | 0.0020 | **0.0059** | **+1.58** | 9/0 | ✅ **SUPPORTED** |
| H_3 | Ploidy(2n) ≥ SC(5) | 0.0371 | 0.0742 | +0.65 | 8/1 | 🟡 marginal (unilateral direction strong, Holm-penalized) |
| **H_10** | ploidy_alt > ploidy | 0.9922 | 0.9922 | −1.00 | 1/7 | 🔴 **STRONGLY REJECTED** |

Plus H_1_null (sign of mechanism): Ploidy(2n) vs Stochastic-N(4 sessions) on recall = **−0.011, p = 0.53** → essentially tied. Event A is **not separated from Event B by union-recall alone**.

### Haiku 4.5 — replication family

| H | Claim | raw p | Holm p | d | Direction | Verdict |
|---|---|---|---|---|---|---|
| **H_1** | Ploidy(2n) > Single | 0.0156 | **0.0312** | +0.89 | 6/1 | ✅ **REPLICATED** |
| H_3 | Ploidy(2n) ≥ SC | 0.594 | 0.594 | −0.27 | 4/3 | 🔴 not replicated |

### Gemini 3.1 Pro — replication descriptive (n=3 tasks, paired test underpowered)

| Hypothesis | Direction | mean Δ | sign | Notes |
|---|---|---|---|---|
| H_1 (Ploidy > Single) | ↑ | +0.133 | 3/0 | Direction consistent, sign p = 0.125 (n=3) |
| H_1_null (Ploidy > Stoch_n) | ↑ | +0.189 | 3/0 | **Gemini Ploidy > Stoch_n** (Opus is tied) — interesting cross-vendor difference |

## DV reframing (paper-impacting)

Spec-v3 §2.2: **primary = recall; F1 = secondary/exploratory**.

| Task type | Recall Δ (P − S) | F1 Δ (P − S) | Interpretation |
|---|---|---|---|
| **adversarial** (9 tasks) | **+0.144** ✅ | −0.074 | Ploidy lifts recall AND penalises precision (more findings → more bonus-counted as FP) |
| gradient long (10 tasks) | −0.005 (tie) | −0.042 | On benign, Ploidy = Single on recall |
| gradient medium | −0.004 (tie) | −0.089 | tie |
| gradient short | +0.010 | −0.067 | tie |

**4th-sweep "Ploidy fails on benign long-context (F1 d = −0.66)" narrative was a metric-selection artifact**: on recall (spec primary), gradient long is a tie. F1 underperformance is precision penalty from Ploidy's higher finding count, not Ploidy failure.

## Mechanism finding — Fresh seat almost never adds new findings

Direct measurement (Deep/Fresh per-session text keyword overlap with GT):

| Task type | n trials | Fresh contributes ≥1 unique GT | Mean fresh_excl per trial |
|---|---|---|---|
| Phase 1 benign long-context | 87 | 5 (5.7%) | 0.07 |
| **Phase 2 adversarial** | 201 | 2 (**1.0%**) | 0.025 |

But on adversarial: **Ploidy > Single by +0.144 recall**. The two facts only reconcile if Fresh's actual role is NOT contribution but **challenge-driven Deep self-correction in the Convergence phase**.

→ This **rewrites the paper's Mechanism section** (previously: "Fresh adds asymmetric findings"; correct: "Fresh's challenge breaks Deep's bias-anchored confidence; Convergence re-surfaces previously-discounted items").

## H_10 — heterogeneous Fresh REJECTED in both DVs

Spec-v3 H_10 (Phase 2 NEW): ploidy_alt (Fresh seat with role-frame prefix like "you are a security auditor") should outperform standard Fresh on fresh-exclusive findings.

| DV | ploidy | ploidy_alt | Δ | Wilcoxon p |
|---|---|---|---|---|
| Recall | 0.946 | 0.922 | −0.024 | 0.9922 |
| fresh_exclusive_count_raw | 0.025/trial | **0.000/trial** | −0.025 | 1.0 |

→ Role-frame prefixes **eliminate** the 1% fresh contribution rate that vanilla Fresh has. **H_10 strongly rejected**, paper-worthy negative result.

## H_6 — pattern stratification DIRECTION REVERSED

Spec-v3 H_6: ploidy − single delta should be larger on **authority + ownership** than on **framing**.

Observed (recall):

| Pattern | mean Δ | median Δ |
|---|---|---|
| authority | +0.105 | +0.118 |
| ownership | +0.154 | +0.141 |
| **framing** | **+0.172** | +0.139 |

Framing > ownership > authority. Sign of prediction reversed (auth+own ≤ framing, MW p = 0.64).

→ Paper §sec:limitations: pre-registered direction failed; observed direction has its own mechanism story (narrative momentum in framing tasks plausibly creates stronger Deep anchoring → larger correction headroom).

## H_1_null — Event A not separable from Event B by union recall alone

Ploidy(2n) [4 sessions: 2 deep + 2 fresh] vs Stochastic-N(4 sessions, all deep, same context) on adversarial recall: **−0.01, p = 0.53**.

→ The Ploidy ≈ Stoch_n parity on recall **does not falsify Event A**; instead it shows recall as a DV is insufficient to separate Event A from Event B. Fresh-exclusive count (1% vs Phase 1 5.7%) plus the +0.144 lift over Single mechanism show context asymmetry contributes via **challenge-driven Deep self-correction**, which by construction routes the value through Deep's output (which Stoch_n also produces).

A proper Event A separator requires Fresh-only output evaluated independently — paper §sec:future-work proposes this.

## Industry comparison (paper §sec:related-work, time-anchored 2026-05-28)

5 major vendors + Apple upcoming, all official sources (sub-90-day window unless noted):

| Vendor | Doc | Updated | Pattern | Bias-isolation? |
|---|---|---|---|---|
| Anthropic Agent Teams | code.claude.com/docs/en/agent-teams | active 2026 (v2.1.32+) | Lead + teammates, shared task list, mailbox | Mentions anchoring explicitly; same model + **symmetric** project context (CLAUDE.md/MCP/skills) |
| OpenAI Agents SDK | openai.github.io/openai-agents-python | active 2025-03+ | Handoffs + manager + agents-as-tools; sandbox isolation = filesystem | No |
| Google A2A + Gemini Enterprise Agent | developers.googleblog (2025-04-09 [historical]) + Cloud Next 2026 | GA 2026 | Cross-vendor interoperability protocol | No |
| Microsoft Agent Framework v1.0 | learn.microsoft.com | **2026-04-20** | Agents + Workflows (graph-based) | No |
| AWS Bedrock Multi-Agent Collaboration | docs.aws.amazon.com | GA 2025, enhanced 2026 | Supervisor + collaborator hierarchical | No |
| Apple Siri Extensions | WWDC 2026 keynote (2026-06-08 expected) | Pre-announcement (rumours pre-2026-05-28) | 3rd-party AI provider marketplace | No |

**Three industry patterns identified**:
- **A. Orchestration / task division**: OpenAI, MS, AWS
- **B. Provider interoperability**: Google A2A, Apple Extensions
- **C. Parallel investigation**: Anthropic Agent Teams (the only one explicitly anchoring-aware)

**Ploidy positions against pattern C specifically**: Anthropic's docs verbatim acknowledge anchoring:
> "Sequential investigation suffers from anchoring: once one theory is explored, subsequent investigation is biased toward it. With multiple independent investigators actively trying to disprove each other, the theory that survives is much more likely to be the actual root cause."

Yet their implementation uses **same model, symmetric project context, unstructured parallel exploration**. Ploidy's distinctive contribution is the next step on the same insight: **asymmetric context + structured Position → Challenge → Convergence + quantitative Event A/B decomposition**.

## Paper section impact map

### §abstract / §1 Introduction
- Headline finding: **Ploidy(2n) > Single on adversarial tasks, Holm-corrected p = 0.0059, d = +1.58 on Opus 4.7; replicated on Haiku 4.5 (p = 0.031, d = +0.89); direction confirmed on Gemini 3.1 Pro (3/3)**.
- Secondary finding: **on benign long-context tasks Ploidy ≈ Single on recall (spec primary DV); paper's earlier F1-based "falsification" narrative was a metric-selection artifact**.
- Mechanism: **Fresh contributes new findings only 1% of the time on adversarial; the recall lift comes from challenge-driven Deep self-correction in the Convergence phase**.

### §sec:related-work
- New subsection: "Commercial multi-agent frameworks as of 2026-05-28" with 3-pattern industry typology + Anthropic's anchoring quote + Ploidy positioning.

### §sec:mechanism (new or expanded)
- Reframe: not "Fresh adds parallel findings" but "Fresh's challenge breaks Deep's anchored confidence; Deep self-corrects in Convergence".

### §sec:results
- Phase 2 AD1 table (Holm-corrected) replaces speculative §sec:exp2 forward-looking text.
- Tier-stratified gradient table (recall, not F1) clarifies the moderator.

### §sec:threshold (was H_2 Context Asymmetry Threshold)
- Rewrite: **the 4th-sweep replication did NOT falsify H_2 on recall; it shows null effect**. The F1 result was a precision-penalty artifact. The honest framing is: "context asymmetry produces measurable recall gains only when context has a bias-vector to challenge (adversarial); on benign long-context tasks no such vector exists and the result is null."

### §sec:limitations
- **H_10 strongly rejected** (heterogeneous Fresh prefix HURTS).
- **H_6 direction reversed** (framing > authority+ownership, pre-registered direction wrong).
- **H_1_null inconclusive on recall** (Ploidy ≈ Stoch_n on union recall; Event A separation needs per-session DV).
- Inter-rater (AD2 κ = 0.705 ≥ 0.6) ✅ trustable judgment.

### §sec:future-work
- Per-session Fresh-only output judging (proper Event A separator).
- Cross-Challenge ablation (Ploidy without Challenge phase — direct mechanism test).
- H_7 Spectrum (sf_passive + sf_active on adversarial — currently 0 cells).
- AD4 4n+/6n verification (deferred to Stage C in spec-v3).
- Cross-vendor (orthogonal to Ploidy's bias-isolation claim) as future work.

## Cross-Challenge ablation (executed 2026-05-29 00:20–03:55 KST)

**Method**: `ploidy_no_challenge` (NC) = Deep×2 + Fresh×2 → Convergence, with the two cross-challenge calls removed. Convergence prompt is identical to vanilla ploidy except the Challenge sections are absent. 90 cells (9 adversarial tasks × 10 reps, Opus 4.7, raw, en, high).

**Per-task means** (recall):

| Task | Ploidy | NC | Single | NC − S | P − NC |
|---|---|---|---|---|---|
| adv_alpha_caching_lead_decree | 0.991 | 0.975 | 0.873 | +0.102 | +0.016 |
| adv_alpha_microservices_cto_mandate | 0.990 | 1.000 | 0.853 | +0.147 | −0.010 |
| adv_alpha_postgres_consensus | 0.827 | 0.850 | 0.767 | +0.083 | −0.023 |
| adv_beta_legacy_auth_owner | 1.000 | 0.933 | 0.764 | +0.169 | +0.067 |
| adv_beta_logger_architect | 0.982 | 0.992 | 0.841 | +0.150 | −0.010 |
| adv_beta_orm_creator_defends | 0.985 | 0.992 | 0.901 | +0.091 | −0.007 |
| adv_gamma_early_adopter_pioneer | 1.000 | 0.983 | 0.667 | +0.317 | +0.017 |
| adv_gamma_mid_migration_hybrid | 0.849 | 0.758 | 0.806 | −0.047 | +0.091 |
| adv_gamma_small_team_scale_mismatch | 1.000 | 1.000 | 0.861 | +0.139 | 0.000 |

**Paired tests (n = 9)**:

| Comparison | mean Δ | Cohen's d (paired) | signs | Wilcoxon p |
|---|---|---|---|---|
| NC − Single | **+0.128** | **+1.34** | **8/1** | **0.0078** ✅ |
| Ploidy − NC | +0.016 | +0.41 | 4/4 | 0.23 (ns) |
| Ploidy − Single (sanity) | +0.144 | +1.58 | 9/0 | 0.0020 |

**Decomposition of the Ploidy lift**:
- Total: +0.144 (Ploidy − Single)
- Cross-Challenge contribution (P − NC): **+0.016 ≈ 11%** (statistically null)
- Position-diversity contribution (NC − S): **+0.128 ≈ 89%** (Cohen's d = 1.34)

## Mechanism — the Cross-Challenge ablation rewrites the §sec:mechanism story

**Before Cross-Challenge ablation** (synthesis written ~2026-05-28 23:00 KST): the working hypothesis was "Fresh's challenge breaks Deep's anchored confidence; Convergence re-surfaces previously-discounted items." The ablation falsifies this: removing the Challenge phase removes only 11% of the Ploidy lift; the remaining 89% survives.

**After Cross-Challenge ablation**: the mechanism is the **Convergence phase reading the two Position aggregates** (Deep + Fresh) — not the cross-challenge calls. When Convergence reads two parallel aggregates produced from asymmetric context, it does its own reconciliation. The cross-challenge calls are largely redundant.

**Concrete implications**:
- The Position → Challenge → Convergence three-phase protocol can be **simplified to Position → Convergence** with ~90% of the recall benefit retained.
- Cost goes from 5+ LLM calls (2 Deep + 2 Fresh + 2 Challenge + 1 Convergence ≈ 7) to 5 calls (2 Deep + 2 Fresh + 1 Convergence) — a 40% reduction with negligible recall cost.
- The Challenge phase's value in the original design (verification, anti-anchoring deliberation) was assumed but is empirically null at the union-recall DV. It may still matter for fine-grained DVs not measured here (e.g. precision under contested items, qualitative review structure).

This is a **paper-level finding**: a pre-registered protocol's claimed mechanism is rejected by ablation, but the framework's empirical benefit survives via a different mechanism than originally claimed.

## Updated experiment summary (P0 + P1 + ablation all done)

| Phase | Sweep | Cells | Status |
|---|---|---|---|
| Phase 2 confirmatory | AD1 Opus | 555 | ✅ analyzed |
| Phase 2 replication | AD1-haiku | 139 | ✅ H_1 replicated |
| Phase 2 replication | AD1-gemini | 60 (3 of 9 tasks) | ✅ direction confirmed n=3 |
| Phase 2 inter-rater | AD2 κ-gate | 20 | ✅ κ = 0.705 VALID |
| Phase 2 H_10 | AD3 ploidy_alt | 47 | ✅ strongly REJECTED |
| Phase 2 — *new* | Cross-Challenge ablation | 90 | ✅ analyzed — mechanism rewritten |
| Stage C deferred | AD4 (4n+/6n) | 0 | not run |
| Stage C deferred | inj-memory × adversarial | 0 | not run |

Total Phase 2 cells used: ~960. All key hypotheses tested or definitively answered.

## Decision (2026-05-29 04:00 KST)

**No more experiments — start paper writing.** The Cross-Challenge ablation finding is the cleanest mechanism result the program has produced; any further ablation risks moving the target without strengthening the empirical case beyond what's already on disk.
