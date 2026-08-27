#!/usr/bin/env bash
# B-hybrid: focused cross-model sweep — 10 long-tier tasks × 3 methods × 3 NEW models × n=3 reps
#
# Models (all FREE):
#   - claude-sonnet  (Claude Max, medium capability)
#   - claude-haiku   (Claude Max, low capability)
#   - gemini-2.5-pro (Gemini CLI, cross-family)
#
# opus-4-7 baseline already exists in canonical store (n=5 × 30 cells = 150 cells per method).
#
# Total new cells: 10 tasks × 3 methods × 3 models × 3 reps = 270
# Expected wall time: ~1.5-2h (task_parallel=4 + method_parallel=3)

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LOG_ROOT="${ROOT}/experiments/logs"
mkdir -p "$LOG_ROOT"

# Long-tier indices in tasks_gradient.py GRADIENT_TASKS (every 3rd starting at 2)
LONG_INDICES="2,5,8,11,14,17,20,23,26,29"

RUNS=3
EFFORT=high
LANG=en
INJECTION=raw
METHODS=single,ccr,ploidy

run_model() {
  local backend="$1"
  local model_id="$2"
  local label="$3"

  echo "$(date) ── B sweep start: ${label} (backend=${backend}, model=${model_id})" | tee -a "${LOG_ROOT}/B-cross-model.log"

  for i in $(seq 1 $RUNS); do
    local log="${LOG_ROOT}/B-${label}-run${i}.log"
    local dir_file="${LOG_ROOT}/B-${label}-run${i}.dir"

    echo "$(date) ── ${label} run #${i} start" | tee -a "$log"

    if [ -s "$dir_file" ]; then
      local resume_dir
      resume_dir="$(cat "$dir_file")"
      echo "$(date) — invoking with --resume ${resume_dir}" | tee -a "$log"
      PYTHONUNBUFFERED=1 PLOIDY_TASK_PARALLEL=4 PLOIDY_METHOD_PARALLEL=3 \
        PLOIDY_RESET_ANCHOR_HOUR=9 PLOIDY_RESET_ANCHOR_MIN=40 \
        python3 "${ROOT}/experiments/src/run_experiment.py" \
        --gradient --tasks "$LONG_INDICES" --methods "$METHODS" \
        --backend "$backend" --model "$model_id" \
        --effort "$EFFORT" --injection "$INJECTION" --lang "$LANG" \
        --resume "$resume_dir" >> "$log" 2>&1
    else
      echo "$(date) — ${label} first invocation" | tee -a "$log"
      PYTHONUNBUFFERED=1 PLOIDY_TASK_PARALLEL=4 PLOIDY_METHOD_PARALLEL=3 \
        PLOIDY_RESET_ANCHOR_HOUR=9 PLOIDY_RESET_ANCHOR_MIN=40 \
        python3 "${ROOT}/experiments/src/run_experiment.py" \
        --gradient --tasks "$LONG_INDICES" --methods "$METHODS" \
        --backend "$backend" --model "$model_id" \
        --effort "$EFFORT" --injection "$INJECTION" --lang "$LANG" >> "$log" 2>&1
      # Capture results dir
      local latest_dir
      latest_dir="$(ls -td "${ROOT}/experiments/results/"*"_effort-${EFFORT}_lang-${LANG}_inj-${INJECTION}" 2>/dev/null | head -1)"
      if [ -n "${latest_dir:-}" ]; then
        echo "$latest_dir" > "$dir_file"
      fi
    fi

    local rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "$(date) ── ${label} run #${i} exit=${rc} (will be retried on next launch)" | tee -a "$log"
      # do not break — continue to next rep
    else
      echo "$(date) ── ${label} run #${i} completed" | tee -a "$log"
    fi
  done

  echo "$(date) ── B sweep done: ${label}" | tee -a "${LOG_ROOT}/B-cross-model.log"
}

# Run all 3 models sequentially
# Sonnet first (fastest), haiku, then gemini
run_model claude sonnet "sonnet"
run_model claude haiku "haiku"
run_model gemini gemini-2.5-pro "gemini"

echo "$(date) ── ALL B sweep done"
