#!/usr/bin/env bash
# Refresh DATA_SOURCES.md numbers from the canonical experiments/results/ tree.
#
# As of 2026-05-28 consolidation, ALL cells from prior scattered sources
# (experiments/src/results/, worktree/.../src/results/) have been rsynced
# into experiments/results/. This script no longer scans the other paths
# (they are kept on disk as backups but the canonical store is the source
# of truth).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 - <<'PY'
import json, glob, os, datetime
from collections import defaultdict

CANONICAL = "experiments/results"
SUPPLEMENTAL = {
    "experiments/cells (sweep specs)": "experiments/cells",
    "experiments/logs (execution logs)": "experiments/logs",
    "experiments/data/processed (analyses)": "experiments/data/processed",
}

print("="*100)
print(f"{'source':<60} {'dirs':>5} {'cells':>6} {'oldest':>12} {'newest':>12}")
print("="*100)

# Canonical scan
cells = 0; dirs = 0; mtimes = []
by_model = defaultdict(int)
by_ploidy_level = defaultdict(int)
for d in os.listdir(CANONICAL):
    full = os.path.join(CANONICAL, d)
    if not os.path.isdir(full): continue
    dirs += 1
    for fp in glob.glob(f"{full}/*.json"):
        if "summary" in fp or "secondary" in fp: continue
        try:
            j = json.load(open(fp))
            if "error" not in j:
                cells += 1
                mtimes.append(os.path.getmtime(fp))
                mod = j.get("model") or j.get("subject_model") or "?"
                by_model[mod] += 1
                m = j.get("method","?")
                dn, fn = j.get("deep_n",1), j.get("fresh_n",1)
                if m == "ploidy" and mod == "claude-opus-4-7":
                    by_ploidy_level[(dn, fn)] += 1
        except: pass

old = datetime.datetime.fromtimestamp(min(mtimes)).strftime("%Y-%m-%d") if mtimes else "—"
new = datetime.datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d") if mtimes else "—"
print(f"  {CANONICAL:<58} {dirs:>5} {cells:>6} {old:>12} {new:>12}")

# Supplemental dirs
for label, path in SUPPLEMENTAL.items():
    if not os.path.isdir(path):
        print(f"  {label:<58} (none)")
        continue
    files = [f for f in glob.glob(f"{path}/**/*", recursive=True) if os.path.isfile(f)]
    if not files: continue
    ts = [os.path.getmtime(f) for f in files]
    old = datetime.datetime.fromtimestamp(min(ts)).strftime("%Y-%m-%d")
    new = datetime.datetime.fromtimestamp(max(ts)).strftime("%Y-%m-%d")
    print(f"  {label:<58} {'—':>5} {len(files):>6} {old:>12} {new:>12}")

print()
print(f"TOTAL VALID CELLS IN CANONICAL: {cells}")

print()
print("=== Cells by model ===")
for mod, n in sorted(by_model.items(), key=lambda x: -x[1]):
    print(f"  {mod:<30} {n}")

print()
print("=== Ploidy cells on opus-4-7 by (deep_n, fresh_n) ===")
for k in sorted(by_ploidy_level.keys()):
    print(f"  {str(k):<15} {by_ploidy_level[k]}")

print()
print("See experiments/DATA_SOURCES.md for inventory + aggregate paired comparisons.")
PY
