"""
scripts/pool_optimizer_demo.py
Demonstrate the pool-size optimizer for pool_size = 10, 50, 100.

Runs simulate_bracket(mode='balanced', season=2025) once, then applies
optimize_for_pool() at each pool size.  The bracket simulation is unchanged;
only the final champion/FF selection strategy shifts.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.team_selector  import simulate_bracket
from lib.pool_optimizer import optimize_for_pool

# ── Run bracket once ─────────────────────────────────────────────────────────
print("Simulating 2025 bracket (balanced mode)…")
bracket = simulate_bracket(mode="balanced", season=2025)

model_champ = bracket["champion"]
model_ff_games = bracket.get("final_four", [])
model_ff = []
seen = set()
for g in model_ff_games:
    for key in ("winner", "loser"):
        t = g.get(key, {})
        if t and t.get("name") and t["name"] not in seen:
            seen.add(t["name"])
            model_ff.append(t)

W = 70
print()
print("=" * W)
print("MODEL BASELINE (pool-agnostic)".center(W))
print("=" * W)
print(f"  Champion  : {model_champ['name']} (seed {model_champ['seed']})")
print(f"  Final Four: " + "  ".join(
    f"{t['name']}(s{t['seed']})" for t in model_ff
))
if bracket.get("champion_win_prob") is not None:
    print(f"  Champ win prob  : {bracket['champion_win_prob']:.3f}")
    print(f"  Champ public pct: {bracket['champion_public_pct']:.3f}")
    print(f"  Champ value_score: {bracket['champion_value_score']:+.4f}")

# ── Optimize for each pool size ──────────────────────────────────────────────
for pool_size in [10, 50, 100]:
    result = optimize_for_pool(bracket, pool_size)
    champ  = result["champion"]
    ff     = result["final_four"]

    changed_marker = "  ← FLIPPED FROM MODEL" if result["champion_changed"] else ""

    print()
    print("=" * W)
    print(f"  POOL SIZE = {pool_size}  [{result['strategy'].upper()} POOL]".center(W))
    print("=" * W)
    print()
    print(f"  Strategy")
    print(f"  {'─'*66}")
    # Wrap description at 66 chars
    desc = result["strategy_description"]
    words, line = [], ""
    for word in desc.split():
        if len(line) + len(word) + 1 <= 66:
            line = (line + " " + word).strip()
        else:
            print(f"  {line}")
            line = word
    if line:
        print(f"  {line}")

    print()
    print(f"  Champion{changed_marker}")
    print(f"  {'─'*66}")
    print(f"  {champ['name']} (seed {champ['seed']})")
    # Wrap rationale
    rat = result["champion_rationale"]
    words, line = [], ""
    for word in rat.split():
        if len(line) + len(word) + 1 <= 66:
            line = (line + " " + word).strip()
        else:
            print(f"  {line}")
            line = word
    if line:
        print(f"  {line}")

    print()
    print(f"  Final Four  (sorted by pool-adjusted score ↓)")
    print(f"  {'─'*66}")
    print(f"  {'Team':<14} {'Seed':>4}  {'PoolScore':>9}  {'Quality':>7}  {'Leverage':>8}  {'Ownership'}")
    for p in ff:
        print(
            f"  {p['name']:<14} {p['seed']:>4}  "
            f"{p['pool_score']:>9.4f}  {p['quality']:>7.4f}  "
            f"{p['leverage']:>8.4f}  {p['ownership_label']}"
        )

    print()
    print(f"  Final Four Rationale")
    print(f"  {'─'*66}")
    rat2 = result["final_four_rationale"]
    words, line = [], ""
    for word in rat2.split():
        if len(line) + len(word) + 1 <= 66:
            line = (line + " " + word).strip()
        else:
            print(f"  {line}")
            line = word
    if line:
        print(f"  {line}")

print()
print("=" * W)
