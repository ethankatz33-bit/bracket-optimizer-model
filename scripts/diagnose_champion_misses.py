"""
scripts/diagnose_champion_misses.py

For every backtest year where predicted champion != actual champion, determine
exactly where the actual champion was eliminated in the PREDICTED bracket.

Elimination buckets
-------------------
  R64_or_R32  : actual champ lost in Round of 64 or Round of 32
  S16         : actual champ lost in Sweet 16
  E8          : actual champ lost in Elite 8
  FF          : actual champ reached Final Four but lost there
  CHAMP_GAME  : actual champ reached Championship game but lost (runner-up)

Output
------
  Per-year table (wrong-champion years only)
  Summary counts and percentages
  Actionable verdict on where to focus fixes
"""
import io
import sys
import contextlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.backtest import run_backtest


@contextlib.contextmanager
def _silent():
    """Suppress stdout during backtest runs (suppresses selector/AP diagnostics)."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old

# ── Helpers ───────────────────────────────────────────────────────────────────

BRACKET_ROUNDS_IN_ORDER = [
    ("round_of_64",  "R64"),
    ("round_of_32",  "R32"),
    ("sweet_16",     "S16"),
    ("elite_8",      "E8"),
    ("final_four",   "FF"),
    ("championship", "CHAMP_GAME"),
]


def _find_elimination_round(bracket: dict, team_name: str) -> str:
    """
    Walk the predicted bracket and return the label of the round in which
    team_name appears as a LOSER.  Returns 'CHAMPION' if they won every game
    (shouldn't happen in wrong-champion years), or 'NOT_FOUND' if the team
    never appeared (eliminated before R64 — play-in loss or not in field).
    """
    for key, label in BRACKET_ROUNDS_IN_ORDER:
        games = bracket.get(key)
        if games is None:
            continue
        # championship is a single dict, all others are lists
        if isinstance(games, dict):
            games = [games]
        for game in games:
            if game.get("loser", {}).get("name") == team_name:
                return label
    # Check if they're the champion
    if bracket.get("champion", {}).get("name") == team_name:
        return "CHAMPION"
    return "NOT_FOUND"


def _was_in_predicted_ff(r: dict, act_name: str) -> bool:
    return any(t["name"] == act_name for t in r["predicted_final_four"])


def _was_in_predicted_championship(bracket: dict, act_name: str) -> bool:
    champ_game = bracket.get("championship", {})
    winner = champ_game.get("winner", {}).get("name")
    loser  = champ_game.get("loser",  {}).get("name")
    return act_name in (winner, loser)


# ── Run backtest for all years ────────────────────────────────────────────────

YEARS = [y for y in range(1990, 2026) if y != 2020]
W = 72

print("=" * W)
print("CHAMPION-MISS DIAGNOSTIC  (1990–2025, balanced mode)".center(W))
print("=" * W)
print()

rows = []
errors = []

for year in YEARS:
    try:
        with _silent():
            r = run_backtest(year, mode="balanced")
    except Exception as e:
        errors.append((year, str(e)))
        continue

    pred_name = r["predicted_champion"]["name"]
    pred_seed = r["predicted_champion"]["seed"]
    act_name  = r["actual_champion"]["name"]
    act_seed  = r["actual_champion"]["seed"]

    if pred_name == act_name:
        rows.append({
            "year":     year,
            "correct":  True,
            "act_name": act_name,
            "act_seed": act_seed,
            "pred_name": pred_name,
            "pred_seed": pred_seed,
            "elim_round": "CHAMPION",
            "in_pred_ff": True,
            "in_pred_champ_game": True,
        })
        continue

    bracket  = r["_predicted_full"]
    elim     = _find_elimination_round(bracket, act_name)
    in_ff    = _was_in_predicted_ff(r, act_name)
    in_champ = _was_in_predicted_championship(bracket, act_name)

    rows.append({
        "year":       year,
        "correct":    False,
        "act_name":   act_name,
        "act_seed":   act_seed,
        "pred_name":  pred_name,
        "pred_seed":  pred_seed,
        "elim_round": elim,
        "in_pred_ff": in_ff,
        "in_pred_champ_game": in_champ,
    })

# ── Per-year table (wrong-champion years only) ────────────────────────────────

wrong = [r for r in rows if not r["correct"]]
correct = [r for r in rows if r["correct"]]

print(f"{'Year':>4}  {'Act champion':<20} {'s':>2}  {'Pred champion':<20} {'s':>2}  "
      f"{'Eliminated in':<14}  {'In pred FF?'}")
print("  " + "─" * (W - 2))

for r in sorted(wrong, key=lambda x: x["year"]):
    in_ff_str = "YES" if r["in_pred_ff"] else "no"
    print(f"  {r['year']:>4}  {r['act_name']:<20} {r['act_seed']:>2}  "
          f"{r['pred_name']:<20} {r['pred_seed']:>2}  "
          f"{r['elim_round']:<14}  {in_ff_str}")

# ── Summary ───────────────────────────────────────────────────────────────────

print()
print("=" * W)
print("SUMMARY".center(W))
print("=" * W)

total     = len(rows)
n_correct = len(correct)
n_wrong   = len(wrong)

# Bucket by elimination round
buckets = {
    "CHAMP_GAME":  [],
    "FF":          [],
    "E8":          [],
    "S16":         [],
    "R64_or_R32":  [],
    "NOT_FOUND":   [],
}
for r in wrong:
    e = r["elim_round"]
    if e in ("R64", "R32"):
        buckets["R64_or_R32"].append(r)
    elif e in buckets:
        buckets[e].append(r)
    else:
        buckets["NOT_FOUND"].append(r)

print(f"\n  Total seasons        : {total}")
print(f"  Correctly predicted  : {n_correct}  ({n_correct/total:.1%})")
print(f"  Wrong champion       : {n_wrong}  ({n_wrong/total:.1%})")

print(f"\n  Where actual champion was eliminated (wrong-champion years):")
print(f"  {'Bucket':<22}  {'Count':>5}  {'% of misses':>11}  Years")
print("  " + "─" * (W - 4))

for label, yrs in [
    ("Champ game (runner-up)", buckets["CHAMP_GAME"]),
    ("Final Four",             buckets["FF"]),
    ("Elite 8",                buckets["E8"]),
    ("Sweet 16",               buckets["S16"]),
    ("R64 or R32",             buckets["R64_or_R32"]),
    ("Not found",              buckets["NOT_FOUND"]),
]:
    pct = len(yrs) / n_wrong if n_wrong else 0
    yrs_str = ", ".join(str(r["year"]) for r in sorted(yrs, key=lambda x: x["year"]))
    print(f"  {label:<22}  {len(yrs):>5}  {pct:>10.1%}  {yrs_str}")

# ── FF-or-later breakdown (late-round vs early-round problem) ─────────────────
late  = len(buckets["CHAMP_GAME"]) + len(buckets["FF"])
early = n_wrong - late

print()
print("─" * W)
print(f"  Late-round miss  (champ reached FF but lost FF or CHAMP): "
      f"{late:>3}  ({late/n_wrong:.1%})")
print(f"  Early-round miss (champ eliminated before FF)            : "
      f"{early:>3}  ({early/n_wrong:.1%})")
print()

if late >= early:
    verdict = (
        "VERDICT: Late-round selection is the dominant problem.\n"
        "  The actual champion reaches the model's Final Four in most miss years.\n"
        "  Focus: improve FF/Championship selection logic."
    )
else:
    verdict = (
        "VERDICT: Early-round bracket routing is the dominant problem.\n"
        "  The actual champion is eliminated before the Final Four in most miss years.\n"
        "  Focus: improve E8/earlier-round seed or rating logic."
    )
print("  " + verdict.replace("\n", "\n  "))

# ── Detail: late-round misses ─────────────────────────────────────────────────
if late > 0:
    print()
    print("─" * W)
    print("  LATE-ROUND MISSES — actual champion was in predicted FF")
    print()
    print(f"  {'Year':>4}  {'Act champ':<20} {'s':>2}  {'Pred champ':<20} {'s':>2}  "
          f"{'Eliminated'}")
    print("  " + "─" * 58)
    for r in sorted(buckets["CHAMP_GAME"] + buckets["FF"],
                    key=lambda x: x["year"]):
        print(f"  {r['year']:>4}  {r['act_name']:<20} {r['act_seed']:>2}  "
              f"{r['pred_name']:<20} {r['pred_seed']:>2}  "
              f"{r['elim_round']}")

# ── Detail: early-round misses ────────────────────────────────────────────────
early_rows = (buckets["E8"] + buckets["S16"] +
              buckets["R64_or_R32"] + buckets["NOT_FOUND"])
if early_rows:
    print()
    print("─" * W)
    print("  EARLY-ROUND MISSES — actual champion did NOT reach predicted FF")
    print()
    print(f"  {'Year':>4}  {'Act champ':<20} {'s':>2}  {'Pred champ':<20} {'s':>2}  "
          f"{'Eliminated'}")
    print("  " + "─" * 58)
    for r in sorted(early_rows, key=lambda x: x["year"]):
        print(f"  {r['year']:>4}  {r['act_name']:<20} {r['act_seed']:>2}  "
              f"{r['pred_name']:<20} {r['pred_seed']:>2}  "
              f"{r['elim_round']}")

if errors:
    print()
    print("─" * W)
    print(f"  ERRORS ({len(errors)} years skipped):")
    for yr, msg in errors:
        print(f"    {yr}: {msg}")

print()
print("=" * W)
