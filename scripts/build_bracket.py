"""
build_bracket.py
CLI entry point for the team-level bracket generator (Step 3).

Usage
-----
  python3 scripts/build_bracket.py [mode]

  mode : conservative | balanced (default) | upset_heavy

Output
------
  Terminal  — round-by-round bracket, upset log, champion + reasoning
  File      — data/processed/generated_bracket.json
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.team_selector import OUTPUT_FILE, simulate_bracket

W    = 70
SEP  = "=" * W
THIN = "─" * W


# ── Formatting helpers ────────────────────────────────────────────────────────

def _team_str(team: dict, width: int = 22) -> str:
    return f"{team['name'][:width]:<{width}} (seed {team['seed']:>2}, rtg {team['rating']:.1f})"


def _upset_tag(is_upset: bool) -> str:
    return "  *** UPSET ***" if is_upset else ""


# ── Section printers ─────────────────────────────────────────────────────────

def print_round(title: str, games) -> None:
    """Print all games in one round."""
    if isinstance(games, dict):
        games = [games]     # Championship is stored as a single dict

    print(f"\n{'─'*W}")
    print(f"  {title}  ({len(games)} game{'s' if len(games)>1 else ''})")
    print(f"{'─'*W}")

    for g in games:
        w, l = g["winner"], g["loser"]
        tag  = _upset_tag(g["is_upset"])
        region_tag = f"  [{g['region']}]" if g["region"] != "National" else ""
        print(f"  W: {_team_str(w)}{region_tag}")
        print(f"  L: {_team_str(l)}{tag}")
        print()


def print_champion(bracket: dict) -> None:
    champ  = bracket["champion"]
    reason = bracket["reasoning"]

    print(f"\n{'═'*W}")
    print(f"  🏆  CHAMPION")
    print(f"{'═'*W}")
    print(f"  {_team_str(champ)}")
    print()
    print(f"  {reason['champion']}")
    print()

    if reason["notable_upsets"] and reason["notable_upsets"] != ["No double-digit seed upsets."]:
        print(f"  Notable double-digit upsets ({len(reason['notable_upsets'])}):")
        for u in reason["notable_upsets"]:
            print(f"    • {u}")
    else:
        print("  No double-digit seed upsets in this bracket.")


def print_upset_summary(bracket: dict) -> None:
    all_games = (
        bracket["round_of_64"] + bracket["round_of_32"] +
        bracket["sweet_16"]    + bracket["elite_8"]     +
        bracket["final_four"]  + [bracket["championship"]]
    )
    upsets = [g for g in all_games if g["is_upset"]]

    print(f"\n{THIN}")
    print(f"  UPSET LOG  ({len(upsets)} total)")
    print(THIN)
    for g in upsets:
        w, l = g["winner"], g["loser"]
        print(
            f"  {g['round']:<16}  "
            f"{w['name']:<22} (S{w['seed']:>2})  over  "
            f"{l['name']:<22} (S{l['seed']:>2})"
        )


def print_s16_notes(bracket: dict) -> None:
    notes = bracket.get("s16_constraint_notes", [])
    if notes:
        print(f"\n{THIN}")
        print(f"  SWEET 16 STRUCTURE ADJUSTMENTS")
        print(THIN)
        for note in notes:
            print(f"  • {note}")


def print_compliance(check: dict) -> None:
    print(f"\n{THIN}")
    print(f"  STRUCTURE COMPLIANCE  (mode: {check['mode']})")
    print(THIN)

    dd = check["double_digit_in_sweet16"]
    status = "✓" if dd["ok"] else "✗"
    print(f"  {status} Double-digit seeds in Sweet 16: "
          f"{dd['actual']} (target: {dd['target']})")

    print(f"\n  Upset counts (informational):")
    for rname, c in check["upset_compliance"].items():
        print(f"  ~  {rname:<18}  actual {c['actual']:>2}  /  historical target {c['target']:>2}")

    overall = "✓  FULLY COMPLIANT" if check["fully_compliant"] else "✗  PARTIALLY COMPLIANT"
    print(f"\n  Overall: {overall}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    valid = {"conservative", "balanced", "upset_heavy"}
    if mode not in valid:
        sys.exit(f"Invalid mode '{mode}'.  Choose from: {', '.join(sorted(valid))}")

    print(SEP)
    print(f"{'MARCH MADNESS  —  BRACKET GENERATOR  (STEP 3)':^{W}}")
    print(f"{'Mode: ' + mode.upper():^{W}}")
    print(SEP)

    # ── Simulate ──────────────────────────────────────────────────────────
    bracket = simulate_bracket(mode)

    # ── Print bracket round by round ──────────────────────────────────────
    rounds = [
        ("ROUND OF 64",   bracket["round_of_64"]),
        ("ROUND OF 32",   bracket["round_of_32"]),
        ("SWEET 16",      bracket["sweet_16"]),
        ("ELITE 8",       bracket["elite_8"]),
        ("FINAL FOUR",    bracket["final_four"]),
        ("CHAMPIONSHIP",  bracket["championship"]),
    ]
    for title, games in rounds:
        print_round(title, games)

    # ── Upset log ─────────────────────────────────────────────────────────
    print_upset_summary(bracket)

    # ── Sweet 16 structure notes ──────────────────────────────────────────
    print_s16_notes(bracket)

    # ── Champion + reasoning ──────────────────────────────────────────────
    print_champion(bracket)

    # ── Compliance ────────────────────────────────────────────────────────
    print_compliance(bracket["structure_check"])

    # ── Save JSON ─────────────────────────────────────────────────────────
    # Strip internal score field before saving
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k != "score"}
        if isinstance(obj, list):
            return [_clean(i) for i in obj]
        return obj

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(_clean(bracket), f, indent=2)

    print(f"\n{SEP}")
    print(f"  Saved → {OUTPUT_FILE}")
    print(SEP)


if __name__ == "__main__":
    main()
