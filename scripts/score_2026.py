"""
scripts/score_2026.py
Score the 2026 bracket predictions against actual tournament results.

Scoring system (ESPN standard):
  Round of 64   =  1 pt per correct pick
  Round of 32   =  2 pts
  Sweet 16      =  4 pts
  Elite 8       =  8 pts
  Final Four    = 16 pts
  Champion      = 32 pts

Max possible score: 192 pts

Usage
-----
  python3 scripts/score_2026.py
  python3 scripts/score_2026.py --json data/processed/future_bracket_2026.json

Sources for actual results:
  Michigan wins 2026 NCAA Championship — https://www.ncaa.com/news/basketball-men/article/2026-04-06/michigan-beats-uconn-wins-2026-mens-basketball-national-championship
  Full bracket results — https://en.wikipedia.org/wiki/2026_NCAA_Division_I_men%27s_basketball_tournament
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

W   = 72
SEP = "=" * W


# ── Actual 2026 tournament results ────────────────────────────────────────────
#
# Keys: set of team names that WON each round.
# "won_r64" = teams that won their Round-of-64 game (advanced to Round of 32).
# ESPN scoring: you earn round points for correctly predicting a team advances
# through that round, regardless of opponent.
#
# Sources: NCAA.com, Wikipedia (2026 NCAA Division I men's basketball tournament)

ACTUAL = {
    # First Four (not ESPN-scored; reported separately)
    "first_four_winners": {
        "Howard", "Texas", "Prairie View A&M", "Miami (OH)",
    },

    # Round of 64 → these teams won their first main-draw game
    "won_r64": {
        # East
        "Duke", "TCU", "St. John's", "Kansas", "Louisville",
        "Michigan State", "UCLA", "Connecticut",
        # South
        "Florida", "Iowa", "Vanderbilt", "Nebraska", "VCU",
        "Illinois", "Texas A&M", "Houston",
        # West
        "Arizona", "Utah State", "High Point", "Arkansas",
        "Texas", "Gonzaga", "Miami (FL)", "Purdue",
        # Midwest
        "Michigan", "Saint Louis", "Texas Tech", "Alabama",
        "Tennessee", "Virginia", "Kentucky", "Iowa State",
    },

    # Round of 32 winners (advanced to Sweet 16)
    "won_r32": {
        # East
        "Duke", "St. John's", "Michigan State", "Connecticut",
        # South
        "Iowa", "Nebraska", "Illinois", "Houston",
        # West
        "Arizona", "Arkansas", "Texas", "Purdue",
        # Midwest
        "Michigan", "Alabama", "Tennessee", "Iowa State",
    },

    # Sweet 16 winners (advanced to Elite 8)
    "won_s16": {
        # East
        "Duke", "Connecticut",
        # South
        "Iowa", "Illinois",
        # West
        "Arizona", "Purdue",
        # Midwest
        "Michigan", "Tennessee",
    },

    # Elite 8 winners (advanced to Final Four)
    "won_e8": {
        "Connecticut",   # East  — beat Duke 73-72
        "Illinois",      # South — beat Iowa 71-59
        "Arizona",       # West  — beat Purdue 79-64
        "Michigan",      # Midwest — beat Tennessee 95-62
    },

    # Final Four winners (advanced to Championship game)
    # Semis: Michigan (Midwest) vs Illinois (South) → Michigan
    #        UConn (East) vs Arizona (West)         → UConn
    "won_ff": {
        "Michigan",
        "Connecticut",
    },

    # Champion
    "champion": "Michigan",  # beat UConn 76-73 (April 6, Lucas Oil Stadium)

    # First Four game details keyed by (region, seed)
    "first_four_games": {
        ("Midwest", 11): {"winner": "Miami (OH)",       "loser": "SMU"},
        ("Midwest", 16): {"winner": "Howard",           "loser": "UMBC"},
        ("South",   16): {"winner": "Prairie View A&M", "loser": "Lehigh"},
        ("West",    11): {"winner": "Texas",            "loser": "NC State"},
    },
}

# ESPN round → point value + set key in ACTUAL
ROUNDS: list[tuple[str, int, str]] = [
    ("round_of_64", 1,  "won_r64"),
    ("round_of_32", 2,  "won_r32"),
    ("sweet_16",    4,  "won_s16"),
    ("elite_8",     8,  "won_e8"),
    ("final_four",  16, "won_ff"),
]


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _score_bracket_path(bracket: dict) -> dict:
    """
    Score the stored bracket path against actual results.

    Returns a breakdown dict:
      {round_key: {"correct": [...], "wrong": [...], "pts": int}}
    plus "champion_path" (the bracket's predicted champion from the path).
    """
    breakdown: dict = {}

    for rnd_key, pts_each, actual_key in ROUNDS:
        actual_winners = ACTUAL[actual_key]
        games = bracket.get(rnd_key, [])
        correct: list[str] = []
        wrong:   list[str] = []

        for game in games:
            predicted_winner = game.get("winner", {}).get("name", "")
            if predicted_winner in actual_winners:
                correct.append(predicted_winner)
            else:
                wrong.append(predicted_winner)

        breakdown[rnd_key] = {
            "correct":  correct,
            "wrong":    wrong,
            "pts":      len(correct) * pts_each,
            "pts_each": pts_each,
        }

    breakdown["champion_path"] = bracket.get("champion", {}).get("name", "?")
    return breakdown


def _score_champion_pick(champion_pick: str) -> int:
    """Return 32 if correct, 0 otherwise."""
    return 32 if champion_pick == ACTUAL["champion"] else 0


def _total(breakdown: dict, champion_pick: str) -> int:
    path_pts  = sum(v["pts"] for k, v in breakdown.items() if isinstance(v, dict) and "pts" in v)
    champ_pts = _score_champion_pick(champion_pick)
    return path_pts + champ_pts


# ── Display ───────────────────────────────────────────────────────────────────

_ROUND_LABELS = {
    "round_of_64": "Round of 64",
    "round_of_32": "Round of 32",
    "sweet_16":    "Sweet 16   ",
    "elite_8":     "Elite 8    ",
    "final_four":  "Final Four ",
}

_MAX_PTS = {
    "round_of_64": 32,
    "round_of_32": 32,
    "sweet_16":    32,
    "elite_8":     32,
    "final_four":  32,
}


def _print_bracket_score(
    label:          str,
    breakdown:      dict,
    champion_pick:  str,
    champ_pts:      int,
    total_pts:      int,
) -> None:
    print()
    print(f"  ── {label} ──")
    print(f"  {'Round':<14} {'Correct':>7}  {'Wrong':>5}  {'Pts':>4}  {'Max':>4}")
    print("  " + "─" * 40)
    path_total = 0
    for rnd_key, _, _ in ROUNDS:
        info     = breakdown[rnd_key]
        n_ok     = len(info["correct"])
        n_games  = n_ok + len(info["wrong"])
        pts      = info["pts"]
        max_pts  = _MAX_PTS[rnd_key]
        path_total += pts
        print(f"  {_ROUND_LABELS[rnd_key]:<14} {n_ok:>3}/{n_games:<3}  "
              f"{'—' if not info['wrong'] else ', '.join(info['wrong'][:3]) + ('…' if len(info['wrong']) > 3 else ''):>28}  "
              f"{pts:>4}  {max_pts:>4}")

    champ_correct = "✓" if champ_pts == 32 else "✗"
    print(f"  {'Champion   ':<14} {champ_correct}  ({champion_pick})"
          f"{'  [correct!]' if champ_pts else '  [actual: ' + ACTUAL['champion'] + ']'}")
    print("  " + "─" * 40)
    print(f"  {'TOTAL':<14} {path_total + champ_pts:>4} / 192")




def _print_summary_table(results: list[dict]) -> None:
    """Print ranked summary table of all bracket types."""
    ranked = sorted(results, key=lambda r: r["total"], reverse=True)
    print()
    print(SEP)
    print("  2026 BRACKET SCORING — RANKED SUMMARY".center(W))
    print(SEP)
    print(f"\n  Actual champion: {ACTUAL['champion']}  |  Final Four: "
          + ", ".join(sorted(ACTUAL['won_e8'])))
    print()
    print(f"  {'#':<3} {'Bracket':<18} {'Champion Pick':<18} "
          f"{'R64':>4}  {'R32':>4}  {'S16':>4}  {'E8':>4}  {'FF':>4}  {'CHM':>4}  {'TOTAL':>6}  {'/ 192':>6}")
    print("  " + "─" * 74)

    for i, r in enumerate(ranked, 1):
        bd   = r["breakdown"]
        chk  = "✓" if r["champ_pts"] == 32 else "✗"
        print(
            f"  {i:<3} {r['label']:<18} {r['champion_pick'] + ' ' + chk:<18} "
            f"{bd['round_of_64']['pts']:>4}  "
            f"{bd['round_of_32']['pts']:>4}  "
            f"{bd['sweet_16']['pts']:>4}  "
            f"{bd['elite_8']['pts']:>4}  "
            f"{bd['final_four']['pts']:>4}  "
            f"{r['champ_pts']:>4}  "
            f"{r['total']:>6}  "
            f"{'(' + str(round(r['total']/192*100)) + '%)':>6}"
        )

    print()
    print(f"  Key upsets the model missed:")
    print(f"    TCU over Ohio State (R64)    — predicted Ohio State")
    print(f"    Utah State over Villanova (R64) — predicted Villanova")
    print(f"    High Point over Wisconsin (R64) — predicted Wisconsin")
    print(f"    Texas over BYU (R64)          — predicted BYU")
    print(f"    Iowa over Florida (R32)        — predicted Florida")
    print(f"    Illinois over VCU (R32)        — model had VCU upsetting Illinois in R32")
    print(f"    Texas over Gonzaga (R32)       — predicted Gonzaga")
    print(f"    Tennessee over Virginia (R32)  — predicted Virginia")
    print(f"    Michigan over Houston (FF)     — predicted Houston in Championship")
    print()
    print(f"  Model wins:")
    print(f"    Michigan champion ✓  (Deterministic + Safe correct, +32 pts each)")
    print(f"    UConn E8 + FF ✓      (predicted Connecticut reaches championship)")
    print(f"    Illinois E8 ✓        (value/contrarian pick made Final Four)")
    print(f"    Arizona E8 + FF ✓    (predicted Arizona, reached correct FF semi)")
    print()
    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Score 2026 bracket predictions vs actual.")
    parser.add_argument("--json", default=str(PROJECT_ROOT / "data/processed/future_bracket_2026.json"),
                        help="Path to saved bracket JSON")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: {json_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    bracket = data.get("bracket", {})

    print(SEP)
    print("  2026 BRACKET SCORING — ESPN STANDARD RULES".center(W))
    print(f"  (R64=1pt  R32=2pt  S16=4pt  E8=8pt  FF=16pt  Champion=32pt)".center(W))
    print(SEP)

    # Score the base bracket path (same for all types except champion pick)
    breakdown = _score_bracket_path(bracket)

    # ── Per-type champion picks ────────────────────────────────────────────
    type_configs: list[dict] = []

    # Deterministic
    det = data.get("deterministic_recommendation", {})
    det_champ = (det.get("champion") or {}).get("name", breakdown["champion_path"])
    type_configs.append({
        "label":          "Deterministic",
        "champion_pick":  det_champ,
        "breakdown":      breakdown,
        "champ_pts":      _score_champion_pick(det_champ),
        "total":          _total(breakdown, det_champ),
    })

    # Safe
    safe = data.get("safe_recommendation", {})
    safe_champ = (safe.get("primary") or {}).get("name", breakdown["champion_path"])
    type_configs.append({
        "label":          "Safe",
        "champion_pick":  safe_champ,
        "breakdown":      breakdown,
        "champ_pts":      _score_champion_pick(safe_champ),
        "total":          _total(breakdown, safe_champ),
    })

    # Value
    val = data.get("value_recommendation", {})
    val_champ = (val.get("primary") or {}).get("name", breakdown["champion_path"])
    type_configs.append({
        "label":          "Value",
        "champion_pick":  val_champ,
        "breakdown":      breakdown,
        "champ_pts":      _score_champion_pick(val_champ),
        "total":          _total(breakdown, val_champ),
    })

    # Contrarian
    con = data.get("contrarian_recommendation", {})
    con_champ = (con.get("primary") or {}).get("name", breakdown["champion_path"])
    type_configs.append({
        "label":          "Contrarian",
        "champion_pick":  con_champ,
        "breakdown":      breakdown,
        "champ_pts":      _score_champion_pick(con_champ),
        "total":          _total(breakdown, con_champ),
    })

    # ── Per-bracket detailed breakdown ────────────────────────────────────
    print()
    print("  DETAILED BREAKDOWN BY BRACKET TYPE")
    print(SEP)
    for cfg in type_configs:
        _print_bracket_score(
            label         = cfg["label"],
            breakdown     = cfg["breakdown"],
            champion_pick = cfg["champion_pick"],
            champ_pts     = cfg["champ_pts"],
            total_pts     = cfg["total"],
        )

    # ── First Four ────────────────────────────────────────────────────────
    first_four = data.get("first_four", [])
    if first_four:
        print()
        print("  FIRST FOUR (informational — not ESPN-scored)")
        print(f"  {'Region':<8} {'Seed':>4}  {'Predicted':<22} {'Actual':<22} {'Match?'}")
        print("  " + "─" * 66)
        for game in first_four:
            pred   = game.get("winner", "?")
            region = game.get("region", "?")
            seed   = int(game.get("seed", 0))
            actual_game = ACTUAL["first_four_games"].get((region, seed), {})
            actual_w    = actual_game.get("winner", "?")
            match       = "✓" if pred == actual_w else "✗"
            print(f"  {region:<8} #{seed:>2}    {pred:<22} {actual_w:<22} {match}")

    # ── Summary table ─────────────────────────────────────────────────────
    _print_summary_table(type_configs)


if __name__ == "__main__":
    main()
