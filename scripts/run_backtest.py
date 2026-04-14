"""
run_backtest.py
CLI entry point for the March Madness historical backtest engine.

Usage
-----
  python3 scripts/run_backtest.py <year> [mode]

  year : any tournament year present in data/raw/TourneyCompactResults.csv
         (default dataset covers 1985–2016)
  mode : conservative | balanced (default) | upset_heavy

Examples
--------
  python3 scripts/run_backtest.py 2016
  python3 scripts/run_backtest.py 2016 balanced
  python3 scripts/run_backtest.py 2010 upset_heavy

Output
------
  Terminal  — backtest summary with accuracy by round + diagnostics
  File      — data/processed/backtests/{year}_{mode}_backtest.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.backtest import run_backtest, ROUND_NAMES, MAX_PICKS_PER_ROUND

W    = 70
SEP  = "=" * W
THIN = "─" * W


# ── Formatting helpers ────────────────────────────────────────────────────────

def _team_label(t: dict) -> str:
    """Format a team dict as 'T1314 (seed  1)'."""
    seed_str = f"seed {t['seed']:>2}" if t["seed"] is not None else "seed  ?"
    return f"{t['name']:<10} ({seed_str})"


def _pct(correct: int, possible: int) -> str:
    return f"{correct / possible:.1%}" if possible > 0 else "—"


# ── Section printers ─────────────────────────────────────────────────────────

def print_header(year: int, mode: str, hist_range: tuple) -> None:
    print(SEP)
    print(f"{'MARCH MADNESS  —  HISTORICAL BACKTEST':^{W}}")
    print(f"{'Year: ' + str(year) + '   Mode: ' + mode.upper():^{W}}")
    print(SEP)
    print(f"  Pre-tournament model: {hist_range[0]}–{hist_range[1]}")
    print(f"  (Only data from seasons before {year} was used to generate this bracket)")


def print_final_four_comparison(result: dict) -> None:
    pred   = result["predicted_final_four"]
    actual = result["actual_final_four"]

    print(f"\n{THIN}")
    print(f"  FINAL FOUR COMPARISON")
    print(THIN)

    # Side-by-side, up to 4 teams each
    header = f"  {'PREDICTED':<34}  {'ACTUAL':<34}"
    print(header)
    print(f"  {'─'*32}  {'─'*32}")

    for i in range(max(len(pred), len(actual))):
        p = _team_label(pred[i])   if i < len(pred)   else ""
        a = _team_label(actual[i]) if i < len(actual) else ""
        print(f"  {p:<34}  {a:<34}")


def print_champion_comparison(result: dict) -> None:
    pred   = result["predicted_champion"]
    actual = result["actual_champion"]
    match  = pred["name"] == actual["name"]
    tag    = "  ✓ CORRECT" if match else "  ✗ WRONG"

    print(f"\n{THIN}")
    print(f"  CHAMPION COMPARISON")
    print(THIN)
    print(f"  Predicted : {_team_label(pred)}")
    print(f"  Actual    : {_team_label(actual)}{tag}")


def print_accuracy(result: dict) -> None:
    detail = result["by_round_detail"]
    total  = result["total_correct"]
    total_p = result["total_possible"]

    print(f"\n{THIN}")
    print(f"  ACCURACY BY ROUND")
    print(THIN)
    print(f"  {'Round':<18}  {'Correct':>7}  {'Possible':>8}  {'Accuracy':>9}")
    print(f"  {'─'*18}  {'─'*7}  {'─'*8}  {'─'*9}")

    for round_name in ROUND_NAMES.values():
        if round_name not in detail:
            continue
        c = detail[round_name]["correct"]
        p = detail[round_name]["possible"]
        print(f"  {round_name:<18}  {c:>7}  {p:>8}  {_pct(c, p):>9}")

    print(f"  {'─'*18}  {'─'*7}  {'─'*8}  {'─'*9}")
    print(f"  {'TOTAL':<18}  {total:>7}  {total_p:>8}  {_pct(total, total_p):>9}")


def print_round_detail(result: dict) -> None:
    """Show which teams were predicted correctly vs incorrectly by round."""
    predicted = result["_predicted_full"]
    actual    = result["_actual_results"]

    BRACKET_KEY = {
        "Round of 64":  "round_of_64",
        "Round of 32":  "round_of_32",
        "Sweet 16":     "sweet_16",
        "Elite 8":      "elite_8",
        "Final Four":   "final_four",
        "Championship": "championship",
    }

    print(f"\n{THIN}")
    print(f"  ROUND-BY-ROUND BREAKDOWN")
    print(THIN)

    for round_name, bracket_key in BRACKET_KEY.items():
        games = predicted.get(bracket_key, [])
        if isinstance(games, dict):
            games = [games]

        actual_names = {t["name"] for t in actual.get(round_name, [])}

        hits   = [g for g in games if g["winner"]["name"] in actual_names]
        misses = [g for g in games if g["winner"]["name"] not in actual_names]

        print(f"\n  {round_name}  ({len(hits)}/{len(games)} correct)")
        for g in hits:
            w = g["winner"]
            print(f"    ✓  {w['name']} (seed {w['seed']:>2})")
        for g in misses:
            w = g["winner"]
            l = g["loser"]
            # Show actual winner of this game slot if we can find them
            print(f"    ✗  predicted {w['name']} (seed {w['seed']:>2}) "
                  f"[actual: {l['name'] if l['name'] in actual_names else '?'}]")


def print_diagnostics(result: dict) -> None:
    dx = result["diagnostics"]

    print(f"\n{THIN}")
    print(f"  DIAGNOSTICS")
    print(THIN)

    labels = [
        ("Chalkiness",     "chalkiness_assessment"),
        ("Upsets",         "upset_assessment"),
        ("Sweet 16 DD",    "sweet_16_double_digit_review"),
        ("Champion pick",  "champion_review"),
    ]
    for label, key in labels:
        text = dx.get(key, "")
        # Wrap to terminal width
        print(f"\n  [{label}]")
        words = text.split()
        line  = "  "
        for word in words:
            if len(line) + len(word) + 1 > W - 2:
                print(line)
                line = "  " + word
            else:
                line = (line + " " + word).lstrip()
                line = "  " + line.lstrip()
        if line.strip():
            print(line)


def print_s16_notes(result: dict) -> None:
    notes = result["_predicted_full"].get("s16_constraint_notes", [])
    if not notes:
        return
    print(f"\n{THIN}")
    print(f"  SWEET 16 STRUCTURE ADJUSTMENTS")
    print(THIN)
    for note in notes:
        print(f"  • {note}")


def print_ratings_diagnostics(result: dict) -> None:
    diag = result.get("ratings_diagnostics", {})
    if not diag or diag.get("source") is None:
        return
    total  = diag.get("total_teams", 64)
    by_id  = diag.get("matched_by_id",   0)
    by_nm  = diag.get("matched_by_name", 0)
    sfb    = diag.get("seed_fallback",   total)
    source = diag.get("source", "?")
    matched = by_id + by_nm

    print(f"\n{THIN}")
    print(f"  TEAM RATINGS COVERAGE  (season {diag.get('season', '?')})")
    print(THIN)
    print(f"  Source              : {source}")
    print(f"  Total teams         : {total}")
    print(f"  Matched by team_id  : {by_id:>3}  ({by_id/total:.0%})" if total else "")
    print(f"  Matched by name     : {by_nm:>3}  ({by_nm/total:.0%})" if total else "")
    print(f"  Real ratings used   : {matched:>3}  ({matched/total:.0%})" if total else "")
    print(f"  Seed-only fallback  : {sfb:>3}  ({sfb/total:.0%})" if total else "")


def print_footer(result: dict) -> None:
    print(f"\n{SEP}")
    print(f"  Saved → {result['_output_file']}")
    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Parse args ────────────────────────────────────────────────────────
    if len(sys.argv) < 2:
        sys.exit(
            "Usage: python3 scripts/run_backtest.py <year> [mode]\n"
            "  year : e.g. 2016\n"
            "  mode : conservative | balanced (default) | upset_heavy"
        )

    try:
        year = int(sys.argv[1])
    except ValueError:
        sys.exit(f"Error: '{sys.argv[1]}' is not a valid year.")

    mode = sys.argv[2] if len(sys.argv) > 2 else "balanced"
    valid_modes = {"conservative", "balanced", "upset_heavy"}
    if mode not in valid_modes:
        sys.exit(
            f"Invalid mode '{mode}'.  Choose from: {', '.join(sorted(valid_modes))}"
        )

    # ── Run backtest ──────────────────────────────────────────────────────
    try:
        result = run_backtest(year, mode)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    # ── Print results ─────────────────────────────────────────────────────
    print_header(year, mode, result["_hist_range"])
    print_ratings_diagnostics(result)
    print_final_four_comparison(result)
    print_champion_comparison(result)
    print_accuracy(result)
    print_s16_notes(result)
    print_diagnostics(result)
    print_footer(result)


if __name__ == "__main__":
    main()
