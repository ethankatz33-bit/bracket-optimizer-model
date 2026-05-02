"""
scripts/compare_pool_brackets.py
Diagnostic: compare bracket picks across conservative / balanced / upset_heavy modes.

Prints:
  • Upset counts by round × mode
  • Pick differences between mode pairs by round
  • Value-boosted picks by mode
  • Current threshold settings

Usage
-----
  python3 scripts/compare_pool_brackets.py
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import helper functions from predict_future_bracket without running main()
_pfb_path = PROJECT_ROOT / "scripts" / "predict_future_bracket.py"
_pfb_spec  = importlib.util.spec_from_file_location("predict_future_bracket", _pfb_path)
_pfb       = importlib.util.module_from_spec(_pfb_spec)
_pfb_spec.loader.exec_module(_pfb)

from lib.team_selector import (
    simulate_bracket,
    UPSET_MIN_WIN_PROB,
    UPSET_MIN_DESIRABILITY,
    UPSET_MIN_MODEL_WP_BY_ROUND,
    UPSET_MAX_BY_ROUND,
    R64_GLOBAL_SEED_CAPS,
    ADV_VALUE_CONFIG,
    ADV_MIN_MODEL_PCT,
    ROUND_VALUE_WEIGHTS,
    _ADV_MODE_MAP,
)

try:
    from lib.team_selector import EARLY_ROUND_UPSET_CONFIG
    _HAS_EARLY_CONFIG = True
except ImportError:
    EARLY_ROUND_UPSET_CONFIG = {}
    _HAS_EARLY_CONFIG = False

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_CSV   = PROJECT_ROOT / "data" / "future" / "future_bracket_2026.csv"
PICKS_CSV   = PROJECT_ROOT / "data" / "future" / "public_picks_2026.csv"
SEASON      = 2026
MODES       = ["conservative", "balanced", "upset_heavy"]
ROUNDS      = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
W           = 88

# ── Load and prepare teams ─────────────────────────────────────────────────────

import pandas as pd

def _load_teams_override() -> dict:
    """Load 2026 CSV, handle First Four, build teams_override."""
    df = pd.read_csv(INPUT_CSV)
    df_64, _ = _pfb._simulate_first_four(df)

    file_picks: dict = {}
    if PICKS_CSV.exists():
        file_picks = _pfb._load_public_picks_file(PICKS_CSV)

    picks, _ = _pfb._build_public_picks(df_64, file_picks, {})
    teams = _pfb._build_teams_override(df_64)

    # Attach public_pick_pct to every team dict
    fuzzy = _pfb._build_fuzzy_lookup(picks)
    for region, seed_map in teams.items():
        for seed, t in seed_map.items():
            name = t["name"]
            pct  = picks.get(name)
            if pct is None:
                pct = picks.get(fuzzy.get(_pfb._normalize_name(name), ""))
            if pct is not None:
                t["public_pick_pct"] = float(pct)
    return teams


# ── Bracket comparison helpers ─────────────────────────────────────────────────

_ROUND_KEY = {
    "Round of 64":  "round_of_64",
    "Round of 32":  "round_of_32",
    "Sweet 16":     "sweet_16",
    "Elite 8":      "elite_8",
    "Final Four":   "final_four",
    "Championship": "championship",
}


def _get_round_games(bracket: dict, round_name: str) -> list[dict]:
    key  = _ROUND_KEY[round_name]
    data = bracket.get(key, [])
    if isinstance(data, dict):   # championship is a single dict
        data = [data]
    return data


def _winner_set(bracket: dict, round_name: str) -> set[str]:
    return {g["winner"]["name"] for g in _get_round_games(bracket, round_name)}


def _upset_count(bracket: dict, round_name: str) -> int:
    return sum(1 for g in _get_round_games(bracket, round_name) if g.get("is_upset"))


def _value_boosted(bracket: dict) -> list[dict]:
    return bracket.get("advancement_value_plays", [])


# ── Part 1: variance comparison ────────────────────────────────────────────────

def run_comparison(brackets: dict[str, dict]) -> None:
    print()
    print("=" * W)
    print("PART 1 — UPSET COUNTS BY ROUND × MODE")
    print("=" * W)

    # Header
    col = 16
    hdr = f"  {'Round':<18}" + "".join(f"  {m:<14}" for m in MODES)
    print(hdr)
    print("  " + "─" * (W - 2))

    for rnd in ROUNDS:
        row = f"  {rnd:<18}"
        for mode in MODES:
            cnt = _upset_count(brackets[mode], rnd)
            row += f"  {cnt:<14}"
        print(row)

    # Total
    print("  " + "─" * (W - 2))
    row = f"  {'TOTAL':<18}"
    for mode in MODES:
        total = sum(_upset_count(brackets[mode], r) for r in ROUNDS)
        row += f"  {total:<14}"
    print(row)

    # Champion row
    print()
    champ_row = f"  {'Champion':<18}"
    for mode in MODES:
        c = brackets[mode]["champion"]
        champ_row += f"  #{c['seed']} {c['name'][:12]:<12}"
    print(champ_row)

    ff_row = f"  {'Final Four':<18}"
    for mode in MODES:
        ff = [g["winner"]["name"][:8] for g in _get_round_games(brackets[mode], "Final Four")]
        ff += [g["loser"]["name"][:8]  for g in _get_round_games(brackets[mode], "Final Four")]
        ff_row += f"  {', '.join(sorted(set(ff)))[:14]:<14}"
    print(ff_row)

    print()
    print("=" * W)
    print("PART 1b — PICK DIFFERENCES BETWEEN MODE PAIRS")
    print("=" * W)

    pairs = [
        ("conservative", "balanced",     "conservative vs value"),
        ("balanced",     "upset_heavy",  "value vs contrarian"),
        ("conservative", "upset_heavy",  "conservative vs contrarian"),
    ]

    for m_a, m_b, label in pairs:
        print(f"\n  {label}:")
        total_diff = 0
        for rnd in ROUNDS:
            a_set = _winner_set(brackets[m_a], rnd)
            b_set = _winner_set(brackets[m_b], rnd)
            diff  = a_set.symmetric_difference(b_set)
            total_diff += len(diff)
            if diff:
                diff_str = ", ".join(sorted(diff))
                print(f"    {rnd:<18}  {len(diff)} different  [{diff_str}]")
            else:
                print(f"    {rnd:<18}  identical")
        print(f"    {'TOTAL DIFFERENCES':<18}  {total_diff}")


def run_value_boosted(brackets: dict[str, dict]) -> None:
    print()
    print("=" * W)
    print("PART 1c — ESPN VALUE-BOOSTED PICKS BY MODE")
    print("=" * W)

    for mode in MODES:
        plays = _value_boosted(brackets[mode])
        print(f"\n  {mode.upper()} ({len(plays)} boosts applied):")
        if not plays:
            print("    (none)")
        else:
            for p in plays:
                print(
                    f"    {p['round']:<16}  #{p['seed']} {p['team']:<20}  "
                    f"vs {p['opponent']:<20}  edge={p['edge']:+.1%}  "
                    f"desir={p['desir_final']:.5f}"
                )


# ── Part 2: threshold inspection ──────────────────────────────────────────────

def run_threshold_inspection() -> None:
    print()
    print("=" * W)
    print("PART 2 — CURRENT THRESHOLD SETTINGS BY MODE")
    print("=" * W)

    print("\n  UPSET_MIN_WIN_PROB:")
    for m, v in UPSET_MIN_WIN_PROB.items():
        print(f"    {m:<16}  {v}")

    print("\n  UPSET_MIN_DESIRABILITY:")
    for m, v in UPSET_MIN_DESIRABILITY.items():
        print(f"    {m:<16}  {v}")

    print("\n  UPSET_MAX_BY_ROUND:")
    hdr = f"    {'Round':<18}" + "".join(f"  {m:<14}" for m in ["conservative", "balanced", "upset_heavy"])
    print(hdr)
    for rnd in ROUNDS:
        row_d = UPSET_MAX_BY_ROUND.get(rnd, {})
        row = f"    {rnd:<18}"
        for m in ["conservative", "balanced", "upset_heavy"]:
            row += f"  {str(row_d.get(m, '?')):<14}"
        print(row)

    print("\n  UPSET_MIN_MODEL_WP_BY_ROUND (eff-margin path defaults):")
    for rnd, v in UPSET_MIN_MODEL_WP_BY_ROUND.items():
        print(f"    {rnd:<18}  {v}")

    print("\n  R64_GLOBAL_SEED_CAPS:")
    for m, caps in R64_GLOBAL_SEED_CAPS.items():
        print(f"    {m:<16}  {caps}")

    print("\n  ADV_VALUE_CONFIG (ESPN advancement boosts):")
    for m, cfg in ADV_VALUE_CONFIG.items():
        print(f"    {m:<16}  min_edge={cfg['min_edge']}  boost={cfg['boost']}")

    if _HAS_EARLY_CONFIG and EARLY_ROUND_UPSET_CONFIG:
        print("\n  EARLY_ROUND_UPSET_CONFIG (new — R64/R32 only):")
        fields = ["r64_extra_upsets", "r32_extra_upsets", "value_boost_mult", "min_edge", "min_model_wp"]
        hdr = f"    {'Mode':<16}" + "".join(f"  {f:<18}" for f in fields)
        print(hdr)
        for m in ["conservative", "balanced", "value", "contrarian", "upset_heavy"]:
            cfg = EARLY_ROUND_UPSET_CONFIG.get(m, {})
            if cfg:
                row = f"    {m:<16}"
                for f in fields:
                    row += f"  {str(cfg.get(f, '?')):<18}"
                print(row)
        print()
        print("  → Early-round thresholds differ by mode: YES" if len(EARLY_ROUND_UPSET_CONFIG) > 1 else
              "  → Early-round thresholds differ by mode: NO (only one entry)")
    else:
        print("\n  EARLY_ROUND_UPSET_CONFIG: NOT YET ADDED")
        print("  → Early-round thresholds differ by mode: NO (all modes identical)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * W)
    print("BRACKET MODE COMPARISON — 2026")
    print("=" * W)

    print("\n  Loading 2026 teams...")
    teams = _load_teams_override()
    print(f"  Loaded {sum(len(v) for v in teams.values())} teams across {len(teams)} regions")

    brackets: dict[str, dict] = {}
    for mode in MODES:
        print(f"\n  Simulating {mode}...", flush=True)
        brackets[mode] = simulate_bracket(
            mode,
            season=SEASON,
            _teams_override=teams,
        )
        champ = brackets[mode]["champion"]
        ff    = [g["winner"]["name"] for g in brackets[mode]["final_four"]]
        ff   += [g["loser"]["name"]  for g in brackets[mode]["final_four"]]
        total_upsets = sum(_upset_count(brackets[mode], r) for r in ROUNDS)
        print(f"    Champion: #{champ['seed']} {champ['name']}  |  FF: {', '.join(sorted(set(ff)))}  |  Total upsets: {total_upsets}")

    run_threshold_inspection()
    run_comparison(brackets)
    run_value_boosted(brackets)

    print()
    print("=" * W)
    print("SUMMARY")
    print("=" * W)
    print()
    print("  Mode          | R64   R32   S16   E8    FF    CG    Total")
    print("  " + "─" * (W - 2))
    for mode in MODES:
        counts = [_upset_count(brackets[mode], r) for r in ROUNDS]
        total  = sum(counts)
        print(f"  {mode:<14}|  " + "  ".join(f"{c:<4}" for c in counts) + f"  {total}")
    print()


if __name__ == "__main__":
    main()
