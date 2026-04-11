"""
build_optimizer.py
CLI entry point for the March Madness seed-level bracket optimizer.

Prints:
  1. Upset target ranges by round
  2. Historical Final Four seed frequency
  3. Three bracket structures (conservative / balanced / upset-heavy)
  4. Recommended champion seed per mode

Saves:
  data/processed/optimal_bracket_structure.json
"""

import json
import sys
from itertools import groupby
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.bracket_optimizer import (
    OUTPUT_FILE,
    compute_optimal_upset_profile,
    compute_seed_round_distribution,
    generate_multiple_structures,
    generate_optimal_bracket_structure,
    load_data,
)

W = 66   # terminal width
SEP  = "=" * W
THIN = "─" * W


# ── Formatting helpers ────────────────────────────────────────────────────────

def _seed_list_summary(seeds: list[int]) -> str:
    """
    Compact representation of a sorted seed list.
    [1,1,1,2,2,3]  →  "1×3  2×2  3"
    """
    parts = []
    for seed, grp in groupby(seeds):
        cnt = sum(1 for _ in grp)
        parts.append(f"{seed}×{cnt}" if cnt > 1 else str(seed))
    return "  ".join(parts)


def _bar(value: float, scale: float = 10.0, width: int = 20) -> str:
    """ASCII bar proportional to value."""
    filled = min(width, round(value * scale))
    return "█" * filled + "░" * (width - filled)


# ── Print sections ────────────────────────────────────────────────────────────

def print_upset_profile(profile: dict) -> None:
    print(f"\n{'UPSET TARGETS BY ROUND':^{W}}")
    print(SEP)
    print(f"  {'Round':<18}  {'Avg':>5}  {'Med':>5}  {'Std':>5}  {'Target range':>14}")
    print(f"  {'-'*18}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*14}")
    for rname, s in profile.items():
        rng = f"[{s['target_min']} – {s['target_max']}]"
        print(
            f"  {rname:<18}  {s['average']:>5.1f}  {s['median']:>5.1f}"
            f"  {s['std']:>5.2f}  {rng:>14}"
        )


def print_ff_distribution(dist: dict, n_years: int) -> None:
    """Historical average Final Four seed frequency."""
    ff = dist["Elite 8"]["advances"]   # wins in E8 = teams reaching FF
    print(f"\n{'FINAL FOUR SEED FREQUENCY  (1985–2016)':^{W}}")
    print(SEP)
    print(f"  {'Seed':>4}  {'Avg/yr':>7}  {'Total':>7}  {'':20}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*20}")
    for seed in range(1, 17):
        avg = ff.get(seed, 0.0)
        if avg < 0.01:
            continue
        total = round(avg * n_years)
        bar   = _bar(avg, scale=8, width=18)
        print(f"  {seed:>4}  {avg:>7.3f}  {total:>7}  {bar}")


def print_structure(mode: str, structure: dict) -> None:
    label = mode.replace("_", "-").upper()
    score = structure.get("score", 0)
    print(f"\n{THIN}")
    print(f"  {label}  —  plausibility score: {score:.4f}")
    print(THIN)

    rows = [
        ("round_of_32",   "Round of 32",   32),
        ("sweet_16",      "Sweet 16",       16),
        ("elite_8",       "Elite 8",         8),
        ("final_four",    "Final Four",       4),
        ("championship",  "Championship",     2),
    ]
    for key, label, size in rows:
        seeds = structure.get(key, [])
        summary = _seed_list_summary(seeds)
        print(f"  {label:<16}({size:>2} teams)  {summary}")

    champ = structure.get("champion")
    print(f"  {'Champion':<16}( 1 team )  Seed {champ}")

    print(f"\n  Upset targets:")
    for rname, cnt in structure.get("upset_profile", {}).items():
        bar = "▪" * cnt
        print(f"    {rname:<18}  {cnt:>2}  {bar}")


def print_champion_summary(structures: dict) -> None:
    print(f"\n{'RECOMMENDED CHAMPION SEED BY MODE':^{W}}")
    print(SEP)
    for mode, structure in structures.items():
        label = mode.replace("_", "-")
        champ = structure.get("champion")
        score = structure.get("score", 0)
        print(f"  {label:<16}  Seed {champ:<4}  (score {score:.4f})")


def print_top_ff_combinations(structures: dict) -> None:
    """Show the Final Four seed lineup for each mode."""
    print(f"\n{'MOST LIKELY FINAL FOUR COMBINATIONS':^{W}}")
    print(SEP)
    for mode, structure in structures.items():
        label = mode.replace("_", "-")
        ff = structure.get("final_four", [])
        print(f"  {label:<16}  {ff}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP)
    print(f"{'MARCH MADNESS  —  BRACKET OPTIMIZER  (STEP 2)':^{W}}")
    print(SEP)

    df, _   = load_data()
    n_years = df["year"].nunique()
    print(
        f"  Dataset : {df['year'].min()}–{df['year'].max()}"
        f"  |  {n_years} seasons  |  {len(df):,} games"
    )

    dist    = compute_seed_round_distribution(df)
    profile = compute_optimal_upset_profile(df)

    # ── 1. Upset ranges ──────────────────────────────────────────────────
    print_upset_profile(profile)

    # ── 2. Final Four seed frequency ─────────────────────────────────────
    print_ff_distribution(dist, n_years)

    # ── 3. Three named bracket structures ────────────────────────────────
    structures = generate_optimal_bracket_structure()
    print(f"\n{'BRACKET STRUCTURES':^{W}}")
    for mode in ("conservative", "balanced", "upset_heavy"):
        print_structure(mode, structures[mode])

    # ── 4. Top Final Four combinations ───────────────────────────────────
    print_top_ff_combinations(structures)

    # ── 5. Champion recommendation ───────────────────────────────────────
    print_champion_summary(structures)

    # ── 6. Extended spectrum (10 structures) ─────────────────────────────
    extended = generate_multiple_structures(n=10)
    print(f"\n{'EXTENDED SPECTRUM  (n=10, sorted by plausibility)':^{W}}")
    print(SEP)
    print(f"  {'Label':<22}  {'Champion':>8}  {'FF seeds':>22}  {'Score':>7}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*22}  {'-'*7}")
    for s in extended:
        lbl   = s.get("label", "")
        champ = s.get("champion", "?")
        ff    = str(s.get("final_four", []))
        score = s.get("score", 0)
        print(f"  {lbl:<22}  {champ:>8}  {ff:>22}  {score:>7.4f}")

    # ── 7. Save JSON ─────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(structures, f, indent=2)

    print(f"\n{SEP}")
    print(f"  Saved → {OUTPUT_FILE}")
    print(SEP)


if __name__ == "__main__":
    main()
