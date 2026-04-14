"""
build_team_ratings.py
CLI for building and inspecting team ratings.

Usage
-----
  python3 scripts/build_team_ratings.py [input_csv]

  input_csv : path to team stats CSV
              (default: data/processed/team_stats.csv)

  If the input file is not found, a sample dataset is generated automatically
  so you can see the model in action before plugging in real data.

Input CSV columns
-----------------
  Required : season, team_name, seed
  Optional : efficiency_margin, offense_rating, defense_rating,
             recent_form, strength_of_schedule, ap_rank_week6

Output
------
  Terminal — top-15 teams by rating, top-15 by champion profile,
             sample matchup win probabilities
  Files    — data/processed/team_ratings.csv
             data/processed/team_ratings.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.team_ratings import (
    build_and_save,
    load_team_data,
    predict_win_probability,
    rate_teams_by_season,
    save_team_ratings,
)

W    = 72
SEP  = "=" * W
THIN = "─" * W


# ════════════════════════════════════════════════════════════════════════════
# Sample data generator
# ════════════════════════════════════════════════════════════════════════════

# Baseline stats per seed — derived from typical NCAA tournament profiles.
# offense_rating : points scored per 100 possessions (higher = better)
# defense_rating : points allowed per 100 possessions (lower = better)
# efficiency_margin = offense - defense
# recent_form    : fraction of last 10 games won (0–1)
# sos            : strength of schedule index (higher = tougher)
_SEED_PROFILES: dict[int, tuple] = {
    #  seed  offense  defense   eff_margin  recent_form   sos    ap_rank_base
    1:  (120.5,  90.2,  +30.3,   0.90,  0.82,   4),
    2:  (118.2,  92.1,  +26.1,   0.86,  0.79,   9),
    3:  (116.0,  93.8,  +22.2,   0.83,  0.75,  16),
    4:  (113.5,  95.5,  +18.0,   0.80,  0.72,  None),
    5:  (111.0,  97.4,  +13.6,   0.76,  0.68,  None),
    6:  (108.8,  99.2,  + 9.6,   0.73,  0.65,  None),
    7:  (106.5, 101.0,  + 5.5,   0.70,  0.61,  None),
    8:  (104.2, 103.0,  + 1.2,   0.66,  0.57,  None),
    9:  (103.8, 104.0,  - 0.2,   0.63,  0.54,  None),
    10: (102.5, 105.5,  - 3.0,   0.60,  0.52,  None),
    11: (101.0, 106.8,  - 5.8,   0.58,  0.49,  None),
    12: ( 99.5, 108.2,  - 8.7,   0.55,  0.46,  None),
    13: ( 97.2, 110.0,  -12.8,   0.51,  0.42,  None),
    14: ( 95.0, 112.5,  -17.5,   0.47,  0.38,  None),
    15: ( 92.5, 115.0,  -22.5,   0.42,  0.33,  None),
    16: ( 89.0, 118.2,  -29.2,   0.36,  0.27,  None),
}

# Fictional team names for the sample bracket (4 regions × 16 seeds = 64 teams)
_SAMPLE_TEAMS: dict[str, dict[int, str]] = {
    "East": {
        1: "Riverside", 2: "Lakewood", 3: "Millfield", 4: "Creekside",
        5: "Westmont", 6: "Hillbrook", 7: "Oakdale", 8: "Fernwood",
        9: "Pinehurst", 10: "Sycamore", 11: "Clearwater", 12: "Elmhurst",
        13: "Ridgemont", 14: "Cedarville", 15: "Maplewood", 16: "Stonegate",
    },
    "West": {
        1: "Summit", 2: "Harborview", 3: "Cliffside", 4: "Bayshore",
        5: "Cascadia", 6: "Redwood", 7: "Lakeview", 8: "Northport",
        9: "Eastbrook", 10: "Pineridge", 11: "Holloway", 12: "Belmont",
        13: "Fairfield", 14: "Greendale", 15: "Ashford", 16: "Brookhaven",
    },
    "South": {
        1: "Midland", 2: "Southport", 3: "Crestview", 4: "Ironwood",
        5: "Sunridge", 6: "Palmview", 7: "Glendale", 8: "Westfield",
        9: "Lakeside", 10: "Thornwood", 11: "Rosewood", 12: "Foxhill",
        13: "Springdale", 14: "Deerfield", 15: "Weston", 16: "Riverdale",
    },
    "Midwest": {
        1: "Northland", 2: "Hillcrest", 3: "Eastgate", 4: "Bluewater",
        5: "Prairieton", 6: "Cannonball", 7: "Redstone", 8: "Wheatfield",
        9: "Cornbury", 10: "Irondale", 11: "Silverstone", 12: "Copperhill",
        13: "Dusty Mesa", 14: "Flatrock", 15: "Timberline", 16: "Plainview",
    },
}


def generate_sample_data(season: int = 2025, rng_seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic 64-team sample dataset for one tournament season.

    Stats are derived from _SEED_PROFILES with small random noise to give
    teams on the same seed line meaningfully different ratings.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []

    for region, seed_map in _SAMPLE_TEAMS.items():
        for seed, team_name in seed_map.items():
            off, defe, eff, form, sos, ap_base = _SEED_PROFILES[seed]

            # Add per-team noise so same-seed teams are distinguishable
            noise = rng.normal(0, 1)   # single noise draw scales all stats
            row = {
                "season":               season,
                "team_name":            team_name,
                "region":               region,
                "seed":                 seed,
                "offense_rating":       round(off  + noise * 2.5, 1),
                "defense_rating":       round(defe - noise * 2.0, 1),
                "efficiency_margin":    round(eff  + noise * 4.0, 1),
                "recent_form":          round(min(1.0, max(0.0, form + noise * 0.06)), 3),
                "strength_of_schedule": round(min(1.0, max(0.0, sos  + noise * 0.04)), 3),
            }
            # AP rank: only for teams seeded 1–4 (others are typically unranked)
            if ap_base is not None:
                row["ap_rank_week6"] = max(1, int(ap_base + rng.integers(-3, 4)))
            else:
                row["ap_rank_week6"] = None

            rows.append(row)

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Print helpers
# ════════════════════════════════════════════════════════════════════════════

def _print_top15(df: pd.DataFrame, score_col: str, title: str) -> None:
    top = (
        df.nlargest(15, score_col)
          [["season", "team_name", "seed", "team_rating", "champion_profile_score"]]
          .reset_index(drop=True)
    )
    print(f"\n{THIN}")
    print(f"  {title}")
    print(THIN)
    print(f"  {'#':>2}  {'Team':<20}  {'Seed':>4}  {'Rating':>8}  {'Champ Profile':>13}")
    print(f"  {'─'*2}  {'─'*20}  {'─'*4}  {'─'*8}  {'─'*13}")
    for rank, row in top.iterrows():
        print(
            f"  {rank+1:>2}  {row['team_name']:<20}  {row['seed']:>4}  "
            f"{row['team_rating']:>8.4f}  {row['champion_profile_score']:>13.4f}"
        )


def _print_matchup(team_a: dict, team_b: dict) -> None:
    probs = predict_win_probability(team_a, team_b)
    ta    = team_a["team_name"]
    tb    = team_b["team_name"]
    sa    = team_a["seed"]
    sb    = team_b["seed"]
    print(
        f"  {ta:<20} (S{sa:>2})  vs  {tb:<20} (S{sb:>2})"
        f"   →  {probs['team_a']:.1%} / {probs['team_b']:.1%}"
    )


def _print_matchup_section(df: pd.DataFrame) -> None:
    print(f"\n{THIN}")
    print(f"  SAMPLE MATCHUP WIN PROBABILITIES")
    print(THIN)
    print(f"  {'Team A':<20}  {'':>4}       {'Team B':<20}  {'':>4}     {'A wins / B wins':>16}")
    print(f"  {'─'*68}")

    def pick(seed: int, prefer_region: str | None = None) -> dict:
        """Pick a team dict by seed, optionally preferring a region."""
        sub = df[df["seed"] == seed]
        if prefer_region:
            r = sub[sub.get("region", pd.Series()) == prefer_region] if "region" in sub.columns else pd.DataFrame()
            if not r.empty:
                sub = r
        return sub.iloc[0].to_dict()

    # 1v16: classic heavy favorite
    _print_matchup(pick(1, "East"),  pick(16, "East"))
    # 2v15
    _print_matchup(pick(2, "East"),  pick(15, "East"))
    # 1v2: championship-style
    _print_matchup(pick(1, "East"),  pick(2, "West"))
    # 5v12: classic upset matchup
    _print_matchup(pick(5, "South"), pick(12, "South"))
    # 6v11: mid-tier upset
    _print_matchup(pick(6, "South"), pick(11, "South"))
    # 1 vs 1: Final Four — same seed, different teams
    _print_matchup(pick(1, "East"),  pick(1, "Midwest"))
    # 3v2: tight late-round game
    _print_matchup(pick(3, "West"),  pick(2, "West"))


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    print(SEP)
    print(f"{'MARCH MADNESS  —  TEAM RATINGS  (VERSION 1)':^{W}}")
    print(SEP)

    # ── Load or generate data ─────────────────────────────────────────────
    using_sample = False
    if input_path is None or not input_path.exists():
        default = PROJECT_ROOT / "data" / "processed" / "team_stats.csv"
        if default.exists():
            print(f"\n  Loading: {default}")
            df_raw = load_team_data(default)
        else:
            print(
                "\n  No input file found — generating a 64-team sample dataset.\n"
                "  To use real data, pass a CSV path as the first argument:\n"
                "    python3 scripts/build_team_ratings.py path/to/team_stats.csv\n"
                f"\n  Expected columns: season, team_name, seed, offense_rating,\n"
                f"  defense_rating, efficiency_margin, recent_form,\n"
                f"  strength_of_schedule, ap_rank_week6 (optional)\n"
            )
            df_raw = generate_sample_data(season=2025)
            using_sample = True
    else:
        print(f"\n  Loading: {input_path}")
        df_raw = load_team_data(input_path)

    seasons = sorted(df_raw["season"].unique())
    print(f"  Seasons: {seasons}")
    print(f"  Teams:   {len(df_raw)} rows")
    if using_sample:
        print(f"  Mode:    SAMPLE DATA (replace with real team_stats.csv for live use)")

    # ── Check which feature columns are present ────────────────────────────
    soft_cols = [
        "efficiency_margin", "offense_rating", "defense_rating",
        "recent_form", "strength_of_schedule", "ap_rank_week6",
    ]
    present = [c for c in soft_cols if c in df_raw.columns]
    missing = [c for c in soft_cols if c not in df_raw.columns]
    print(f"\n  Feature columns present : {present}")
    if missing:
        print(f"  Missing (will be skipped): {missing}")

    # ── Rate teams ────────────────────────────────────────────────────────
    df_rated = rate_teams_by_season(df_raw)
    save_team_ratings(df_rated)

    # ── Print top-15 tables ───────────────────────────────────────────────
    _print_top15(df_rated, "team_rating",            "TOP 15  —  TEAM RATING  (game picks + upset EV)")
    _print_top15(df_rated, "champion_profile_score", "TOP 15  —  CHAMPION PROFILE  (deep run potential)")

    # ── Sample matchup probabilities ──────────────────────────────────────
    _print_matchup_section(df_rated)

    # ── Footer ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    csv_out  = PROJECT_ROOT / "data" / "processed" / "team_ratings.csv"
    json_out = PROJECT_ROOT / "data" / "processed" / "team_ratings.json"
    print(f"  Saved → {csv_out}")
    print(f"  Saved → {json_out}")
    if using_sample:
        print(f"\n  To add real data: save a CSV to data/processed/team_stats.csv")
        print(f"  or pass the path as: python3 scripts/build_team_ratings.py <path>")
    print(SEP)


if __name__ == "__main__":
    main()
