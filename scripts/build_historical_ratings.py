"""
build_historical_ratings.py
Derive per-team per-season ratings for every tournament team from 1990–2016
using only pre-tournament data (tournament game history from prior seasons).

Features computed per (target_season, team_id)
-----------------------------------------------
  offense_rating       : avg tournament points scored in prior seasons
  defense_rating       : avg tournament points allowed in prior seasons
  efficiency_margin    : avg point differential in prior seasons
  recent_form          : win rate across all games in last RECENT_WINDOW seasons
  strength_of_schedule : avg (17 − opponent_seed) / 16 faced in prior seasons
                         (1.0 = played only #1-seeds; 0.0 = only #16-seeds)

All features use seasons STRICTLY before the target season.
Teams with no prior history receive NaN → z-score fills to 0 (population mean).

Outputs
-------
  data/processed/historical_team_ratings.csv
      season, team_id, team_name, seed, offense_rating, defense_rating,
      efficiency_margin, recent_form, strength_of_schedule,
      team_rating, champion_profile_score
  data/processed/team_id_name_map.csv
      team_id, team_name   (initially "T{id}" — enrich with real names later)

Usage
-----
  python3 scripts/build_historical_ratings.py [start_year] [end_year]

  Defaults: start_year=1990, end_year=2016
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.team_ratings import rate_teams_by_season

RAW_RESULTS = PROJECT_ROOT / "data" / "raw" / "TourneyCompactResults.csv"
RAW_SEEDS   = PROJECT_ROOT / "data" / "raw" / "TourneySeeds.csv"
OUTPUT_CSV  = PROJECT_ROOT / "data" / "processed" / "historical_team_ratings.csv"
ID_MAP_CSV  = PROJECT_ROOT / "data" / "processed" / "team_id_name_map.csv"

RECENT_WINDOW = 3   # seasons of history to use for recent_form


# ════════════════════════════════════════════════════════════════════════════
# Data preparation
# ════════════════════════════════════════════════════════════════════════════

def _parse_seed_num(raw: object) -> int | None:
    m = re.search(r"(\d+)", str(raw))
    return int(m.group(1)) if m else None


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load raw CSVs and build the seed lookup dict.

    Returns
    -------
    results    : DataFrame with tournament game results
    seeds_df   : DataFrame with seed_num parsed and validated
    seed_lookup: {(season, team_id) → seed_num}
    """
    results  = pd.read_csv(RAW_RESULTS)
    seeds_df = pd.read_csv(RAW_SEEDS)

    seeds_df["seed_num"] = seeds_df["Seed"].apply(_parse_seed_num)
    seeds_df = seeds_df.dropna(subset=["seed_num"]).copy()
    seeds_df["seed_num"] = seeds_df["seed_num"].astype(int)
    seeds_df["Team"]     = seeds_df["Team"].astype(int)

    seed_lookup: dict[tuple[int, int], int] = {
        (int(row.Season), int(row.Team)): int(row.seed_num)
        for row in seeds_df.itertuples()
    }

    return results, seeds_df, seed_lookup


def _build_game_log(results: pd.DataFrame, seed_lookup: dict) -> pd.DataFrame:
    """
    Convert the compact results table into a long-format game log where each
    row represents one team's perspective on one game:

    columns: season, team_id, scored, allowed, won, opp_id, opp_seed
    """
    w = results[["Season", "Wteam", "Wscore", "Lteam", "Lscore"]].copy()
    l = results[["Season", "Lteam", "Lscore", "Wteam", "Wscore"]].copy()

    w.columns = ["season", "team_id", "scored", "opp_id", "allowed"]
    l.columns = ["season", "team_id", "scored", "opp_id", "allowed"]
    w["won"] = 1
    l["won"] = 0

    log = pd.concat([w, l], ignore_index=True)
    log["team_id"] = log["team_id"].astype(int)
    log["opp_id"]  = log["opp_id"].astype(int)

    log["opp_seed"] = [
        seed_lookup.get((s, o))
        for s, o in zip(log["season"], log["opp_id"])
    ]

    return log


# ════════════════════════════════════════════════════════════════════════════
# Feature computation
# ════════════════════════════════════════════════════════════════════════════

def _compute_features(
    target_season: int,
    game_log:      pd.DataFrame,
    seeds_df:      pd.DataFrame,
) -> pd.DataFrame:
    """
    For each team seeded in target_season, compute pre-tournament features
    from all tournament game history strictly before target_season.

    Returns a DataFrame with one row per team (may include play-in pairs
    at the same seed number — they receive separate ratings by team_id).
    """
    season_seeds = seeds_df[seeds_df["Season"] == target_season]
    if season_seeds.empty:
        return pd.DataFrame()

    prior = game_log[game_log["season"] < target_season]
    recent_cutoff = target_season - RECENT_WINDOW

    rows: list[dict] = []

    for row in season_seeds.itertuples():
        team_id  = int(row.Team)
        seed_num = int(row.seed_num)

        tg = prior[prior["team_id"] == team_id]

        if tg.empty:
            rows.append({
                "season":               target_season,
                "team_id":              team_id,
                "team_name":            f"T{team_id}",
                "seed":                 seed_num,
                "offense_rating":       float("nan"),
                "defense_rating":       float("nan"),
                "efficiency_margin":    float("nan"),
                "recent_form":          float("nan"),
                "strength_of_schedule": float("nan"),
            })
            continue

        offense    = float(tg["scored"].mean())
        defense    = float(tg["allowed"].mean())
        efficiency = float((tg["scored"] - tg["allowed"]).mean())

        recent  = tg[tg["season"] >= recent_cutoff]
        form    = float(recent["won"].mean()) if len(recent) > 0 else float("nan")

        valid_opp_seeds = tg["opp_seed"].dropna()
        if len(valid_opp_seeds) > 0:
            sos = float(((17.0 - valid_opp_seeds) / 16.0).mean())
        else:
            sos = float("nan")

        rows.append({
            "season":               target_season,
            "team_id":              team_id,
            "team_name":            f"T{team_id}",
            "seed":                 seed_num,
            "offense_rating":       round(offense, 2),
            "defense_rating":       round(defense, 2),
            "efficiency_margin":    round(efficiency, 2),
            "recent_form":          round(form, 4) if not np.isnan(form) else float("nan"),
            "strength_of_schedule": round(sos, 4) if not np.isnan(sos) else float("nan"),
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def build_historical_ratings(
    start_year: int = 1990,
    end_year:   int = 2016,
) -> pd.DataFrame:
    """
    Build and return the full historical ratings DataFrame.
    Does NOT write to disk — call main() for that.
    """
    results, seeds_df, seed_lookup = _load_data()
    game_log = _build_game_log(results, seed_lookup)

    all_frames: list[pd.DataFrame] = []
    for season in range(start_year, end_year + 1):
        frame = _compute_features(season, game_log, seeds_df)
        if not frame.empty:
            all_frames.append(frame)

    raw_df = pd.concat(all_frames, ignore_index=True)

    # Run through the ratings pipeline — normalizes within each season
    rated = rate_teams_by_season(raw_df)

    return rated


def main() -> None:
    # ── Parse args ────────────────────────────────────────────────────────
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 1990
    end_year   = int(sys.argv[2]) if len(sys.argv) > 2 else 2016

    print("=" * 70)
    print(f"{'BUILDING HISTORICAL TEAM RATINGS':^70}")
    print(f"{'Seasons: ' + str(start_year) + '–' + str(end_year):^70}")
    print("=" * 70)

    results, seeds_df, seed_lookup = _load_data()
    game_log = _build_game_log(results, seed_lookup)

    all_frames: list[pd.DataFrame] = []
    no_history_total = 0

    for season in range(start_year, end_year + 1):
        frame = _compute_features(season, game_log, seeds_df)
        if frame.empty:
            continue
        no_hist = frame["offense_rating"].isna().sum()
        no_history_total += no_hist
        all_frames.append(frame)
        print(f"  {season}: {len(frame):>2} teams  "
              f"({no_hist} with no prior history)")

    if not all_frames:
        print("No data found for requested year range.")
        return

    raw_df = pd.concat(all_frames, ignore_index=True)
    print(f"\n  Total rows: {len(raw_df)}")
    print(f"  Teams with no prior history: {no_history_total} "
          f"({no_history_total / len(raw_df):.1%})")

    # ── Run ratings pipeline ──────────────────────────────────────────────
    print("\n  Running rating model...")
    rated = rate_teams_by_season(raw_df)

    # ── Save historical_team_ratings.csv ──────────────────────────────────
    keep_cols = [
        "season", "team_id", "team_name", "seed",
        "offense_rating", "defense_rating", "efficiency_margin",
        "recent_form", "strength_of_schedule",
        "team_rating", "champion_profile_score",
    ]
    keep_cols = [c for c in keep_cols if c in rated.columns]

    out = (
        rated[keep_cols]
        .sort_values(["season", "team_rating"], ascending=[True, False])
        .round({
            "team_rating":             4,
            "champion_profile_score":  4,
            "offense_rating":          2,
            "defense_rating":          2,
            "efficiency_margin":       2,
            "recent_form":             4,
            "strength_of_schedule":    4,
        })
        .reset_index(drop=True)
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved → {OUTPUT_CSV}")

    # ── Save team_id_name_map.csv ─────────────────────────────────────────
    all_teams = seeds_df[["Team"]].drop_duplicates().copy()
    all_teams["team_id"]   = all_teams["Team"].astype(int)
    all_teams["team_name"] = "T" + all_teams["Team"].astype(str)
    id_map = (
        all_teams[["team_id", "team_name"]]
        .sort_values("team_id")
        .reset_index(drop=True)
    )
    id_map.to_csv(ID_MAP_CSV, index=False)
    print(f"  Saved → {ID_MAP_CSV}  ({len(id_map)} teams)")

    # ── Coverage summary ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  FEATURE COVERAGE")
    print("─" * 70)
    feature_cols = [
        "offense_rating", "defense_rating", "efficiency_margin",
        "recent_form", "strength_of_schedule",
    ]
    for col in feature_cols:
        if col not in out.columns:
            continue
        has = out[col].notna().sum()
        pct = has / len(out)
        print(f"  {col:<26}  {has:>4} / {len(out)}  ({pct:.0%})")

    # ── Top-5 for most recent year ────────────────────────────────────────
    last = out[out["season"] == end_year].head(5)
    print(f"\n  Top 5 teams in {end_year}:")
    print(f"  {'Team':<10}  {'Seed':>4}  {'Rating':>8}  {'Champ':>8}  "
          f"{'OffRtg':>7}  {'DefRtg':>7}  {'Margin':>7}")
    print("  " + "─" * 65)
    for _, r in last.iterrows():
        off = f"{r['offense_rating']:>7.1f}" if pd.notna(r['offense_rating']) else "    N/A"
        dfc = f"{r['defense_rating']:>7.1f}" if pd.notna(r['defense_rating']) else "    N/A"
        mar = f"{r['efficiency_margin']:>7.1f}" if pd.notna(r['efficiency_margin']) else "    N/A"
        print(f"  {r['team_name']:<10}  {r['seed']:>4}  "
              f"{r['team_rating']:>8.4f}  {r['champion_profile_score']:>8.4f}  "
              f"{off}  {dfc}  {mar}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
