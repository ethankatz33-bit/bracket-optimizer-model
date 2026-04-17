"""
scripts/debug_gonzaga.py
Diagnose the Gonzaga 2017 and 2021 1:1 no-match mapping failure.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from lib.data_merger import normalize_name, _build_kaggle_paths, _POSTSEASON_TO_ROUND

# ── Load data ─────────────────────────────────────────────────────────────────
cbb_raw = pd.read_csv(PROJECT_ROOT / "data/raw/cbb.csv")
cbb = cbb_raw[cbb_raw["SEED"].notna()].copy()
cbb["SEED"] = cbb["SEED"].astype(int)
cbb = cbb.rename(columns={
    "YEAR": "season", "TEAM": "team_name_raw",
    "SEED": "seed",   "POSTSEASON": "postseason",
    "ADJOE": "adjoe", "ADJDE": "adjde", "BARTHAG": "barthag",
})
cbb["team_name"] = cbb["team_name_raw"].apply(normalize_name)

kaggle_paths = _build_kaggle_paths(
    PROJECT_ROOT / "data/raw/TourneyCompactResults.csv",
    PROJECT_ROOT / "data/raw/TourneySeeds.csv",
)

# ── Inspect each year ─────────────────────────────────────────────────────────
for year in [2017, 2021]:
    print(f"=== Gonzaga {year} ===")
    print()

    cbb_row = cbb[(cbb["season"] == year) & (cbb["team_name"] == "gonzaga")]

    if cbb_row.empty:
        print("  [ERROR] No cbb row found for Gonzaga in this year.")
        print()
        continue

    row = cbb_row.iloc[0]
    season   = int(row["season"])
    seed     = int(row["seed"])
    postseason = row["postseason"]
    derived_round = _POSTSEASON_TO_ROUND.get(str(postseason))

    print(f"  cbb row:")
    print(f"    season     = {season}")
    print(f"    team_name_raw = {row['team_name_raw']!r}")
    print(f"    seed       = {seed}")
    print(f"    postseason = {postseason!r}")
    print(f"    derived_round (via _POSTSEASON_TO_ROUND) = {derived_round}")
    print()

    # All Kaggle candidates for this season + seed (any round)
    kp_all = kaggle_paths[
        (kaggle_paths["season"]   == season) &
        (kaggle_paths["seed_num"] == seed)
    ].copy()

    print(f"  All Kaggle candidates (season={season}, seed={seed}):")
    if kp_all.empty:
        print("    [none]")
    else:
        print(f"    {'team_id':>8}  {'max_round':>9}  {'is_champion':>11}")
        print(f"    {'─'*8}  {'─'*9}  {'─'*11}")
        for _, r in kp_all.iterrows():
            print(f"    {int(r['team_id']):>8}  {int(r['max_round']):>9}  {str(r['is_champion']):>11}")
    print()

    # Which rows match derived_round
    kp_match = kp_all[kp_all["max_round"] == derived_round] if derived_round is not None else pd.DataFrame()

    print(f"  Matching derived_round={derived_round}: {len(kp_match)} of {len(kp_all)} candidates")
    if not kp_match.empty:
        for _, r in kp_match.iterrows():
            print(f"    team_id={int(r['team_id'])}  max_round={int(r['max_round'])}  is_champion={r['is_champion']}")
    else:
        print("    [no match — this is the 1:1 miss]")
    print()
