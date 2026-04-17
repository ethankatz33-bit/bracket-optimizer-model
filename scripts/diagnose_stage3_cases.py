"""
scripts/diagnose_stage3_cases.py
Diagnose why Stage 3 fails to resolve specific high-impact unmatched teams.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from lib.data_merger import normalize_name, _build_kaggle_paths

# ── Load data ─────────────────────────────────────────────────────────────────
cbb_raw = pd.read_csv(PROJECT_ROOT / "data/raw/cbb.csv")
cbb = cbb_raw[cbb_raw["SEED"].notna()].copy()
cbb["SEED"] = cbb["SEED"].astype(int)
cbb = cbb.rename(columns={
    "YEAR": "season", "TEAM": "team_name_raw",
    "SEED": "seed",   "POSTSEASON": "postseason",
    "ADJOE": "adjoe", "BARTHAG": "barthag",
})
cbb["team_name"] = cbb["team_name_raw"].apply(normalize_name)

kp = _build_kaggle_paths(
    PROJECT_ROOT / "data/raw/TourneyCompactResults.csv",
    PROJECT_ROOT / "data/raw/TourneySeeds.csv",
)

merged = pd.read_csv(PROJECT_ROOT / "data/processed/merged_team_stats.csv")
assigned_ids = set(merged["team_id"].dropna().astype(int))

# ── Diagnose ──────────────────────────────────────────────────────────────────
CASES = [
    (2018,  1, "Kansas",         "F4"),
    (2018, 11, "Loyola Chicago", "F4"),
    (2021,  2, "Houston",        "F4"),
    (2022,  2, "Duke",           "F4"),
    (2023,  5, "Miami FL",       "F4"),
]

W = 66
for season, seed, name, post in CASES:
    kp_all   = kp[(kp["season"] == season) & (kp["seed_num"] == seed)]
    kp_avail = kp_all[~kp_all["team_id"].isin(assigned_ids)]

    cbb_unmatched = merged[
        (merged["season"] == season) &
        (merged["seed"]   == seed)   &
        (merged["team_id"].isna())
    ]

    print("=" * W)
    print(f"  {season}  {name}  (seed {seed}, {post})".center(W))
    print("=" * W)
    print(f"  Total Kaggle candidates (seed={seed})  : {len(kp_all)}")
    print(f"  Available (unassigned) Kaggle          : {len(kp_avail)}")
    print(f"  Unmatched cbb rows     (seed={seed})  : {len(cbb_unmatched)}")
    print()

    print(f"  Available Kaggle IDs:")
    if kp_avail.empty:
        print("    [none]")
    else:
        print(f"    {'team_id':>8}  {'max_round':>9}  {'is_champion':>11}")
        print(f"    {'─'*8}  {'─'*9}  {'─'*11}")
        for _, r in kp_avail.sort_values("max_round", ascending=False).iterrows():
            print(f"    {int(r['team_id']):>8}  {int(r['max_round']):>9}  {str(r['is_champion']):>11}")

    print()
    print(f"  Unmatched cbb teams (seed={seed}):")
    if cbb_unmatched.empty:
        print("    [none]")
    else:
        for _, r in cbb_unmatched.iterrows():
            print(f"    {r['team_name_raw']:<28}  postseason={r['postseason']}")

    print()
