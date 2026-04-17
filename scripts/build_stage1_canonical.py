"""
scripts/build_stage1_canonical.py
Build a Stage-1-only canonical name → team_id mapping and assess coverage.
"""
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from lib.data_merger import normalize_name, _build_kaggle_paths, _POSTSEASON_TO_ROUND

# ── Load data ─────────────────────────────────────────────────────────────────
cbb = pd.read_csv(PROJECT_ROOT / "data/raw/cbb.csv")
cbb = cbb[cbb["SEED"].notna()].copy()
cbb["SEED"] = cbb["SEED"].astype(int)
cbb = cbb.rename(columns={
    "YEAR": "season", "TEAM": "team_name_raw",
    "SEED": "seed",   "POSTSEASON": "postseason",
    "ADJOE": "adjoe",  "BARTHAG": "barthag",
})
cbb["team_name"] = cbb["team_name_raw"].apply(normalize_name)

kp = _build_kaggle_paths(
    PROJECT_ROOT / "data/raw/TourneyCompactResults.csv",
    PROJECT_ROOT / "data/raw/TourneySeeds.csv",
)

# ── Stage-1-only matching ─────────────────────────────────────────────────────
confirmed: dict[tuple, int] = {}

for season, s_cbb in cbb.groupby("season"):
    season = int(season)
    s_kp = kp[kp["season"] == season]
    if s_kp.empty:
        continue

    for postseason, ps in s_cbb.groupby("postseason"):
        rr = _POSTSEASON_TO_ROUND.get(postseason)
        if rr is None:
            continue

        for seed_val, sg in ps.groupby("seed"):
            # Round-6: seed-free matching
            if rr == 6:
                is_c = (postseason == "Champions")
                cands = s_kp[
                    (s_kp["max_round"] == 6) & (s_kp["is_champion"] == is_c)
                ]
            else:
                cands = s_kp[
                    (s_kp["seed_num"] == seed_val) & (s_kp["max_round"] == rr)
                ]

            if len(sg) == 1 and len(cands) == 1:
                key = (season, sg.iloc[0]["team_name"])
                confirmed[key] = int(cands.iloc[0]["team_id"])

# ── Build canonical mapping ───────────────────────────────────────────────────
name_to_ids: dict[str, set[int]] = defaultdict(set)
for (s, nm), tid in confirmed.items():
    name_to_ids[nm].add(tid)

canonical  = {nm: next(iter(ids)) for nm, ids in name_to_ids.items() if len(ids) == 1}
ambiguous  = {nm: sorted(ids)     for nm, ids in name_to_ids.items() if len(ids) > 1}

# ── Summary ───────────────────────────────────────────────────────────────────
W = 64
print("=" * W)
print("STAGE-1 CANONICAL BUILD SUMMARY".center(W))
print("=" * W)
print(f"  Total Stage-1 confirmed matches : {len(confirmed)}")
print(f"  Unique team names               : {len(name_to_ids)}")
print(f"  Canonical (1 ID per name)       : {len(canonical)}")
print(f"  Ambiguous (multiple IDs)        : {len(ambiguous)}")

if ambiguous:
    print()
    print("  Ambiguous names (same name → multiple team_ids):")
    for nm, ids in sorted(ambiguous.items()):
        seasons = sorted(s for (s, n) in confirmed if n == nm)
        print(f"    {nm:<28}  IDs={ids}  seasons={seasons}")

# ── Key team coverage ─────────────────────────────────────────────────────────
KEY_TEAMS = [
    "kansas",
    "duke",
    "houston",
    "loyola chicago",
    "miami florida",
    "villanova",
    "north carolina",
    "gonzaga",
]

print()
print("=" * W)
print("KEY TEAM COVERAGE".center(W))
print("=" * W)
print(f"  {'Team':<28}  {'canonical_id':>12}  {'confirmed_seasons'}")
print("  " + "─" * 58)

for t in KEY_TEAMS:
    tid = canonical.get(t)
    seasons = sorted(s for (s, n) in confirmed if n == t)
    print(f"  {t:<28}  {str(tid):>12}  {seasons}")

print("=" * W)
