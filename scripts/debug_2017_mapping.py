"""
scripts/debug_2017_mapping.py
Diagnose the gap between the existing name->ID map and 2017 TourneySeeds IDs.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from lib.data_merger import normalize_name

# ── Load ──────────────────────────────────────────────────────────────────────
merged  = pd.read_csv(PROJECT_ROOT / "data/processed/merged_team_stats.csv")
seeds   = pd.read_csv(PROJECT_ROOT / "data/raw/TourneySeeds.csv")
cbb     = pd.read_csv(PROJECT_ROOT / "data/raw/cbb.csv")

# ── Build base name→ID map from 2013–2016 confirmed matches ──────────────────
name_to_id: dict[str, int] = {}
for _, r in merged[merged["team_id"].notna()].iterrows():
    name_to_id[r["team_name"]] = int(r["team_id"])

# ── 2017 cbb tournament teams ─────────────────────────────────────────────────
cbb17 = cbb[(cbb["YEAR"] == 2017) & cbb["SEED"].notna()].copy()
cbb17["team_name"] = cbb17["TEAM"].apply(normalize_name)
cbb17["seed_int"]  = cbb17["SEED"].astype(int)
cbb17 = cbb17.sort_values(["seed_int", "ADJOE"], ascending=[True, False]).reset_index(drop=True)

# ── 2017 TourneySeeds ─────────────────────────────────────────────────────────
seeds17 = seeds[seeds["Season"] == 2017].copy()
seeds17["seed_num"] = seeds17["Seed"].str.extract(r"(\d+)").astype(int)
all_ids17 = set(seeds17["Team"].astype(int))
mapped_ids = set(name_to_id.values())

already_assigned = all_ids17 & mapped_ids
unassigned_ids   = all_ids17 - mapped_ids
missing_cbb      = [r["team_name"] for _, r in cbb17.iterrows() if r["team_name"] not in name_to_id]

# ── Report ────────────────────────────────────────────────────────────────────
W = 64
print("=" * W)
print("2017 TEAM-ID MAPPING DIAGNOSTIC".center(W))
print("=" * W)
print(f"  Base name→ID map entries (from 2013–2016): {len(name_to_id)}")
print(f"  cbb 2017 tournament teams:                 {len(cbb17)}")
print(f"  TourneySeeds 2017 entries:                 {len(seeds17)}")
print()
print(f"  Seeds17 IDs already in map:  {len(already_assigned)}")
print(f"  Seeds17 IDs unassigned:      {len(unassigned_ids)}")
print(f"  cbb17 teams with no ID:      {len(missing_cbb)}")

print()
print("─" * W)
print("  ALL cbb 2017 TEAMS  (seed | normalized name | mapped?)")
print("─" * W)
for _, r in cbb17.iterrows():
    nm   = r["team_name"]
    sid  = int(r["seed_int"])
    post = r["POSTSEASON"]
    tid  = name_to_id.get(nm, "—")
    mark = "✓" if nm in name_to_id else "✗ MISSING"
    print(f"  seed {sid:>2}  {nm:<30}  {str(tid):<6}  {post:<10}  {mark}")

print()
print("─" * W)
print("  UNASSIGNED TourneySeeds 2017 IDs")
print("─" * W)
for _, r in seeds17[seeds17["Team"].isin(unassigned_ids)].sort_values("Seed").iterrows():
    print(f"  {r['Seed']:<6}  team_id={int(r['Team'])}")
