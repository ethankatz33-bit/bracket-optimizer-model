"""
scripts/debug_2017_seed_gap.py
Break down the 18-missing-cbb vs 19-unassigned-IDs discrepancy by seed,
splitting direct vs play-in slots.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from lib.data_merger import normalize_name

# ── Load ──────────────────────────────────────────────────────────────────────
merged = pd.read_csv(PROJECT_ROOT / "data/processed/merged_team_stats.csv")
seeds  = pd.read_csv(PROJECT_ROOT / "data/raw/TourneySeeds.csv")
cbb    = pd.read_csv(PROJECT_ROOT / "data/raw/cbb.csv")

# ── Build name→ID map ─────────────────────────────────────────────────────────
name_to_id: dict[str, int] = {}
for _, r in merged[merged["team_id"].notna()].iterrows():
    name_to_id[r["team_name"]] = int(r["team_id"])

# ── 2017 TourneySeeds: parse seed slot ───────────────────────────────────────
seeds17 = seeds[seeds["Season"] == 2017].copy()
seeds17["seed_num"]  = seeds17["Seed"].str.extract(r"(\d+)").astype(int)
seeds17["is_playin"] = seeds17["Seed"].str.len() > 3   # e.g. "W11a" vs "W11"
seeds17["team_id"]   = seeds17["Team"].astype(int)

mapped_ids    = set(name_to_id.values())
unassigned    = seeds17[~seeds17["team_id"].isin(mapped_ids)].copy()

# ── 2017 cbb tournament teams ─────────────────────────────────────────────────
cbb17 = cbb[(cbb["YEAR"] == 2017) & cbb["SEED"].notna()].copy()
cbb17["team_name"] = cbb17["TEAM"].apply(normalize_name)
cbb17["seed_int"]  = cbb17["SEED"].astype(int)
missing_cbb = cbb17[~cbb17["team_name"].isin(name_to_id)].copy()

# ── Summary ───────────────────────────────────────────────────────────────────
W = 72
print("=" * W)
print("2017 SEED-GAP DIAGNOSTIC".center(W))
print("=" * W)
print(f"  Total unassigned TourneySeeds17 IDs : {len(unassigned)}")
print(f"  Total missing cbb17 teams           : {len(missing_cbb)}")
print(f"  Net gap                             : {len(unassigned) - len(missing_cbb):+d}")

# ── Unassigned IDs by seed (direct vs play-in) ────────────────────────────────
print()
print("─" * W)
print("  UNASSIGNED TourneySeeds17 IDs  (seed | slot | play-in? | team_id)")
print("─" * W)
for _, r in unassigned.sort_values(["seed_num", "Seed"]).iterrows():
    pi = "play-in" if r["is_playin"] else "direct "
    print(f"  seed {r['seed_num']:>2}  {r['Seed']:<6}  {pi}  id={r['team_id']}")

print()
print(f"  Direct slots unassigned : {(~unassigned['is_playin']).sum()}")
print(f"  Play-in slots unassigned: {unassigned['is_playin'].sum()}")

# ── Missing cbb teams by seed ─────────────────────────────────────────────────
print()
print("─" * W)
print("  MISSING cbb17 TEAMS  (seed | normalized name | postseason)")
print("─" * W)
for _, r in missing_cbb.sort_values("seed_int").iterrows():
    print(f"  seed {r['seed_int']:>2}  {r['team_name']:<32}  {r['POSTSEASON']}")

print()
print(f"  cbb17 missing count: {len(missing_cbb)}")

# ── Cross-reference: play-in IDs with no cbb match ───────────────────────────
playin_unassigned = unassigned[unassigned["is_playin"]]
direct_unassigned = unassigned[~unassigned["is_playin"]]

print()
print("─" * W)
print("  ANALYSIS: Are direct-slot IDs all explainable by missing cbb names?")
print("─" * W)
print(f"  Direct unassigned: {len(direct_unassigned)}")
print(f"  Missing cbb teams: {len(missing_cbb)}")
if len(direct_unassigned) == len(missing_cbb):
    print("  ✓ Counts match — play-in slots account for the extra unassigned ID(s)")
else:
    diff = len(direct_unassigned) - len(missing_cbb)
    print(f"  Gap = {diff:+d} — direct unassigned vs missing cbb still mismatched")

# Show all play-in seeds for context
print()
print("─" * W)
print("  ALL play-in slots in 2017 TourneySeeds (for reference)")
print("─" * W)
playin_all = seeds17[seeds17["is_playin"]].sort_values("Seed")
for _, r in playin_all.iterrows():
    in_map = "✓ mapped" if r["team_id"] in mapped_ids else "✗ unassigned"
    print(f"  {r['Seed']:<6}  id={r['team_id']}  {in_map}")
