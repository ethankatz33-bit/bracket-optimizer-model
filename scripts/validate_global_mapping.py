"""
scripts/validate_global_mapping.py
Validate the global per-(season, seed) team-ID mapping for 2017–2025.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

df = pd.read_csv(PROJECT_ROOT / "data/processed/merged_team_stats.csv")

W = 72

# ── Section 1: target teams ───────────────────────────────────────────────────
TARGETS = [
    (2018, "Kansas"),
    (2018, "Loyola Chicago"),
    (2021, "Houston"),
    (2022, "Duke"),
    (2023, "Miami FL"),
]

print("=" * W)
print("SECTION 1: SAMPLE RESOLVED MATCHES".center(W))
print("=" * W)
print(f"  {'Yr':>4}  {'Team':<22}  {'Seed':>4}  {'Post':<10}  {'team_id':>8}  match_type")
print("  " + "─" * 66)

for yr, name in TARGETS:
    row = df[(df["season"] == yr) & (df["team_name_raw"] == name)]
    if row.empty:
        print(f"  {yr:>4}  {name:<22}  [NOT FOUND IN MERGED]")
        continue
    r = row.iloc[0]
    tid = int(r["team_id"]) if pd.notna(r["team_id"]) else "NaN"
    print(
        f"  {yr:>4}  {name:<22}  {int(r['seed']):>4}  "
        f"{r['postseason']:<10}  {str(tid):>8}  {r['match_type']}"
    )

print()

# ── Section 2: additional RANK_MATCH examples ─────────────────────────────────
sample = (
    df[
        (df["season"] >= 2017) &
        (df["match_type"] == "RANK_MATCH") &
        (df["postseason"].isin(["F4", "E8", "S16"]))
    ]
    .sort_values(["season", "postseason", "seed"])
    .head(10)
)

print("=" * W)
print("SECTION 2: ADDITIONAL RANK_MATCH EXAMPLES (2017–2025)".center(W))
print("=" * W)
print(f"  {'Yr':>4}  {'Team':<26}  {'Seed':>4}  {'Post':<6}  {'team_id':>8}")
print("  " + "─" * 56)

for _, r in sample.iterrows():
    print(
        f"  {int(r['season']):>4}  {r['team_name_raw']:<26}  {int(r['seed']):>4}  "
        f"{r['postseason']:<6}  {int(r['team_id']):>8}"
    )

print()
print("=" * W)
total_nan = df["team_id"].isna().sum()
non_r68   = df[(df["team_id"].isna()) & (df["postseason"] != "R68")]
hi_impact = non_r68[non_r68["postseason"].isin({"S16", "E8", "F4", "2ND", "Champions"})]
print(f"  Total NaN team_id          : {total_nan}")
print(f"  Non-R68 NaN                : {len(non_r68)}")
print(f"  High-impact NaN (S16+)     : {len(hi_impact)}")
print(f"  High-impact NaN 2017–2025  : {len(hi_impact[hi_impact['season'] >= 2017])}")
print("=" * W)
