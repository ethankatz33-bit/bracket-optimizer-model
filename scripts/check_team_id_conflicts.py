"""
scripts/check_team_id_conflicts.py
Sanity-check the global team-ID mapping for 2017–2025.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

df = pd.read_csv(PROJECT_ROOT / "data/processed/merged_team_stats.csv")

W = 80

# ── Section 1: target mappings ────────────────────────────────────────────────
TARGETS = [
    (2022, "Duke"),
    (2023, "Miami FL"),
    (2021, "Houston"),
    (2018, "Loyola Chicago"),
    (2018, "Kansas"),
]

print("=" * W)
print("TARGET MAPPINGS".center(W))
print("=" * W)
print(f"  {'Yr':>4}  {'team_name_raw':<22}  {'team_name':<22}  {'team_id':>8}  match_type")
print("  " + "─" * 72)

for yr, name in TARGETS:
    rows = df[(df["season"] == yr) & (df["team_name_raw"] == name)]
    if rows.empty:
        print(f"  {yr:>4}  {name:<22}  [NOT FOUND]")
        continue
    for _, r in rows.iterrows():
        tid = int(r["team_id"]) if pd.notna(r["team_id"]) else "NaN"
        print(
            f"  {yr:>4}  {name:<22}  {r['team_name']:<22}  "
            f"{str(tid):>8}  {r['match_type']}"
        )

print()

# ── Section 2: team_id collision check ───────────────────────────────────────
print("=" * W)
print("TEAM_ID COLLISION CHECK (2017–2025)".center(W))
print("=" * W)

modern = df[(df["season"] >= 2017) & (df["team_id"].notna())].copy()
modern["team_id"] = modern["team_id"].astype(int)

conflicts = []
for tid, grp in modern.groupby("team_id"):
    names = sorted(grp["team_name"].unique())
    if len(names) > 1:
        seasons = sorted(grp["season"].unique().tolist())
        conflicts.append((int(tid), names, seasons))

if not conflicts:
    print("  No conflicts — each team_id maps to exactly one school in 2017–2025.")
else:
    print(f"  {len(conflicts)} team_id(s) mapped to multiple school names:\n")
    for tid, names, seasons in sorted(conflicts):
        print(f"  T{tid}:")
        for nm in names:
            rows_for = modern[(modern["team_id"] == tid) & (modern["team_name"] == nm)]
            yrs = sorted(rows_for["season"].unique().tolist())
            print(f"    {nm:<30}  seasons={yrs}")

print("=" * W)
