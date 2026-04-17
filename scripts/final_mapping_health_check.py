"""
scripts/final_mapping_health_check.py
Health check for the team-ID mapping in merged_team_stats.csv.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

df = pd.read_csv(PROJECT_ROOT / "data/processed/merged_team_stats.csv")

W = 72

# ── 1. Total NaN team_id ──────────────────────────────────────────────────────
print("=" * W)
print("MAPPING HEALTH CHECK".center(W))
print("=" * W)
print(f"\n  Total NaN team_id          : {df['team_id'].isna().sum()} / {len(df)}")

# ── 2. Conflict detection (2017+) ─────────────────────────────────────────────
modern = df[(df["season"] >= 2017) & df["team_id"].notna()].copy()
modern["team_id"] = modern["team_id"].astype(int)

conflicts = []
for tid, grp in modern.groupby("team_id"):
    names = sorted(grp["team_name"].unique())
    if len(names) > 1:
        seasons = sorted(grp["season"].unique().tolist())
        conflicts.append((int(tid), names, seasons))
conflicts.sort()

print(f"  Conflict team_ids (2017+)  : {len(conflicts)}")

if conflicts:
    print(f"\n  First 5 conflicts:")
    print(f"  {'team_id':>8}  names → seasons")
    print("  " + "─" * 62)
    for tid, names, seasons in conflicts[:5]:
        for i, nm in enumerate(names):
            yrs = sorted(
                modern[(modern["team_id"] == tid) & (modern["team_name"] == nm)]["season"].unique().tolist()
            )
            prefix = f"  T{tid:<7}" if i == 0 else "  " + " " * 8
            print(f"  T{tid:<7}  {nm:<30}  seasons={yrs}" if i == 0
                  else f"           {nm:<30}  seasons={yrs}")

# ── 3. High-impact unresolved (S16+) ─────────────────────────────────────────
HI_POST = {"S16", "E8", "F4", "2ND", "Champions"}
hi = df[df["team_id"].isna() & df["postseason"].isin(HI_POST)].copy()
hi = hi.sort_values(["season", "postseason", "seed"])

print(f"\n  High-impact NaN (S16+)     : {len(hi)}")

if not hi.empty:
    print()
    print("=" * W)
    print("HIGH-IMPACT UNRESOLVED TEAMS (S16+, NaN team_id)".center(W))
    print("=" * W)
    print(f"  {'Season':>6}  {'team_name_raw':<28}  {'Post':<10}  {'Seed':>4}")
    print("  " + "─" * 56)
    for _, r in hi.iterrows():
        print(f"  {int(r['season']):>6}  {r['team_name_raw']:<28}  "
              f"{r['postseason']:<10}  {int(r['seed']):>4}")
else:
    print("\n  No high-impact unresolved teams.")

# ── 4. Target team status ─────────────────────────────────────────────────────
TARGETS = [
    (2018, "Kansas"),
    (2018, "Loyola Chicago"),
    (2021, "Houston"),
    (2022, "Duke"),
    (2023, "Miami FL"),
]

print()
print("=" * W)
print("TARGET TEAM STATUS".center(W))
print("=" * W)
print(f"  {'Year':>4}  {'team_name_raw':<22}  {'team_id':>8}  match_type")
print("  " + "─" * 52)

for yr, name in TARGETS:
    rows = df[(df["season"] == yr) & (df["team_name_raw"] == name)]
    if rows.empty:
        print(f"  {yr:>4}  {name:<22}  [NOT FOUND IN MERGED]")
    else:
        r   = rows.iloc[0]
        tid = int(r["team_id"]) if pd.notna(r["team_id"]) else "NaN"
        print(f"  {yr:>4}  {name:<22}  {str(tid):>8}  {r['match_type']}")

print("=" * W)
