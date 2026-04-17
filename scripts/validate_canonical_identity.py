"""
scripts/validate_canonical_identity.py
Validates the canonical_team_name identity system in merged_team_stats.csv.

Checks:
  1. Total unique canonical teams (overall + by season)
  2. Duplicate canonical names within the same season (should be 0)
  3. 1 canonical_team_name → multiple team_name_raw values (expected: some)
  4. 1 team_name_raw → multiple canonical_team_name values (should be 0)
  5. Sample rows before vs after (raw name vs canonical)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

df = pd.read_csv(PROJECT_ROOT / "data/processed/merged_team_stats.csv")

W = 72

# ── 1. Overall counts ─────────────────────────────────────────────────────────
total_rows = len(df)
total_unique_canonical = df["canonical_team_name"].nunique()
total_unique_raw       = df["team_name_raw"].nunique()
total_seasons          = df["season"].nunique()

print("=" * W)
print("CANONICAL IDENTITY VALIDATION".center(W))
print("=" * W)
print(f"  Total rows                     : {total_rows}")
print(f"  Total unique canonical names   : {total_unique_canonical}")
print(f"  Total unique raw names         : {total_unique_raw}")
print(f"  Total seasons                  : {total_seasons} ({df['season'].min()}–{df['season'].max()})")

# ── 2. Unique canonical teams by season ──────────────────────────────────────
print()
print("=" * W)
print("UNIQUE CANONICAL TEAMS BY SEASON".center(W))
print("=" * W)
print(f"  {'Season':>6}  {'canonical':>9}  {'raw':>6}  {'rows':>5}")
print("  " + "─" * 32)
for season in sorted(df["season"].unique()):
    s = df[df["season"] == season]
    print(f"  {int(season):>6}  {s['canonical_team_name'].nunique():>9}  "
          f"{s['team_name_raw'].nunique():>6}  {len(s):>5}")

# ── 3. Duplicate canonical names within the same season ───────────────────────
print()
print("=" * W)
print("DUPLICATE CANONICAL NAMES WITHIN A SEASON (should be 0)".center(W))
print("=" * W)

dups = (
    df.groupby(["season", "canonical_team_name"])
      .size()
      .reset_index(name="count")
)
dups = dups[dups["count"] > 1].sort_values(["season", "canonical_team_name"])

if dups.empty:
    print("  None — no duplicate canonical names within any season.")
else:
    print(f"  Found {len(dups)} duplicate (season, canonical_team_name) pairs:\n")
    print(f"  {'Season':>6}  {'canonical_team_name':<30}  {'count':>5}")
    print("  " + "─" * 46)
    for _, r in dups.iterrows():
        print(f"  {int(r['season']):>6}  {r['canonical_team_name']:<30}  {int(r['count']):>5}")

# ── 4. One canonical name → multiple raw names ────────────────────────────────
print()
print("=" * W)
print("1 CANONICAL → MULTIPLE RAW NAMES (expected: some aliases)".center(W))
print("=" * W)

canon_to_raw = (
    df.groupby("canonical_team_name")["team_name_raw"]
      .apply(lambda x: sorted(x.unique().tolist()))
      .reset_index()
)
canon_multi = canon_to_raw[canon_to_raw["team_name_raw"].apply(len) > 1]

print(f"  canonical names with 2+ raw names: {len(canon_multi)}")
if not canon_multi.empty:
    print()
    print(f"  {'canonical_team_name':<30}  raw_names")
    print("  " + "─" * 68)
    for _, r in canon_multi.iterrows():
        raws = r["team_name_raw"]
        print(f"  {r['canonical_team_name']:<30}  {raws[0]}")
        for nm in raws[1:]:
            print(f"  {'':<30}  {nm}")

# ── 5. One raw name → multiple canonical names (should be 0) ──────────────────
print()
print("=" * W)
print("1 RAW NAME → MULTIPLE CANONICAL NAMES (should be 0)".center(W))
print("=" * W)

raw_to_canon = (
    df.groupby("team_name_raw")["canonical_team_name"]
      .apply(lambda x: sorted(x.unique().tolist()))
      .reset_index()
)
raw_multi = raw_to_canon[raw_to_canon["canonical_team_name"].apply(len) > 1]

print(f"  raw names with 2+ canonical names: {len(raw_multi)}")
if not raw_multi.empty:
    print()
    print(f"  {'team_name_raw':<30}  canonical_names")
    print("  " + "─" * 68)
    for _, r in raw_multi.iterrows():
        canons = r["canonical_team_name"]
        print(f"  {r['team_name_raw']:<30}  {canons[0]}")
        for nm in canons[1:]:
            print(f"  {'':<30}  {nm}")

# ── 6. Sample rows before vs after ───────────────────────────────────────────
print()
print("=" * W)
print("SAMPLE ROWS: raw name vs canonical name".center(W))
print("=" * W)
SAMPLE_TEAMS = [
    (2018, "Kansas"),
    (2018, "Loyola Chicago"),
    (2021, "Houston"),
    (2022, "Duke"),
    (2023, "Miami FL"),
    (2019, "Virginia"),
    (2016, "Villanova"),
]
print(f"  {'Year':>4}  {'team_name_raw':<25}  {'canonical_team_name':<25}  {'team_id':>8}  match_type")
print("  " + "─" * 72)
for yr, raw in SAMPLE_TEAMS:
    rows = df[(df["season"] == yr) & (df["team_name_raw"] == raw)]
    if rows.empty:
        print(f"  {yr:>4}  {raw:<25}  [NOT FOUND]")
    else:
        r = rows.iloc[0]
        tid = int(r["team_id"]) if pd.notna(r["team_id"]) else "NaN"
        print(f"  {yr:>4}  {r['team_name_raw']:<25}  {r['canonical_team_name']:<25}  "
              f"{str(tid):>8}  {r['match_type']}")

# ── 7. canonical_team_name == team_name check ─────────────────────────────────
print()
print("=" * W)
print("canonical_team_name == team_name (consistency check)".center(W))
print("=" * W)
if "team_name" in df.columns:
    mismatch = df[df["canonical_team_name"] != df["team_name"]]
    print(f"  Rows where canonical_team_name != team_name: {len(mismatch)}")
    if not mismatch.empty:
        print()
        for _, r in mismatch.head(10).iterrows():
            print(f"  {int(r['season'])}  raw={r['team_name_raw']!r}  "
                  f"canonical={r['canonical_team_name']!r}  team_name={r['team_name']!r}")
else:
    print("  team_name column not present — skipping.")

print("=" * W)
print("VALIDATION COMPLETE".center(W))
print("=" * W)
