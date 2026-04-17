"""
scripts/build_max_round_table.py
Build a clean (season, team_id) → max_round table from TourneyCompactResults.csv.

Round definitions (Daynum → round):
  134/135 → 0  First Four (play-in, 2011+)
  136/137 → 1  Round of 64
  138/139 → 2  Round of 32
  143/144 → 3  Sweet 16
  145/146 → 4  Elite 8
  152     → 5  Final Four
  154     → 6  Championship
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

results = pd.read_csv(PROJECT_ROOT / "data/raw/TourneyCompactResults.csv")
seeds   = pd.read_csv(PROJECT_ROOT / "data/raw/TourneySeeds.csv")

# ── Daynum → round mapping ────────────────────────────────────────────────────
DAYNUM_ROUND: dict[int, int] = {
    134: 0, 135: 0,   # First Four (play-in)
    136: 1, 137: 1,   # Round of 64
    138: 2, 139: 2,   # Round of 32
    143: 3, 144: 3,   # Sweet 16
    145: 4, 146: 4,   # Elite 8
    152: 5,           # Final Four
    154: 6,           # Championship
}

# Verify every Daynum in the dataset is covered
unknown_days = set(results["Daynum"].unique()) - set(DAYNUM_ROUND)
if unknown_days:
    print(f"WARNING: unmapped Daynum values: {sorted(unknown_days)}")

results["round"] = results["Daynum"].map(DAYNUM_ROUND)

# ── Build (season, team_id) → max_round ──────────────────────────────────────
# Both winners and losers "reached" the round they played in.
# A team's max_round is the highest round in which they appeared.
# (Winners appear in higher rounds as they advance; losers stop at this round.)

winners = results[["Season", "Wteam", "round"]].rename(
    columns={"Season": "season", "Wteam": "team_id"}
)
losers  = results[["Season", "Lteam", "round"]].rename(
    columns={"Season": "season", "Lteam": "team_id"}
)

all_appearances = pd.concat([winners, losers], ignore_index=True)
max_round_df = (
    all_appearances
    .groupby(["season", "team_id"], as_index=False)["round"]
    .max()
    .rename(columns={"round": "max_round"})
)

W = 72

# ── Verification: 2018 champion and F4 losers ─────────────────────────────────
print("=" * W)
print("VERIFICATION: 2018 champion & Final Four".center(W))
print("=" * W)

mr2018 = max_round_df[max_round_df["season"] == 2018]

champ = mr2018[mr2018["max_round"] == 6]
f4    = mr2018[mr2018["max_round"] == 5]

print(f"\n  2018 champion  (max_round=6):")
for _, r in champ.iterrows():
    print(f"    team_id={int(r['team_id'])}  max_round={int(r['max_round'])}")

print(f"\n  2018 Final Four losers  (max_round=5):")
for _, r in f4.iterrows():
    print(f"    team_id={int(r['team_id'])}  max_round={int(r['max_round'])}")

# ── 2018 seed-1 teams with max_round ─────────────────────────────────────────
print()
print("=" * W)
print("2018 SEED-1 TEAMS WITH MAX_ROUND".center(W))
print("=" * W)

seed1_2018 = seeds[(seeds["Season"] == 2018) & (seeds["Seed"].astype(str).str.strip() == "1")]

print(f"\n  {'TeamID':>8}  {'max_round':>9}  note")
print("  " + "─" * 40)

for _, s in seed1_2018.iterrows():
    tid = int(s["Team"])
    row = mr2018[mr2018["team_id"] == tid]
    mr  = int(row["max_round"].iloc[0]) if not row.empty else None
    note = ""
    if mr == 6:
        note = "← champion"
    elif mr == 5:
        note = "← Final Four (Kansas candidate)"
    elif mr is not None and mr <= 4:
        note = f"← eliminated round {mr}"
    print(f"  {tid:>8}  {str(mr):>9}  {note}")

# ── All F4 teams by seed in 2018 ──────────────────────────────────────────────
print()
print("=" * W)
print("2018 ALL FINAL FOUR TEAMS (max_round >= 5) WITH SEED".center(W))
print("=" * W)

f4_and_champ = mr2018[mr2018["max_round"] >= 5].copy()
f4_and_champ = f4_and_champ.merge(
    seeds[seeds["Season"] == 2018].rename(columns={"Season": "season", "Team": "team_id"}),
    on=["season", "team_id"],
    how="left",
)

print(f"\n  {'TeamID':>8}  {'Seed':>5}  {'max_round':>9}")
print("  " + "─" * 30)
for _, r in f4_and_champ.sort_values("max_round", ascending=False).iterrows():
    seed_val = int(r["Seed"]) if pd.notna(r.get("Seed")) else "?"
    print(f"  {int(r['team_id']):>8}  {str(seed_val):>5}  {int(r['max_round']):>9}")

OUT = PROJECT_ROOT / "data" / "processed" / "max_round_table.csv"
max_round_df.to_csv(OUT, index=False)

print()
print("=" * W)
print(f"  Total (season, team_id) rows in max_round table: {len(max_round_df)}")
print(f"  Seasons covered: {max_round_df['season'].min()} – {max_round_df['season'].max()}")
print(f"  Saved → {OUT}")
print("=" * W)
