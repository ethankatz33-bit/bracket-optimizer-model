"""
scripts/check_2018_seed1.py
Cross-reference 2018 seed-1 teams across TourneySeeds, TourneyCompactResults,
and ncaa_tournament_games.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

seeds   = pd.read_csv(PROJECT_ROOT / "data/raw/TourneySeeds.csv")
results = pd.read_csv(PROJECT_ROOT / "data/raw/TourneyCompactResults.csv")
games   = pd.read_csv(PROJECT_ROOT / "data/raw/ncaa_tournament_games.csv")

W = 72

# ── 1. 2018 seed-1 teams from TourneySeeds ────────────────────────────────────
print("=" * W)
print("2018 SEED-1 TEAMS  (TourneySeeds.csv)".center(W))
print("=" * W)
seed1_2018 = seeds[(seeds["Season"] == 2018) & (seeds["Seed"].astype(str).str.strip() == "1")]
print(f"  {'Seed':<8}  {'TeamID':>8}")
print("  " + "─" * 20)
for _, r in seed1_2018.iterrows():
    print(f"  {r['Seed']:<8}  {int(r['Team']):>8}")

# ── 2. All round_name values for 2018 ────────────────────────────────────────
print()
print("=" * W)
print("2018 ROUND NAMES  (ncaa_tournament_games.csv)".center(W))
print("=" * W)
rounds_2018 = (
    games[games["year"] == 2018][["round", "round_name"]]
    .drop_duplicates()
    .sort_values("round")
)
for _, r in rounds_2018.iterrows():
    print(f"  round={int(r['round'])}  {r['round_name']}")

# ── 3. F4 (round 5) and championship (round 6) for 2018 ──────────────────────
print()
print("=" * W)
print("2018 FINAL FOUR & CHAMPIONSHIP GAMES".center(W))
print("=" * W)
late_2018 = games[(games["year"] == 2018) & (games["round"].isin([5, 6]))].sort_values("round")
print(f"  {'Rd':>3}  {'round_name':<22}  {'W_team':>8}  {'L_team':>8}  {'W_seed':>6}  {'L_seed':>6}")
print("  " + "─" * 62)
for _, r in late_2018.iterrows():
    print(
        f"  {int(r['round']):>3}  {r['round_name']:<22}  "
        f"{int(r['winning_team']):>8}  {int(r['losing_team']):>8}  "
        f"{int(r['winning_seed']):>6}  {int(r['losing_seed']):>6}"
    )

print("=" * W)
