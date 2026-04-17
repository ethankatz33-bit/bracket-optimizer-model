"""
scripts/high_impact_nan.py
Identify high-impact unresolved teams (S16 or deeper) with NaN team_id.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

df = pd.read_csv(PROJECT_ROOT / "data/processed/merged_team_stats.csv")

# ── Filter ─────────────────────────────────────────────────────────────────────
non_r68 = df[(df["team_id"].isna()) & (df["postseason"] != "R68")].copy()

HIGH_IMPACT = {"S16", "E8", "F4", "2ND", "Champions"}
hi = non_r68[non_r68["postseason"].isin(HIGH_IMPACT)].copy()

# ── Counts ─────────────────────────────────────────────────────────────────────
print(f"  Total non-R68 NaN rows : {len(non_r68)}")
print(f"  High-impact (S16+)     : {len(hi)}")
print(f"  of which 2017–2025     : {len(hi[hi['season'] >= 2017])}")
print()

# ── Sort ───────────────────────────────────────────────────────────────────────
POST_ORDER = {"Champions": 0, "2ND": 1, "F4": 2, "E8": 3, "S16": 4}
hi["_ord"] = hi["postseason"].map(POST_ORDER)
hi = hi.sort_values(["season", "_ord", "seed"]).reset_index(drop=True)

# ── Table ──────────────────────────────────────────────────────────────────────
W = 58
print("=" * W)
print("HIGH-IMPACT UNRESOLVED (S16 or deeper)".center(W))
print("=" * W)
print(f"  {'Yr':>4}  {'Team':<28}  {'Seed':>4}  Post")
print("  " + "─" * 54)
for _, r in hi.iterrows():
    print(f"  {int(r['season']):>4}  {r['team_name_raw']:<28}  {int(r['seed']):>4}  {r['postseason']}")
print("=" * W)
