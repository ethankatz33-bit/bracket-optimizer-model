"""
scripts/check_2021_postseason_groups.py
Compare 2021 cbb postseason groups against Kaggle max_round pool sizes.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from lib.data_merger import _build_kaggle_paths, _POSTSEASON_TO_ROUND

kp = _build_kaggle_paths(
    PROJECT_ROOT / "data/raw/TourneyCompactResults.csv",
    PROJECT_ROOT / "data/raw/TourneySeeds.csv",
)

cbb_raw = pd.read_csv(PROJECT_ROOT / "data/raw/cbb.csv")
cbb2021 = cbb_raw[(cbb_raw["YEAR"] == 2021) & cbb_raw["SEED"].notna()].copy()
kp2021  = kp[kp["season"] == 2021]

W = 66

# ── Summary counts ────────────────────────────────────────────────────────────
print("=" * W)
print("2021 DATASET SIZES".center(W))
print("=" * W)
print(f"  cbb 2021 tournament teams : {len(cbb2021)}")
print(f"  kp  2021 entries          : {len(kp2021)}")

# ── Per-postseason comparison ─────────────────────────────────────────────────
print()
print("=" * W)
print("PER-POSTSEASON GROUP: cbb vs Kaggle pool".center(W))
print("=" * W)
print(f"  {'Postseason':<12}  {'cbb_n':>5}  {'round':>5}  {'kp_pool':>8}  {'match?'}")
print("  " + "─" * 54)

for post, grp in sorted(cbb2021.groupby("POSTSEASON"),
                        key=lambda x: -_POSTSEASON_TO_ROUND.get(x[0], -1)):
    rr = _POSTSEASON_TO_ROUND.get(post)
    if rr is None:
        print(f"  {post:<12}  {len(grp):>5}  {'?':>5}  {'?':>8}  UNKNOWN POSTSEASON")
        continue

    if rr == 6:
        kp_c = len(kp2021[(kp2021["max_round"] == 6) & (kp2021["is_champion"] == True)])
        kp_r = len(kp2021[(kp2021["max_round"] == 6) & (kp2021["is_champion"] == False)])
        champ_cbb = (grp["POSTSEASON"] == "Champions").sum()
        runnr_cbb = (grp["POSTSEASON"] == "2ND").sum()
        print(f"  {post:<12}  {len(grp):>5}  {rr:>5}  {kp_c:>3}c/{kp_r}r   "
              f"{'OK' if kp_c == 1 and kp_r == 1 else 'MISMATCH'}")
    else:
        kp_n   = len(kp2021[kp2021["max_round"] == rr])
        n_cbb  = len(grp)
        status = "OK" if kp_n == n_cbb else f"MISMATCH (kp={kp_n})"
        print(f"  {post:<12}  {n_cbb:>5}  {rr:>5}  {kp_n:>8}  {status}")

# ── 2021 Daynum distribution ──────────────────────────────────────────────────
print()
print("=" * W)
print("2021 DAYNUM DISTRIBUTION  (TourneyCompactResults)".center(W))
print("=" * W)
results = pd.read_csv(PROJECT_ROOT / "data/raw/TourneyCompactResults.csv")
res2021 = results[results["Season"] == 2021]
day_counts = res2021.groupby("Daynum").size().sort_index()

DAYNUM_ROUND = {
    134: 0, 135: 0,
    136: 1, 137: 1,
    138: 2, 139: 2,
    143: 3, 144: 3,
    145: 4, 146: 4,
    152: 5,
    154: 6,
}
print(f"  {'Daynum':>7}  {'games':>6}  {'round':>6}")
print("  " + "─" * 24)
for daynum, cnt in day_counts.items():
    rnd = DAYNUM_ROUND.get(int(daynum), "?")
    print(f"  {int(daynum):>7}  {int(cnt):>6}  {str(rnd):>6}")

print()
print(f"  Total 2021 games in TourneyCompactResults: {len(res2021)}")
print("=" * W)
