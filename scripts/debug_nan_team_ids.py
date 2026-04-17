"""
scripts/debug_nan_team_ids.py
Diagnose every row in merged_team_stats.csv where team_id is NaN.

For each unmatched row, classify the root cause by comparing the
(season, seed, postseason) group sizes in cbb.csv vs the Kaggle
tournament-path table.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from lib.data_merger import normalize_name, _build_kaggle_paths, _POSTSEASON_TO_ROUND

# ── Load data ─────────────────────────────────────────────────────────────────
merged = pd.read_csv(PROJECT_ROOT / "data/processed/merged_team_stats.csv")
nan_rows = merged[merged["team_id"].isna()].copy()

cbb_raw = pd.read_csv(PROJECT_ROOT / "data/raw/cbb.csv")
cbb = cbb_raw[cbb_raw["SEED"].notna()].copy()
cbb["SEED"] = cbb["SEED"].astype(int)
cbb = cbb.rename(columns={
    "YEAR": "season", "TEAM": "team_name_raw",
    "SEED": "seed",   "POSTSEASON": "postseason",
    "ADJOE": "adjoe", "ADJDE": "adjde", "BARTHAG": "barthag",
})
cbb["team_name"] = cbb["team_name_raw"].apply(normalize_name)

kaggle_paths = _build_kaggle_paths(
    PROJECT_ROOT / "data/raw/TourneyCompactResults.csv",
    PROJECT_ROOT / "data/raw/TourneySeeds.csv",
)

# ── Classify each NaN row ─────────────────────────────────────────────────────
cause_counts: dict[str, int] = {}
rows_out: list[dict] = []

for _, row in nan_rows.iterrows():
    season = int(row["season"])
    seed   = int(row["seed"])
    post   = str(row["postseason"])
    name   = row["team_name_raw"]

    round_reached = _POSTSEASON_TO_ROUND.get(post)

    cbb_group = cbb[
        (cbb["season"] == season) &
        (cbb["seed"]   == seed)   &
        (cbb["postseason"] == post)
    ]

    if round_reached is not None:
        kgl_group = kaggle_paths[
            (kaggle_paths["season"]    == season) &
            (kaggle_paths["seed_num"]  == seed)   &
            (kaggle_paths["max_round"] == round_reached)
        ]
    else:
        kgl_group = pd.DataFrame()

    n_cbb = len(cbb_group)
    n_kgl = len(kgl_group)

    if post == "R68":
        cause = "play-in (R68)"
    elif round_reached is None:
        cause = "unknown postseason"
    elif n_cbb == 0:
        cause = "cbb team not in season"
    elif n_kgl == 0:
        cause = "kaggle > cbb size mismatch"   # kaggle group empty → 0
    elif n_cbb > n_kgl:
        cause = "cbb > kaggle size mismatch"
    elif n_kgl > n_cbb:
        cause = "kaggle > cbb size mismatch"
    elif n_cbb == n_kgl == 1:
        cause = "1:1 no match"
    else:
        cause = f"rank-match failed (n={n_cbb})"

    cause_counts[cause] = cause_counts.get(cause, 0) + 1
    rows_out.append({
        "season": season,
        "team_name_raw": name,
        "seed": seed,
        "postseason": post,
        "cbb_cnt": n_cbb,
        "kgl_cnt": n_kgl,
        "cause": cause,
    })

# ── Print detail table ────────────────────────────────────────────────────────
W = 88
print("=" * W)
print("NaN TEAM_ID DETAIL".center(W))
print("=" * W)
print(f"{'Yr':>4}  {'Team':<28}  {'Seed':>4}  {'Post':<10}  {'CBB':>4}  {'KGL':>4}  Cause")
print("─" * W)

for r in rows_out:
    print(
        f"{r['season']:>4}  {r['team_name_raw']:<28}  {r['seed']:>4}  "
        f"{r['postseason']:<10}  {r['cbb_cnt']:>4}  {r['kgl_cnt']:>4}  {r['cause']}"
    )

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * W)
print("CAUSE SUMMARY".center(W))
print("=" * W)
print(f"  Total NaN rows : {len(nan_rows)}")
print(f"  In 2013–2016   : {sum(1 for r in rows_out if 2013 <= r['season'] <= 2016)}")
print(f"  In 2017–2025   : {sum(1 for r in rows_out if 2017 <= r['season'] <= 2025)}")
print(f"  Non-R68 (used) : {sum(1 for r in rows_out if r['postseason'] != 'R68')}")
print()
print(f"  {'Count':>5}  Cause")
print(f"  {'─'*5}  {'─'*40}")
for cause, cnt in sorted(cause_counts.items(), key=lambda x: -x[1]):
    print(f"  {cnt:>5}  {cause}")
print("=" * W)
