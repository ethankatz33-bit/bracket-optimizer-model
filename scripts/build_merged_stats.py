"""
build_merged_stats.py
Build merged_team_stats.csv from external datasets.

Usage
-----
  python3 scripts/build_merged_stats.py

Inputs  (data/raw/)
------
  cbb.csv               — required  (Kaggle college basketball dataset, 2013–2023)
  kenpom_torvik.csv     — optional  (KenPom / Bart Torvik ratings)
  ap_week6.csv          — optional  (AP Poll Week 6 rankings)

Mapping strategy
----------------
  CONFIRMED  : unique tournament-path match (season, seed, round_reached, champion_flag)
  RANK_MATCH : rank-based disambiguation within ambiguous groups using offensive efficiency
  UNMATCHED  : could not resolve to a Kaggle team_id

Output  (data/processed/)
------
  merged_team_stats.csv
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from lib.data_merger import (
    load_cbb,
    load_kenpom_torvik,
    load_ap_week6,
    build_team_id_map,
    build_merged_stats,
)

RAW       = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT    = PROCESSED / "merged_team_stats.csv"

W   = 72
SEP = "=" * W
THN = "─" * W


def main() -> None:
    print(SEP)
    print(f"{'BUILD MERGED TEAM STATS':^{W}}")
    print(SEP)

    # ── Load required: cbb.csv ────────────────────────────────────────────
    cbb_path = RAW / "cbb.csv"
    if not cbb_path.exists():
        sys.exit(f"Error: {cbb_path} not found.  Place cbb.csv in data/raw/.")

    print(f"\n  Loading cbb.csv …", end=" ")
    cbb_df = load_cbb(cbb_path)
    print(f"{len(cbb_df)} tournament team-seasons  "
          f"({cbb_df['season'].min()}–{cbb_df['season'].max()})")

    # ── Load optional: kenpom_torvik.csv ──────────────────────────────────
    kt_path = RAW / "kenpom_torvik.csv"
    kt_df   = load_kenpom_torvik(kt_path)
    if kt_df is not None:
        print(f"  Loaded kenpom_torvik.csv  ({len(kt_df)} rows)")
    else:
        print(f"  kenpom_torvik.csv not found — skipping")

    # ── Load optional: ap_week6.csv ───────────────────────────────────────
    ap_path = RAW / "ap_week6.csv"
    ap_df   = load_ap_week6(ap_path)
    if ap_df is not None and not ap_df.empty:
        print(f"  Loaded ap_week6.csv  ({len(ap_df)} top-12 team entries)")
    elif ap_path.exists():
        print(f"  ap_week6.csv exists but contains no data rows — skipping")
    else:
        print(f"  ap_week6.csv not found — skipping")

    # ── Build team-ID mapping ─────────────────────────────────────────────
    print(f"\n  Building team-ID mapping …")
    confirmed, estimated, unmatched_log = build_team_id_map(
        cbb_df        = cbb_df,
        results_path  = RAW  / "TourneyCompactResults.csv",
        seeds_path    = RAW  / "TourneySeeds.csv",
        hist_path     = PROCESSED / "historical_team_ratings.csv",
    )

    n_total     = len(cbb_df)
    n_confirmed = len(confirmed)
    n_estimated = len(estimated)
    n_unmatched = len(unmatched_log)

    print(f"  {'Confirmed (unique path):':<30} {n_confirmed:>4} / {n_total}  "
          f"({n_confirmed/n_total:.1%})")
    print(f"  {'Rank-based (estimated):':<30} {n_estimated:>4} / {n_total}  "
          f"({n_estimated/n_total:.1%})")
    print(f"  {'Unmatched:':<30} {n_unmatched:>4} / {n_total}  "
          f"({n_unmatched/n_total:.1%})")
    print(f"  {'Total mapped:':<30} {n_confirmed+n_estimated:>4} / {n_total}  "
          f"({(n_confirmed+n_estimated)/n_total:.1%})")

    # ── Build merged stats ────────────────────────────────────────────────
    print(f"\n  Building merged stats …", end=" ")
    merged = build_merged_stats(
        cbb_df    = cbb_df,
        confirmed = confirmed,
        estimated = estimated,
        ap_df     = ap_df,
        kt_df     = kt_df,
    )
    print("done")

    # ── Save output ───────────────────────────────────────────────────────
    PROCESSED.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)

    # ── Summary report ────────────────────────────────────────────────────
    print(f"\n{THN}")
    print(f"  MATCH SUMMARY BY SEASON")
    print(THN)
    print(f"  {'Season':>6}  {'Total':>6}  {'Confirmed':>9}  {'RankMatch':>9}  {'Unmatched':>9}  {'Rate':>6}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*6}")

    for season, grp in merged.groupby("season"):
        n   = len(grp)
        nc  = (grp["match_type"] == "CONFIRMED").sum()
        nr  = (grp["match_type"] == "RANK_MATCH").sum()
        nu  = (grp["match_type"] == "UNMATCHED").sum()
        pct = f"{(nc+nr)/n:.0%}"
        print(f"  {season:>6}  {n:>6}  {nc:>9}  {nr:>9}  {nu:>9}  {pct:>6}")

    print(f"  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*6}")
    n   = len(merged)
    nc  = (merged["match_type"] == "CONFIRMED").sum()
    nr  = (merged["match_type"] == "RANK_MATCH").sum()
    nu  = (merged["match_type"] == "UNMATCHED").sum()
    print(f"  {'TOTAL':>6}  {n:>6}  {nc:>9}  {nr:>9}  {nu:>9}  {(nc+nr)/n:.0%}")

    # ── Feature coverage ──────────────────────────────────────────────────
    print(f"\n{THN}")
    print(f"  FEATURE COVERAGE")
    print(THN)
    feat_cols = [
        "offensive_efficiency", "defensive_efficiency",
        "efficiency_margin", "kenpom_torvik_rating",
        "ap_rank_week6", "ap_top12_flag",
    ]
    for col in feat_cols:
        if col == "ap_top12_flag":
            # ap_top12_flag = 1 for any team present in ap_week6.csv (file is the top-12 list)
            n_top12 = (merged["ap_top12_flag"] == 1).sum()
            print(f"  {col:<28}  {n_top12:>4}/{n}  ({n_top12/n:.0%})  [teams in ap_week6.csv]")
        else:
            n_present = merged[col].notna().sum()
            print(f"  {col:<28}  {n_present:>4}/{n}  ({n_present/n:.0%})")

    # ── AP Week 6 coverage detail ─────────────────────────────────────────
    print(f"\n{THN}")
    print(f"  AP WEEK 6 COVERAGE DETAIL")
    print(THN)

    ap_flagged_seasons = sorted(
        merged[merged["ap_top12_flag"] == 1]["season"].unique()
    )
    n_total_flagged = (merged["ap_top12_flag"] == 1).sum()

    if n_total_flagged == 0:
        print(f"  No AP top-12 teams loaded  (ap_week6.csv is empty).")
        print(f"  To add AP data:")
        print(f"    1. Edit data/raw/ap_week6.csv — add one row per top-12 team per season")
        print(f"    2. Re-run this script.")
    else:
        print(f"  {'Season':>6}  {'Top12 teams':>12}  {'Total teams':>12}  {'Coverage':>9}")
        print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*9}")
        for season, grp in merged.groupby("season"):
            if season not in ap_flagged_seasons:
                continue
            n_top12 = (grp["ap_top12_flag"] == 1).sum()
            n_total = len(grp)
            print(f"  {season:>6}  {n_top12:>12}  {n_total:>12}  {n_top12/n_total:>9.1%}")
        print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*9}")
        print(f"  {'TOTAL':>6}  {n_total_flagged:>12}  {n:>12}")

        # Teams in ap_week6.csv that didn't match anything in merged stats
        if ap_df is not None and not ap_df.empty:
            ap_keys     = set(zip(ap_df["season"].astype(int), ap_df["team_name"]))
            merged_keys = set(zip(merged["season"].astype(int), merged["team_name"]))
            unmatched_ap = ap_keys - merged_keys
            if unmatched_ap:
                print(f"\n  AP ENTRIES WITH NO MATCH IN MERGED STATS  ({len(unmatched_ap)})")
                print(f"  (check team name spelling — normalization may differ)")
                for season_key, name_key in sorted(unmatched_ap):
                    raw = ap_df[
                        (ap_df["season"] == season_key) &
                        (ap_df["team_name"] == name_key)
                    ]["team_name_raw"].iloc[0]
                    print(f"  {season_key}  {raw}")

    # ── Unmatched team list ────────────────────────────────────────────────
    if unmatched_log:
        print(f"\n{THN}")
        print(f"  UNMATCHED TEAMS  ({len(unmatched_log)} entries)")
        print(THN)
        for entry in sorted(unmatched_log):
            print(f"  • {entry}")
    else:
        print(f"\n  All teams matched.")

    # ── Confirmed sample ──────────────────────────────────────────────────
    print(f"\n{THN}")
    print(f"  CONFIRMED MATCH SAMPLES  (first 15)")
    print(THN)
    sample = merged[merged["match_type"] == "CONFIRMED"].head(15)
    for _, r in sample.iterrows():
        print(f"  {r['season']}  seed {r['seed']:>2}  "
              f"{r['team_name_raw']:<22}  →  T{r['team_id']:<6}  "
              f"ADJOE={r['offensive_efficiency']:.1f}  "
              f"ADJDE={r['defensive_efficiency']:.1f}")

    print(f"\n{SEP}")
    print(f"  Saved → {OUTPUT}")
    print(f"  Rows: {len(merged)}   Columns: {len(merged.columns)}")
    print(SEP)


if __name__ == "__main__":
    main()
