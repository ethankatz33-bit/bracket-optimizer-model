"""
build_ap_week6_template.py
Generate a blank ap_week6.csv template for seasons 2013–2016.

Usage
-----
  python3 scripts/build_ap_week6_template.py

Output
------
  data/raw/ap_week6.csv   (headers only — sparse top-12 list)

Format
------
  Columns: season, team_name, ap_rank_week6

  ap_week6.csv is a SPARSE list — only AP top-12 teams at Week 6 are listed.
  Presence of a (season, team_name) row sets ap_top12_flag = 1.
  ap_rank_week6 is optional; include it for rank-ordering but it is not required.

How to fill in
--------------
  Add one row per team per season that was in the AP top 12 at Week 6.
  Use the exact team name from cbb.csv (normalization handles variants).
  Do NOT list teams that were outside the top 12.

Example
-------
  season,team_name,ap_rank_week6
  2016,Villanova,1
  2016,Kansas,2
  2016,Virginia,3
  ...
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from lib.data_merger import normalize_name

CBB_PATH       = PROJECT_ROOT / "data" / "raw" / "cbb.csv"
OUT_PATH       = PROJECT_ROOT / "data" / "raw" / "ap_week6.csv"
TARGET_SEASONS = [2013, 2014, 2015, 2016]

W   = 72
SEP = "=" * W
THN = "─" * W


def main() -> None:
    print(SEP)
    print(f"{'BUILD AP WEEK 6 TEMPLATE':^{W}}")
    print(SEP)

    # ── Build reference list of all tournament teams (for lookup convenience) ──
    if CBB_PATH.exists():
        cbb = pd.read_csv(CBB_PATH)
        cbb = cbb[cbb["SEED"].notna() & cbb["YEAR"].isin(TARGET_SEASONS)].copy()
        cbb["SEED"] = cbb["SEED"].astype(int)
        cbb = cbb.rename(columns={"YEAR": "season", "TEAM": "team_name_raw", "SEED": "seed"})
        print(f"\n  Reference: {len(cbb)} tournament teams across {TARGET_SEASONS[0]}–{TARGET_SEASONS[-1]}")
        print(f"\n  Top seeds by season (likely AP top-12 candidates):")
        print(f"  {'─'*60}")
        for season in TARGET_SEASONS:
            top = (cbb[cbb["season"] == season]
                   .sort_values("seed")
                   .drop_duplicates("team_name_raw")
                   .head(5)["team_name_raw"]
                   .tolist())
            print(f"  {season}  seeds 1–2: {', '.join(top[:5])}")
    else:
        print(f"\n  (cbb.csv not found — writing empty template)")

    # ── Write headers-only template ───────────────────────────────────────────
    template = pd.DataFrame(columns=["season", "team_name", "ap_rank_week6"])
    template.to_csv(OUT_PATH, index=False)

    print(f"\n{SEP}")
    print(f"  Saved → {OUT_PATH}  (0 data rows — ready to fill in)")
    print(f"\n  Add one row per AP top-12 team per season.")
    print(f"  ap_top12_flag = 1 for every team listed in this file.")
    print(f"  ap_rank_week6 is optional — include it for ordering, not required.")
    print(SEP)


if __name__ == "__main__":
    main()
