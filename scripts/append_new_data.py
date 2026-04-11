"""
append_new_data.py
Appends a newer season range (e.g. 2016-2025) to the two master CSVs that
feed the rest of the pipeline:

    data/raw/TourneyCompactResults.csv
    data/raw/TourneySeeds.csv

────────────────────────────────────────────────────────────────────────────
SOURCE FILES REQUIRED
────────────────────────────────────────────────────────────────────────────
Place these files in  data/march-madness-data/  before running:

  results.csv — one row per game, columns:
    year        int    e.g. 2017
    round       int    1=R64, 2=R32, 3=Sweet16, 4=Elite8, 5=FF, 6=Champ
    team        int    winning team ID  (must match IDs in TourneySeeds)
    team_score  int    winning team score
    opponent    int    losing team ID
    opp_score   int    losing team score

  seeds.csv — one row per team per season, columns:
    year        int    e.g. 2017
    team        int    team ID
    seed        str/int  "W01" or 1  (both formats accepted)

────────────────────────────────────────────────────────────────────────────
NOTES
────────────────────────────────────────────────────────────────────────────
* Deduplication: rows are keyed on (Season, Wteam, Lteam).  Re-running this
  script is safe — it will not create duplicate rows.
* Overlap: the existing master data ends at 2016.  If results.csv also
  contains 2016 rows they will be deduplicated automatically.
* After running this script, run  python scripts/run_pipeline.py  to
  regenerate cleaned_games.csv and seed_probabilities.json.
"""
import re
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
NEW_DATA_DIR = PROJECT_ROOT / "data" / "march-madness-data"
NEW_RESULTS  = NEW_DATA_DIR / "results.csv"
NEW_SEEDS    = NEW_DATA_DIR / "seeds.csv"

COMPACT_PATH = PROJECT_ROOT / "data" / "raw" / "TourneyCompactResults.csv"
SEEDS_PATH   = PROJECT_ROOT / "data" / "raw" / "TourneySeeds.csv"

# Canonical Daynum value for each round (used by load_data.py to map rounds)
ROUND_TO_DAYNUM: dict[int, int] = {
    0: 134,   # Play-In
    1: 136,   # Round of 64
    2: 138,   # Round of 32
    3: 143,   # Sweet 16
    4: 145,   # Elite 8
    5: 152,   # Final Four
    6: 154,   # Championship
}


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _check_file(path: Path) -> None:
    if not path.exists():
        sys.exit(
            f"ERROR: required source file not found:\n  {path}\n"
            f"See the docstring at the top of this script for the expected "
            f"column layout."
        )


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: {label} is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def _parse_seed(val) -> int | None:
    """Accept both raw strings ('W01', 'Z16a') and plain integers (1, 12)."""
    if pd.isna(val):
        return None
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else None


# ────────────────────────────────────────────────────────────────────────────
# Build new compact rows
# ────────────────────────────────────────────────────────────────────────────

def build_compact(results_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert new results into TourneyCompactResults schema.

    Input columns required: year, round, team, team_score, opponent, opp_score
    Output columns:  Season, Daynum, Wteam, Wscore, Lteam, Lscore, Wloc, Numot
    """
    _require_columns(results_raw,
                     ["year", "round", "team", "team_score", "opponent", "opp_score"],
                     "results.csv")

    # Validate 'round' values
    bad_rounds = set(results_raw["round"].unique()) - set(ROUND_TO_DAYNUM)
    if bad_rounds:
        sys.exit(
            f"ERROR: results.csv contains unknown round values: {sorted(bad_rounds)}\n"
            f"Expected: {sorted(ROUND_TO_DAYNUM.keys())}  "
            f"(0=Play-In, 1=R64, 2=R32, 3=S16, 4=E8, 5=FF, 6=Champ)"
        )

    df = results_raw.copy()

    # Determine winner/loser per row
    # (team/team_score are not guaranteed to be the winner in the source)
    df["Wteam"]  = df.apply(
        lambda r: r["team"] if r["team_score"] >= r["opp_score"] else r["opponent"],
        axis=1,
    ).astype(int)
    df["Lteam"]  = df.apply(
        lambda r: r["opponent"] if r["team_score"] >= r["opp_score"] else r["team"],
        axis=1,
    ).astype(int)
    df["Wscore"] = df[["team_score", "opp_score"]].max(axis=1).astype(int)
    df["Lscore"] = df[["team_score", "opp_score"]].min(axis=1).astype(int)

    df["Season"] = df["year"].astype(int)
    df["Daynum"] = df["round"].map(ROUND_TO_DAYNUM).astype(int)
    df["Wloc"]   = "N"     # location unknown for new data; neutral assumed
    df["Numot"]  = 0       # overtime unknown for new data

    return df[["Season", "Daynum", "Wteam", "Wscore", "Lteam", "Lscore",
               "Wloc", "Numot"]]


# ────────────────────────────────────────────────────────────────────────────
# Build new seeds rows
# ────────────────────────────────────────────────────────────────────────────

def build_seeds(seeds_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert new seeds into TourneySeeds schema.

    Input columns required: year, team, seed
    Output columns: Season, Seed, Team
    """
    _require_columns(seeds_raw, ["year", "team", "seed"], "seeds.csv")

    df = seeds_raw.copy()
    df["Season"] = df["year"].astype(int)
    df["Team"]   = df["team"].astype(int)

    # Normalise seed to int (accepts "W01", "Z16a", or plain 12)
    df["Seed"] = df["seed"].apply(_parse_seed)
    nulls = df["Seed"].isna().sum()
    if nulls:
        print(f"  Warning: {nulls} seed rows could not be parsed and will be dropped.")
    df = df.dropna(subset=["Seed"])
    df["Seed"] = df["Seed"].astype(int)

    return df[["Season", "Seed", "Team"]]


# ────────────────────────────────────────────────────────────────────────────
# Append with deduplication
# ────────────────────────────────────────────────────────────────────────────

def append_deduplicated(existing_path: Path,
                        new_rows: pd.DataFrame,
                        dedup_keys: list[str],
                        label: str) -> pd.DataFrame:
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        before   = len(existing)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=dedup_keys, keep="first")
        added    = len(combined) - before
        print(f"  {label}: {before:,} existing + {len(new_rows):,} new "
              f"→ {added:,} non-duplicate rows added → {len(combined):,} total")
    else:
        combined = new_rows
        print(f"  {label}: no existing file — writing {len(combined):,} rows")
    return combined


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _check_file(NEW_RESULTS)
    _check_file(NEW_SEEDS)

    print(f"Reading new source data from {NEW_DATA_DIR}")
    results_raw = pd.read_csv(NEW_RESULTS)
    seeds_raw   = pd.read_csv(NEW_SEEDS)
    print(f"  results.csv : {len(results_raw):,} rows, "
          f"seasons {results_raw['year'].min()}–{results_raw['year'].max()}")
    print(f"  seeds.csv   : {len(seeds_raw):,} rows")

    print("\nTransforming to master schema...")
    new_compact = build_compact(results_raw)
    new_seeds   = build_seeds(seeds_raw)

    print("\nAppending to master CSVs (deduplicating on Season+Wteam+Lteam / Season+Team)...")
    master_compact = append_deduplicated(
        COMPACT_PATH, new_compact,
        dedup_keys=["Season", "Wteam", "Lteam"],
        label="TourneyCompactResults",
    )
    master_seeds = append_deduplicated(
        SEEDS_PATH, new_seeds,
        dedup_keys=["Season", "Team"],
        label="TourneySeeds",
    )

    # Sort before saving
    master_compact = master_compact.sort_values(["Season", "Daynum"]).reset_index(drop=True)
    master_seeds   = master_seeds.sort_values(["Season", "Team"]).reset_index(drop=True)

    master_compact.to_csv(COMPACT_PATH, index=False)
    master_seeds.to_csv(SEEDS_PATH, index=False)

    print(f"\nSaved → {COMPACT_PATH}")
    print(f"Saved → {SEEDS_PATH}")
    print("\nNext step: python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
