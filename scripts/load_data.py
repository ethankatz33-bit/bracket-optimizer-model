"""
load_data.py
Ingests the Kaggle NCAA dataset from two source files:

  data/raw/TourneyCompactResults.csv  — one row per game
  data/raw/TourneySeeds.csv           — seed string per team per season

Steps:
  1. Parse numeric seed from Seed strings ("W01"→1, "X12"→12, "Z16a"→16).
  2. Merge winning-team seed and losing-team seed into the results table.
  3. Map Daynum to a round number (1=R64 … 6=Championship).
  4. Rename Season→year and write data/raw/ncaa_tournament_games.csv.
"""
import re
import pandas as pd
from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
RESULTS_FILE  = RAW_DIR / "TourneyCompactResults.csv"
SEEDS_FILE    = RAW_DIR / "TourneySeeds.csv"
OUTPUT_FILE   = RAW_DIR / "ncaa_tournament_games.csv"

# Kaggle Daynum values → round number used throughout the rest of the pipeline
# 134/135 = First-Four play-in games (filtered out later in clean_data.py)
DAYNUM_TO_ROUND = {
    134: 0, 135: 0,   # Play-In (First Four)
    136: 1, 137: 1,   # Round of 64
    138: 2, 139: 2,   # Round of 32
    143: 3, 144: 3,   # Sweet 16
    145: 4, 146: 4,   # Elite 8
    152: 5,            # Final Four
    154: 6,            # Championship
}

ROUND_NAMES = {
    0: "Play-In",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}


def parse_seed(raw: str) -> int | None:
    """
    Extract the numeric seed from a Kaggle seed string.
    Examples: "W01" → 1,  "X12" → 12,  "Z16a" → 16,  "Y11b" → 11
    """
    if pd.isna(raw):
        return None
    m = re.search(r"(\d+)", str(raw))
    return int(m.group(1)) if m else None


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Load source files                                                 #
    # ------------------------------------------------------------------ #
    for path in (RESULTS_FILE, SEEDS_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Place TourneyCompactResults.csv and TourneySeeds.csv "
                f"in {RAW_DIR}"
            )

    results = pd.read_csv(RESULTS_FILE)
    seeds   = pd.read_csv(SEEDS_FILE)
    print(f"Loaded results : {len(results):,} rows  "
          f"(seasons {results['Season'].min()}–{results['Season'].max()})")
    print(f"Loaded seeds   : {len(seeds):,} rows  "
          f"(seasons {seeds['Season'].min()}–{seeds['Season'].max()})")

    # ------------------------------------------------------------------ #
    # 2. Parse numeric seed from seed strings                              #
    # ------------------------------------------------------------------ #
    seeds["seed_num"] = seeds["Seed"].apply(parse_seed)
    unparsed = seeds["seed_num"].isna().sum()
    if unparsed:
        print(f"  Warning: {unparsed} seed strings could not be parsed — will drop.")
    seeds = seeds.dropna(subset=["seed_num"])
    seeds["seed_num"] = seeds["seed_num"].astype(int)

    # ------------------------------------------------------------------ #
    # 3. Merge winning-team seed                                           #
    # ------------------------------------------------------------------ #
    w_seeds = seeds[["Season", "Team", "seed_num"]].rename(
        columns={"Team": "Wteam", "seed_num": "winning_seed"}
    )
    df = results.merge(w_seeds, on=["Season", "Wteam"], how="left")

    # 4. Merge losing-team seed                                            #
    l_seeds = seeds[["Season", "Team", "seed_num"]].rename(
        columns={"Team": "Lteam", "seed_num": "losing_seed"}
    )
    df = df.merge(l_seeds, on=["Season", "Lteam"], how="left")

    missing_seeds = df[["winning_seed", "losing_seed"]].isna().any(axis=1).sum()
    if missing_seeds:
        print(f"  Warning: {missing_seeds} games have no seed data — will be "
              f"dropped during cleaning.")

    # ------------------------------------------------------------------ #
    # 5. Map Daynum → round number and round name                          #
    # ------------------------------------------------------------------ #
    unknown_days = set(df["Daynum"].unique()) - set(DAYNUM_TO_ROUND)
    if unknown_days:
        print(f"  Warning: unmapped Daynum values: {sorted(unknown_days)} — "
              f"those rows will have round=NaN and be dropped in clean_data.py.")

    df["round"]      = df["Daynum"].map(DAYNUM_TO_ROUND)
    df["round_name"] = df["round"].map(ROUND_NAMES)

    # ------------------------------------------------------------------ #
    # 6. Rename / select columns and save                                  #
    # ------------------------------------------------------------------ #
    df = df.rename(columns={"Season": "year", "Wteam": "winning_team",
                             "Lteam": "losing_team"})

    keep = ["year", "round", "round_name",
            "winning_team", "losing_team",
            "winning_seed", "losing_seed"]
    df = df[keep].sort_values(["year", "round"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved → {OUTPUT_FILE}  ({len(df):,} rows)")
    print(f"Columns : {list(df.columns)}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
