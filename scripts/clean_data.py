"""
clean_data.py
Cleans the merged tournament data produced by load_data.py:

  - Filters to seasons >= 1985
  - Drops play-in games (round 0 / First Four) — both teams are the same
    seed (e.g. two 16-seeds), which would distort matchup statistics
  - Drops rows with missing seeds or rounds
  - Casts seeds to int (load_data.py already extracted numerics, but this
    guards against any residual floats from the merge)
  - Adds matchup column  →  "{lower_seed}_vs_{higher_seed}"  (e.g. "5_vs_12")
  - Adds upset column    →  1 if winning_seed > losing_seed, else 0

Saves to data/processed/cleaned_games.csv.
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE   = PROJECT_ROOT / "data" / "raw"       / "ncaa_tournament_games.csv"
OUTPUT_FILE  = PROJECT_ROOT / "data" / "processed" / "cleaned_games.csv"


def main() -> None:
    print(f"Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    # ------------------------------------------------------------------ #
    # 1. Filter to 1985-present                                            #
    # ------------------------------------------------------------------ #
    before = len(df)
    df = df[df["year"] >= 1985].copy()
    print(f"  Kept {len(df):,} rows (≥ 1985); dropped {before - len(df):,}")

    # ------------------------------------------------------------------ #
    # 2. Drop play-in games (round 0)                                      #
    # Play-in games pit two teams of the same seed against each other      #
    # (e.g. 16a vs 16b). They are excluded so seed-matchup stats reflect   #
    # only the main 64-team bracket.                                        #
    # ------------------------------------------------------------------ #
    before = len(df)
    df = df[df["round"] != 0]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} play-in games (round 0 / First Four)")

    # ------------------------------------------------------------------ #
    # 3. Drop rows with missing seeds or round                             #
    # ------------------------------------------------------------------ #
    before = len(df)
    df = df.dropna(subset=["winning_seed", "losing_seed", "round"])
    if before - len(df):
        print(f"  Dropped {before - len(df):,} rows with missing seed/round data")

    df["winning_seed"] = df["winning_seed"].astype(int)
    df["losing_seed"]  = df["losing_seed"].astype(int)
    df["round"]        = df["round"].astype(int)

    # ------------------------------------------------------------------ #
    # 4. Derived columns                                                   #
    # ------------------------------------------------------------------ #
    # matchup: lower seed number first — "5_vs_12" whether 5 won or 12 won
    df["matchup"] = df.apply(
        lambda r: (
            f"{min(r['winning_seed'], r['losing_seed'])}"
            "_vs_"
            f"{max(r['winning_seed'], r['losing_seed'])}"
        ),
        axis=1,
    )

    # upset: 1 when the numerically higher (worse) seed wins.
    # 8 vs 9 is treated as a pick'em — neither outcome is an upset.
    is_8_9 = (df["winning_seed"].isin([8, 9])) & (df["losing_seed"].isin([8, 9]))
    df["upset"] = ((df["winning_seed"] > df["losing_seed"]) & ~is_8_9).astype(int)

    # ------------------------------------------------------------------ #
    # 5. Select and order columns                                          #
    # ------------------------------------------------------------------ #
    base = ["year", "round", "round_name", "winning_seed", "losing_seed",
            "matchup", "upset"]
    optional = ["winning_team", "losing_team"]
    keep = base + [c for c in optional if c in df.columns]
    df = df[keep].sort_values(["year", "round"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 6. Save                                                              #
    # ------------------------------------------------------------------ #
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    n_years  = df["year"].nunique()
    n_upsets = df["upset"].sum()
    rounds   = df.groupby("round")["round_name"].first().to_dict()
    print(f"\nSaved → {OUTPUT_FILE}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Seasons : {df['year'].min()}–{df['year'].max()} ({n_years} years)")
    print(f"  Rounds  : {rounds}")
    print(f"  Upsets  : {n_upsets:,} ({n_upsets / len(df):.1%} of all games)")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
