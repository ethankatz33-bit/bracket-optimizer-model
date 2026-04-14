"""
compute_probabilities.py
Derives three probability tables from the cleaned tournament data:

  1. matchup_win_rates   — upset win-rate keyed as "higher_vs_lower"
                           (e.g. "12_vs_5": 0.35 means 12-seeds beat 5-seeds 35 %)
  2. advancement_rates   — per-seed fraction of R1 appearances that
                           reached each subsequent round
  3. avg_upsets_per_round — tournament-average upset count per round
"""
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE  = PROJECT_ROOT / "data" / "processed" / "cleaned_games.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "seed_probabilities.json"

# human-readable labels for round numbers in the output
ROUND_LABELS = {
    1: "round_64",
    2: "round_32",
    3: "sweet_16",
    4: "elite_8",
    5: "final_four",
    6: "championship",
}

# ── Recency weighting ─────────────────────────────────────────────────────────
# Mirrors the weights used in lib/backtest.py.  Newer seasons contribute more
# to upset rates and advancement rates so the model reflects modern trends.
def _season_weight(year: int) -> float:
    if year >= 2013:
        return 1.0
    if year >= 2005:
        return 0.7
    return 0.4


# ────────────────────────────────────────────────────────────────────────────
# 1. Matchup win rates
# ────────────────────────────────────────────────────────────────────────────
def compute_matchup_win_rates(df: pd.DataFrame) -> dict[str, float]:
    """
    Recency-weighted upset win-rate for every seed matchup.
    Key format: "{higher}_vs_{lower}".

    Each game is weighted by _season_weight(year) so recent seasons contribute
    more than older ones.  Requires a 'year' column in df.
    """
    w_col = df["year"].map(_season_weight) if "year" in df.columns else pd.Series(1.0, index=df.index)
    rates: dict[str, float] = {}

    for matchup, grp in df.groupby("matchup"):
        parts = matchup.split("_vs_")
        lower_seed  = int(parts[0])
        higher_seed = int(parts[1])

        if lower_seed == higher_seed:
            continue

        grp_w      = w_col.loc[grp.index]
        total_w    = float(grp_w.sum())
        upset_w    = float(grp_w[grp["winning_seed"] == higher_seed].sum())
        rates[f"{higher_seed}_vs_{lower_seed}"] = round(upset_w / total_w, 4) if total_w > 0 else 0.0

    rates = dict(
        sorted(rates.items(),
               key=lambda kv: (int(kv[0].split("_vs_")[0]),
                               int(kv[0].split("_vs_")[1])))
    )
    return rates


# ────────────────────────────────────────────────────────────────────────────
# 2. Advancement rates by seed
# ────────────────────────────────────────────────────────────────────────────
def compute_advancement_rates(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Recency-weighted per-seed advancement rates.

    For each seed (1-16), the fraction of weighted Round-of-64 appearances
    that advanced to each subsequent stage.  Recent seasons are up-weighted
    relative to older ones via _season_weight(year).

    round_64   = P(win the opening game)
    sweet_16   = P(reached Sweet 16)
    elite_8    = P(reached Elite 8)
    final_four = P(reached Final Four)
    champion   = P(won the title)
    """
    w_col = df["year"].map(_season_weight) if "year" in df.columns else pd.Series(1.0, index=df.index)
    r1    = df[df["round"] == 1]
    r1_w  = w_col.loc[r1.index]

    appearances_w: dict[int, float] = {}
    for seed in range(1, 17):
        mask = (r1["winning_seed"] == seed) | (r1["losing_seed"] == seed)
        appearances_w[seed] = float(r1_w[mask].sum())

    advancement: dict[str, dict[str, float]] = {}

    for seed in range(1, 17):
        n_w = appearances_w.get(seed, 0.0)
        if n_w == 0:
            continue

        r64_w = float(r1_w[r1["winning_seed"] == seed].sum())
        rates: dict[str, float] = {"round_64": round(r64_w / n_w, 4)}

        for round_num, label in [
            (2, "sweet_16"),
            (3, "elite_8"),
            (4, "final_four"),
        ]:
            rnd    = df[df["round"] == round_num]
            wins_w = float(w_col.loc[rnd.index][rnd["winning_seed"] == seed].sum())
            rates[label] = round(wins_w / n_w, 4)

        champ   = df[df["round"] == 6]
        champ_w = float(w_col.loc[champ.index][champ["winning_seed"] == seed].sum())
        rates["champion"] = round(champ_w / n_w, 4)

        advancement[str(seed)] = rates

    return advancement


# ────────────────────────────────────────────────────────────────────────────
# 3. Average upsets per round
# ────────────────────────────────────────────────────────────────────────────
def compute_avg_upsets_per_round(df: pd.DataFrame) -> dict[str, float]:
    n_years = df["year"].nunique()
    result: dict[str, float] = {}

    for round_num, label in ROUND_LABELS.items():
        rnd = df[df["round"] == round_num]
        avg = round(rnd["upset"].sum() / n_years, 2)
        result[label] = avg

    return result


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Loading cleaned data: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"  {len(df):,} games, {df['year'].nunique()} seasons")

    print("Computing matchup win rates...")
    matchup_win_rates = compute_matchup_win_rates(df)

    print("Computing advancement rates by seed...")
    advancement_rates = compute_advancement_rates(df)

    print("Computing average upsets per round...")
    avg_upsets = compute_avg_upsets_per_round(df)

    output = {
        "matchup_win_rates": matchup_win_rates,
        "advancement_rates": advancement_rates,
        "avg_upsets_per_round": avg_upsets,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {OUTPUT_FILE}")

    # ── Summary printout ─────────────────────────────────────────────────
    print("\n── First-round upset win rates ──")
    # Standard R64 matchups always sum to 17 (1+16, 2+15, … 8+9)
    r1_keys = [k for k in matchup_win_rates
               if int(k.split("_vs_")[0]) + int(k.split("_vs_")[1]) == 17]
    for k in sorted(r1_keys, key=lambda x: int(x.split("_vs_")[0])):
        print(f"  {k:>8s}: {matchup_win_rates[k]:.1%}")

    print("\n── Advancement rates — Seed 1 ──")
    for stage, rate in advancement_rates.get("1", {}).items():
        print(f"  {stage:<12s}: {rate:.1%}")

    print("\n── Average upsets per round ──")
    for rnd, avg in avg_upsets.items():
        print(f"  {rnd:<14s}: {avg:.1f}")


if __name__ == "__main__":
    main()
