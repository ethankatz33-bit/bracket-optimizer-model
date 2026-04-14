"""
lib/backtest.py
Historical backtest engine for the March Madness bracket generator.

For a target year and mode, this module:
  1. Builds a probability model from ALL seasons strictly before the target year
     (true pre-tournament information only).
  2. Loads the actual teams seeded in the target year's bracket, using
     play-in game results to resolve a/b seed slots.
  3. Generates a full bracket via the existing simulator with the
     pre-tournament model injected as overrides.
  4. Loads the actual tournament results for the target year.
  5. Scores each round (correct picks = predicted winner == actual winner).
  6. Produces four diagnostic assessments.
  7. Saves a JSON record to data/processed/backtests/{year}_{mode}_backtest.json.

Supported years: any year in the dataset that has at least MIN_PRIOR_SEASONS
seasons of prior data.  With the default dataset that is 1990–2016.

Public API
----------
  run_backtest(year, mode) → dict
"""

import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RAW_RESULTS  = PROJECT_ROOT / "data" / "raw" / "TourneyCompactResults.csv"
RAW_SEEDS    = PROJECT_ROOT / "data" / "raw" / "TourneySeeds.csv"
BACKTEST_DIR = PROJECT_ROOT / "data" / "processed" / "backtests"

# Minimum prior seasons needed for stable probability estimates.
MIN_PRIOR_SEASONS = 5

# Kaggle Daynum → round number (mirrors load_data.py)
DAYNUM_TO_ROUND = {
    134: 0, 135: 0,   # Play-In
    136: 1, 137: 1,   # Round of 64
    138: 2, 139: 2,   # Round of 32
    143: 3, 144: 3,   # Sweet 16
    145: 4, 146: 4,   # Elite 8
    152: 5,            # Final Four
    154: 6,            # Championship
}

ROUND_NAMES = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}

# Region letter (TourneySeeds.csv) → human-readable name used in bracket
REGION_LETTER_TO_NAME = {
    "W": "East",
    "X": "West",
    "Y": "South",
    "Z": "Midwest",
}

# Per-seed average ratings derived from MOCK_TEAMS (mean across 4 regions).
# Used to assign team ratings when actual pre-tournament ratings aren't available.
# All teams of the same seed receive the same base rating so the simulation is
# driven by seed quality and historical probabilities, not fabricated strength gaps.
SEED_BASE_RATING: dict[int, float] = {
    1:  93.6,   2:  86.6,   3:  81.9,   4:  76.8,
    5:  71.8,   6:  67.0,   7:  62.2,   8:  57.2,
    9:  52.6,   10: 48.9,   11: 44.8,   12: 40.9,
    13: 36.8,   14: 32.8,   15: 28.8,   16: 22.8,
}

# Maximum correct picks possible per round
MAX_PICKS_PER_ROUND = {
    "Round of 64":  32,
    "Round of 32":  16,
    "Sweet 16":      8,
    "Elite 8":       4,
    "Final Four":    2,
    "Championship":  1,
}


# ════════════════════════════════════════════════════════════════════════════
# Raw data loading
# ════════════════════════════════════════════════════════════════════════════

def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load TourneyCompactResults.csv and TourneySeeds.csv from disk."""
    for path in (RAW_RESULTS, RAW_SEEDS):
        if not path.exists():
            raise FileNotFoundError(f"Required raw file missing: {path}")
    return pd.read_csv(RAW_RESULTS), pd.read_csv(RAW_SEEDS)


def _parse_seed_num(raw) -> int | None:
    """Extract numeric seed from strings like 'W01', 'X12', 'Z16a', 'Y11b'."""
    m = re.search(r"(\d+)", str(raw))
    return int(m.group(1)) if m else None


# ════════════════════════════════════════════════════════════════════════════
# Pre-tournament model construction
# ════════════════════════════════════════════════════════════════════════════

# ── Recency weighting ─────────────────────────────────────────────────────────
# Newer seasons are weighted more heavily when computing upset rates and
# advancement rates so the model reflects modern basketball trends.
#
#   2015–2025  → 1.00  (modern era — full weight)
#   2005–2014  → 0.70  (mid-era)
#   1990–2004  → 0.40  (early era — structurally similar but noisier / older)
def _season_weight(year: int) -> float:
    if year >= 2015:
        return 1.0
    if year >= 2005:
        return 0.7
    return 0.4


def _build_historical_df(
    results: pd.DataFrame,
    seeds: pd.DataFrame,
    max_year: int,
) -> pd.DataFrame:
    """
    In-memory replication of the load_data + clean_data pipeline for all
    seasons in [1985, max_year).

    Produces a DataFrame in the same format as cleaned_games.csv plus a
    'weight' column for recency-weighted probability estimation:
      year, round, winning_seed, losing_seed, matchup, upset, weight
    """
    # Parse numeric seeds
    s = seeds.copy()
    s["seed_num"] = s["Seed"].apply(_parse_seed_num)
    s = s.dropna(subset=["seed_num"])
    s["seed_num"] = s["seed_num"].astype(int)

    # Merge winning- and losing-team seeds onto results
    w_map = s[["Season", "Team", "seed_num"]].rename(
        columns={"Team": "Wteam", "seed_num": "winning_seed"})
    l_map = s[["Season", "Team", "seed_num"]].rename(
        columns={"Team": "Lteam", "seed_num": "losing_seed"})

    df = results.merge(w_map, on=["Season", "Wteam"], how="left")
    df = df.merge(l_map, on=["Season", "Lteam"], how="left")

    df["round"] = df["Daynum"].map(DAYNUM_TO_ROUND)
    df = df.rename(columns={"Season": "year"})

    # Apply the same filters as clean_data.py
    df = df[(df["year"] >= 1985) & (df["year"] < max_year)]
    df = df[df["round"] != 0]   # exclude play-in games
    df = df.dropna(subset=["winning_seed", "losing_seed", "round"])
    df["winning_seed"] = df["winning_seed"].astype(int)
    df["losing_seed"]  = df["losing_seed"].astype(int)
    df["round"]        = df["round"].astype(int)

    # Derived columns
    df["matchup"] = df.apply(
        lambda r: (
            f"{min(r['winning_seed'], r['losing_seed'])}"
            "_vs_"
            f"{max(r['winning_seed'], r['losing_seed'])}"
        ),
        axis=1,
    )
    is_8_9 = df["winning_seed"].isin([8, 9]) & df["losing_seed"].isin([8, 9])
    df["upset"]  = ((df["winning_seed"] > df["losing_seed"]) & ~is_8_9).astype(int)
    df["weight"] = df["year"].map(_season_weight)

    return df.sort_values(["year", "round"]).reset_index(drop=True)


def _compute_matchup_win_rates(df: pd.DataFrame) -> dict[str, float]:
    """
    Recency-weighted upset win-rate per seed matchup, keyed as "{higher}_vs_{lower}".

    Each game is weighted by _season_weight(year): recent seasons contribute more
    to the upset rates than older seasons, reflecting modern basketball trends.
    """
    w_col = df["weight"] if "weight" in df.columns else pd.Series(1.0, index=df.index)
    rates: dict[str, float] = {}
    for matchup, grp in df.groupby("matchup"):
        parts       = matchup.split("_vs_")
        lower_seed  = int(parts[0])
        higher_seed = int(parts[1])
        if lower_seed == higher_seed:
            continue
        grp_w       = w_col.loc[grp.index]
        total_w     = float(grp_w.sum())
        upset_w     = float(grp_w[grp["winning_seed"] == higher_seed].sum())
        rates[f"{higher_seed}_vs_{lower_seed}"] = round(upset_w / total_w, 4) if total_w > 0 else 0.0
    return dict(
        sorted(rates.items(),
               key=lambda kv: (int(kv[0].split("_vs_")[0]),
                               int(kv[0].split("_vs_")[1])))
    )


def _compute_advancement_rates(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Recency-weighted per-seed fraction of R64 appearances reaching each round.

    Denominators and numerators are both weighted sums so the rates naturally
    reflect how seed tiers have performed in recent years vs. the full history.
    """
    w_col = df["weight"] if "weight" in df.columns else pd.Series(1.0, index=df.index)
    r1    = df[df["round"] == 1]
    r1_w  = w_col.loc[r1.index]

    appearances_w = {
        seed: float(
            r1_w[(r1["winning_seed"] == seed) | (r1["losing_seed"] == seed)].sum()
        )
        for seed in range(1, 17)
    }

    advancement: dict[str, dict[str, float]] = {}
    for seed in range(1, 17):
        n_w = appearances_w.get(seed, 0.0)
        if n_w == 0:
            continue
        r64_w  = float(r1_w[r1["winning_seed"] == seed].sum())
        rates: dict[str, float] = {"round_64": round(r64_w / n_w, 4)}
        for round_num, label in [(2, "sweet_16"), (3, "elite_8"), (4, "final_four")]:
            rnd   = df[df["round"] == round_num]
            wins_w = float(w_col.loc[rnd.index][rnd["winning_seed"] == seed].sum())
            rates[label] = round(wins_w / n_w, 4)
        champ   = df[df["round"] == 6]
        champ_w = float(w_col.loc[champ.index][champ["winning_seed"] == seed].sum())
        rates["champion"] = round(champ_w / n_w, 4)
        advancement[str(seed)] = rates
    return advancement


def _build_pretournament_model(
    target_year: int,
    results:     pd.DataFrame,
    seeds:       pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Build historical DataFrame and probability/structure dicts from all
    seasons strictly before target_year.

    Returns
    -------
    hist_df    : cleaned DataFrame of historical games
    probs      : dict with matchup_win_rates + advancement_rates
    structures : dict with conservative/balanced/upset_heavy bracket structures
    """
    # Lazy import to avoid circular dep; bracket_optimizer is a lib-level module
    from lib.bracket_optimizer import generate_bracket_structure_from_data

    hist_df = _build_historical_df(results, seeds, max_year=target_year)

    probs = {
        "matchup_win_rates": _compute_matchup_win_rates(hist_df),
        "advancement_rates": _compute_advancement_rates(hist_df),
    }
    structures = generate_bracket_structure_from_data(hist_df, probs["advancement_rates"])

    return hist_df, probs, structures


# ════════════════════════════════════════════════════════════════════════════
# Target-year team field construction
# ════════════════════════════════════════════════════════════════════════════

def _resolve_playin_winners(
    target_year: int,
    results:     pd.DataFrame,
    seeds:       pd.DataFrame,
) -> dict[tuple[str, int], int]:
    """
    For play-in (First Four) games in the target year, return a mapping:
      (region_letter, seed_num) → winning team_id

    This is used to fill play-in seed slots (11a/11b, 16a/16b) with the
    team that actually entered the main bracket.  Play-in results are
    publicly known before the main bracket tips off, so using them here
    preserves the "pre-tournament information only" guarantee.
    """
    year_seeds = seeds[seeds["Season"] == target_year].copy()
    year_seeds["seed_num"] = year_seeds["Seed"].apply(_parse_seed_num)
    year_seeds["region_letter"] = year_seeds["Seed"].str[0]
    year_seeds = year_seeds.dropna(subset=["seed_num"])
    year_seeds["seed_num"] = year_seeds["seed_num"].astype(int)

    playin_games = results[
        (results["Season"] == target_year) & results["Daynum"].isin([134, 135])
    ]

    winner_for: dict[tuple[str, int], int] = {}
    for _, game in playin_games.iterrows():
        w_row = year_seeds[year_seeds["Team"] == game["Wteam"]]
        if not w_row.empty:
            rl  = w_row.iloc[0]["region_letter"]
            sn  = int(w_row.iloc[0]["seed_num"])
            winner_for[(rl, sn)] = int(game["Wteam"])

    return winner_for


def _is_numeric_seed_format(year_seeds: pd.DataFrame) -> bool:
    """Return True when TourneySeeds uses plain integer seeds (2018+ format)."""
    if year_seeds.empty:
        return False
    sample = year_seeds["Seed"].iloc[0]
    return str(sample).isdigit()


def _build_year_teams(
    target_year:   int,
    results:       pd.DataFrame,
    seeds:         pd.DataFrame,
) -> dict[str, dict[int, dict]]:
    """
    Build the 64-team field for target_year in the same format as MOCK_TEAMS:
      { region_name: { seed_num: {"name": "T{id}", "rating": float} } }

    Handles two TourneySeeds.csv formats:
      • W/X/Y/Z prefix format (1985–2017): region is encoded in the seed string.
      • Plain-integer format (2018+): seed is a bare number; teams are
        distributed across the 4 regions in sorted team_id order.

    Play-in winners are resolved from TourneyCompactResults (Daynum 134/135)
    when the Daynum column is present.  If not, the first listed team is used.
    """
    year_seeds = seeds[seeds["Season"] == target_year].copy()
    year_seeds["seed_num"]      = year_seeds["Seed"].apply(_parse_seed_num)
    year_seeds["region_letter"] = year_seeds["Seed"].str[0]
    year_seeds["is_playin"]     = year_seeds["Seed"].str.match(r".*[ab]$", case=False)
    year_seeds = year_seeds.dropna(subset=["seed_num"])
    year_seeds["seed_num"] = year_seeds["seed_num"].astype(int)

    team_field: dict[str, dict[int, dict]] = {
        name: {} for name in REGION_LETTER_TO_NAME.values()
    }

    if _is_numeric_seed_format(year_seeds):
        # ── 2018+ plain-integer format ────────────────────────────────────
        # Resolve play-in winners from Daynum 134/135 games (if present).
        year_results = results[results["Season"] == target_year]
        playin_games = year_results[year_results["Daynum"].isin([134, 135])]
        playin_winner_ids: set[int] = set(playin_games["Wteam"].astype(int).tolist())
        playin_loser_ids:  set[int] = set(playin_games["Lteam"].astype(int).tolist())

        region_names = list(REGION_LETTER_TO_NAME.values())  # 4 regions

        for seed_num in range(1, 17):
            seed_teams = (
                year_seeds[year_seeds["seed_num"] == seed_num]
                .sort_values("Team")
                .reset_index(drop=True)
            )

            # Exclude known play-in losers; prefer known play-in winners.
            if not playin_loser_ids:
                # No play-in data — use all listed teams
                bracket_teams = seed_teams
            else:
                non_losers  = seed_teams[~seed_teams["Team"].isin(playin_loser_ids)]
                bracket_teams = non_losers if not non_losers.empty else seed_teams

            bracket_ids = bracket_teams["Team"].astype(int).tolist()

            for i, region_name in enumerate(region_names):
                if i < len(bracket_ids):
                    team_id = bracket_ids[i]
                else:
                    team_id = seed_num * 9000  # placeholder for missing slot
                team_field[region_name][seed_num] = {
                    "name":   f"T{team_id}",
                    "rating": SEED_BASE_RATING.get(seed_num, 20.0),
                }
    else:
        # ── Original W/X/Y/Z prefix format (1985–2017) ───────────────────
        playin_winners = _resolve_playin_winners(target_year, results, seeds)

        for region_letter, region_name in REGION_LETTER_TO_NAME.items():
            region_df = year_seeds[year_seeds["region_letter"] == region_letter]

            for seed_num in range(1, 17):
                slot_teams = region_df[region_df["seed_num"] == seed_num]

                if slot_teams.empty:
                    team_id = seed_num * 9000
                elif len(slot_teams) == 1 and not slot_teams.iloc[0]["is_playin"]:
                    team_id = int(slot_teams.iloc[0]["Team"])
                else:
                    team_id = playin_winners.get(
                        (region_letter, seed_num),
                        int(slot_teams.iloc[0]["Team"]),
                    )

                team_field[region_name][seed_num] = {
                    "name":   f"T{team_id}",
                    "rating": SEED_BASE_RATING.get(seed_num, 20.0),
                }

    return team_field


# ════════════════════════════════════════════════════════════════════════════
# Actual result loading
# ════════════════════════════════════════════════════════════════════════════

def _load_actual_results(
    target_year: int,
    results:     pd.DataFrame,
    seeds:       pd.DataFrame,
) -> dict[str, list[dict]]:
    """
    Load actual game outcomes for target_year.

    Returns
    -------
    {
      round_name: [
        { "name": "T{id}", "team_id": int, "seed": int, "region": str },
        ...  ← one entry per game winner in that round
      ]
    }
    """
    # Build team_id → (seed_num, region_name) lookup for this year
    year_seeds = seeds[seeds["Season"] == target_year].copy()
    year_seeds["seed_num"]      = year_seeds["Seed"].apply(_parse_seed_num)
    year_seeds["region_letter"] = year_seeds["Seed"].str[0]
    year_seeds = year_seeds.dropna(subset=["seed_num"])
    year_seeds["seed_num"] = year_seeds["seed_num"].astype(int)

    team_info: dict[int, dict] = {}
    for _, row in year_seeds.iterrows():
        tid = int(row["Team"])
        if tid not in team_info:   # keep first (lowest seed for play-in teams)
            team_info[tid] = {
                "seed":   int(row["seed_num"]),
                "region": REGION_LETTER_TO_NAME.get(row["region_letter"], row["region_letter"]),
            }

    year_results = results[results["Season"] == target_year].copy()
    year_results["round"] = year_results["Daynum"].map(DAYNUM_TO_ROUND)
    year_results = year_results[year_results["round"] >= 1]   # exclude play-in

    actual: dict[str, list[dict]] = {}
    for round_num, round_name in ROUND_NAMES.items():
        round_games = year_results[year_results["round"] == round_num]
        winners = []
        for _, game in round_games.iterrows():
            tid    = int(game["Wteam"])
            info   = team_info.get(tid, {"seed": None, "region": "?"})
            winners.append({
                "name":    f"T{tid}",
                "team_id": tid,
                "seed":    info["seed"],
                "region":  info["region"],
            })
        actual[round_name] = winners

    return actual


# ════════════════════════════════════════════════════════════════════════════
# Scoring
# ════════════════════════════════════════════════════════════════════════════

def _score_bracket(
    predicted:      dict,
    actual_results: dict[str, list[dict]],
) -> dict:
    """
    Score the predicted bracket round by round.

    A pick is correct when the team predicted to win a game actually did win
    that game, identified by team name ("T{id}").

    Returns
    -------
    {
      "total_correct":  int,
      "total_possible": int,
      "by_round": {
        round_name: { "correct": int, "possible": int }
      }
    }
    """
    BRACKET_KEY = {
        "Round of 64":  "round_of_64",
        "Round of 32":  "round_of_32",
        "Sweet 16":     "sweet_16",
        "Elite 8":      "elite_8",
        "Final Four":   "final_four",
        "Championship": "championship",
    }

    by_round: dict[str, dict] = {}
    total_correct  = 0
    total_possible = 0

    for round_name, bracket_key in BRACKET_KEY.items():
        games = predicted.get(bracket_key, [])
        if isinstance(games, dict):
            games = [games]

        pred_winners = {g["winner"]["name"] for g in games}
        actual_names = {t["name"] for t in actual_results.get(round_name, [])}

        correct  = len(pred_winners & actual_names)
        possible = MAX_PICKS_PER_ROUND[round_name]

        by_round[round_name] = {"correct": correct, "possible": possible}
        total_correct  += correct
        total_possible += possible

    return {
        "total_correct":  total_correct,
        "total_possible": total_possible,
        "by_round":       by_round,
    }


# ════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ════════════════════════════════════════════════════════════════════════════

def _build_diagnostics(
    predicted:      dict,
    actual_results: dict[str, list[dict]],
    scoring:        dict,
    mode:           str,
    year:           int,
) -> dict:
    """
    Four short assessments comparing the predicted bracket's character
    against actual tournament outcomes.
    """
    # ── 1. Chalkiness ─────────────────────────────────────────────────────
    pred_ff_seeds = [g["winner"]["seed"] for g in predicted["elite_8"]]
    low_seeds_in_ff = sum(1 for s in pred_ff_seeds if s <= 2)

    if low_seeds_in_ff >= 4:
        chalk = (
            f"Very chalky: all {low_seeds_in_ff}/4 predicted Final Four teams "
            f"were 1- or 2-seeds. The bracket leaned heavily on chalk."
        )
    elif low_seeds_in_ff == 3:
        chalk = (
            f"Slightly chalky: {low_seeds_in_ff}/4 predicted Final Four teams "
            f"were 1- or 2-seeds. Modestly conservative."
        )
    elif low_seeds_in_ff == 2:
        chalk = (
            f"Balanced: {low_seeds_in_ff}/4 predicted Final Four teams were "
            f"1- or 2-seeds, matching the historical average well."
        )
    else:
        chalk = (
            f"Bold: only {low_seeds_in_ff}/4 predicted Final Four teams were "
            f"1- or 2-seeds. Lower seeds dominated the predicted Final Four."
        )

    # ── 2. Upset assessment ───────────────────────────────────────────────
    all_games = (
        predicted["round_of_64"] + predicted["round_of_32"] +
        predicted["sweet_16"]    + predicted["elite_8"]     +
        predicted["final_four"]  + [predicted["championship"]]
    )
    total_upsets = sum(1 for g in all_games if g["is_upset"])

    # Historical (1985–2016): mean ~15.5 upsets/tournament, range 9–21, std ~3.2.
    # Under-pick threshold: <11 (more than 1.5 std below mean).
    # Over-pick threshold:  >19 (more than 1 std above mean).
    if total_upsets < 11:
        upset_msg = (
            f"Under-picked upsets: only {total_upsets} upsets predicted. "
            f"Tournaments historically average ~15–16 upsets (range 9–21). "
            f"Note: fewer upset picks tends to improve bracket score because "
            f"correctly predicting the specific upset is harder than picking the favorite."
        )
    elif total_upsets > 19:
        upset_msg = (
            f"Over-picked upsets: {total_upsets} upsets predicted. "
            f"Tournaments historically average ~15–16 upsets (range 9–21). "
            f"Excess upset picks increase variance and lower expected score."
        )
    else:
        upset_msg = (
            f"Reasonable upset volume: {total_upsets} upsets predicted "
            f"(historical mean ~15.5, range 9–21). "
            f"Conservative brackets (~11–12) tend to outscore balanced ones "
            f"because wrong upset picks are doubly costly in bracket scoring."
        )

    # ── 3. Sweet 16 double-digit seed review ──────────────────────────────
    s16_teams  = [g["winner"] for g in predicted["round_of_32"]]
    dd_in_pred = [t for t in s16_teams if t["seed"] and t["seed"] >= 10]
    actual_s16_names = {t["name"] for t in actual_results.get("Sweet 16", [])}
    s16_notes  = predicted.get("s16_constraint_notes", [])

    if dd_in_pred:
        dd_team    = dd_in_pred[0]
        did_advance = dd_team["name"] in actual_s16_names
        forced_tag  = " (forced by structure constraint)" if s16_notes else ""
        outcome     = (
            f"correctly advanced to the actual Sweet 16 in {year}"
            if did_advance
            else f"did NOT reach the actual Sweet 16 in {year}"
        )
        s16_review = (
            f"Predicted DD seed: {dd_team['name']} (seed {dd_team['seed']})"
            f"{forced_tag}. This team {outcome}."
        )
    else:
        s16_review = (
            "No double-digit seed was predicted to reach the Sweet 16 in this bracket."
        )

    # ── 4. Champion review ────────────────────────────────────────────────
    pred_champ        = predicted["champion"]
    actual_champ_names = {t["name"] for t in actual_results.get("Championship", [])}
    champ_correct      = pred_champ["name"] in actual_champ_names

    seed = pred_champ["seed"]
    if champ_correct:
        champ_review = (
            f"Champion correctly predicted: {pred_champ['name']} (seed {seed}). "
            f"The model's champion profile logic identified the right winner."
        )
    else:
        actual_str = (
            ", ".join(sorted(actual_champ_names)) if actual_champ_names else "unknown"
        )
        if seed == 1:
            profile_note = (
                "Picking a 1-seed is historically sound (~15% of titles). "
                "The model made the high-probability choice."
            )
        elif seed <= 3:
            profile_note = (
                f"A seed-{seed} champion is uncommon but within the historical range."
            )
        else:
            profile_note = (
                f"A seed-{seed} champion is a rare historical outcome."
            )
        champ_review = (
            f"Champion incorrect: predicted {pred_champ['name']} (seed {seed}), "
            f"actual champion was {actual_str}. {profile_note}"
        )

    return {
        "chalkiness_assessment":        chalk,
        "upset_assessment":             upset_msg,
        "sweet_16_double_digit_review": s16_review,
        "champion_review":              champ_review,
    }


# ════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════

def run_backtest(year: int, mode: str = "balanced") -> dict:
    """
    Run a full historical backtest for the given year and bracket mode.

    Parameters
    ----------
    year : int
        Target tournament year.  Must be present in the raw dataset and have
        at least MIN_PRIOR_SEASONS years of prior data.
    mode : "conservative" | "balanced" | "upset_heavy"

    Returns
    -------
    A dict matching the JSON output schema, plus two private keys used by
    the CLI printer:
      "_predicted_full"  — the complete bracket dict from simulate_bracket()
      "_actual_results"  — the actual-results dict keyed by round name
      "_hist_range"      — (first_year, last_year) of training window
      "_output_file"     — path where JSON was saved
    """
    from lib.team_selector import simulate_bracket   # local import to avoid circular

    results, seeds = _load_raw()

    # ── Validate year ─────────────────────────────────────────────────────
    available = sorted(results["Season"].unique())
    if year not in available:
        raise ValueError(
            f"Year {year} not found in the raw dataset "
            f"(available: {available[0]}–{available[-1]}). "
            f"Run scripts/append_new_data.py to extend coverage."
        )

    prior_years = [y for y in available if y < year]
    if len(prior_years) < MIN_PRIOR_SEASONS:
        raise ValueError(
            f"Only {len(prior_years)} season(s) of prior data before {year}. "
            f"Need at least {MIN_PRIOR_SEASONS} for reliable probability estimates."
        )

    # ── Step 1: pre-tournament probability model ──────────────────────────
    hist_df, probs, structures = _build_pretournament_model(year, results, seeds)
    n_hist = hist_df["year"].nunique()
    hist_range = (int(hist_df["year"].min()), int(hist_df["year"].max()))

    # ── Step 2: actual team field for target year ─────────────────────────
    teams = _build_year_teams(year, results, seeds)

    # ── Step 3: generate bracket ──────────────────────────────────────────
    predicted = simulate_bracket(
        mode,
        season=year,
        _teams_override=teams,
        _probs_override=probs,
        _structure_override=structures,
    )

    # ── Step 4: actual results ────────────────────────────────────────────
    actual_results = _load_actual_results(year, results, seeds)

    # ── Step 5: score ─────────────────────────────────────────────────────
    scoring = _score_bracket(predicted, actual_results)

    # ── Step 6: diagnostics ───────────────────────────────────────────────
    diagnostics = _build_diagnostics(predicted, actual_results, scoring, mode, year)

    # ── Step 7: assemble output record ───────────────────────────────────
    def _team_summary(t: dict) -> dict:
        d = {"name": t["name"], "seed": t["seed"]}
        if "profile_score" in t:
            d["profile_score"] = t["profile_score"]
        return d

    def _winner_list(game_list) -> list[dict]:
        if isinstance(game_list, dict):
            game_list = [game_list]
        return [_team_summary(g["winner"]) for g in game_list]

    # "Final Four teams" = the 4 teams that WON the Elite 8 and entered the FF.
    # actual_results["Elite 8"] = winners of E8 games = FF entrants (4 teams).
    # actual_results["Final Four"] = winners of FF semis = championship teams (2 teams).
    pred_ff   = _winner_list(predicted["elite_8"])
    actual_ff = [
        {"name": t["name"], "seed": t["seed"]}
        for t in actual_results.get("Elite 8", [])
    ]
    pred_champ = _team_summary(predicted["champion"])
    actual_champ_list = actual_results.get("Championship", [])
    actual_champ = (
        {"name": actual_champ_list[0]["name"], "seed": actual_champ_list[0]["seed"]}
        if actual_champ_list else {"name": "unknown", "seed": None}
    )

    output = {
        "year":                   year,
        "mode":                   mode,
        "pretournament_seasons":  n_hist,
        "pretournament_range":    f"{hist_range[0]}–{hist_range[1]}",
        "predicted_final_four":   pred_ff,
        "actual_final_four":      actual_ff,
        "predicted_champion":     pred_champ,
        "actual_champion":        actual_champ,
        "total_correct":          scoring["total_correct"],
        "total_possible":         scoring["total_possible"],
        "by_round": {
            r: v["correct"] for r, v in scoring["by_round"].items()
        },
        "by_round_detail":        scoring["by_round"],
        "diagnostics":            diagnostics,
        "ratings_diagnostics":    predicted.get("ratings_diagnostics", {}),
        "champion_value_score":   predicted.get("champion_value_score"),
        "champion_win_prob":      predicted.get("champion_win_prob"),
        "champion_public_pct":    predicted.get("champion_public_pct"),
    }

    # ── Step 8: save JSON ─────────────────────────────────────────────────
    def _strip_score(obj):
        if isinstance(obj, dict):
            return {k: _strip_score(v) for k, v in obj.items() if k != "score"}
        if isinstance(obj, list):
            return [_strip_score(i) for i in obj]
        return obj

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    out_file = BACKTEST_DIR / f"{year}_{mode}_backtest.json"
    with open(out_file, "w") as f:
        json.dump(_strip_score(output), f, indent=2)

    # Attach private keys for CLI printer (not written to JSON)
    output["_predicted_full"] = predicted
    output["_actual_results"] = actual_results
    output["_hist_range"]     = hist_range
    output["_output_file"]    = str(out_file)

    return output
