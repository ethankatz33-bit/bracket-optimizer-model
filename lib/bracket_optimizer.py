"""
lib/bracket_optimizer.py
Seed-level NCAA bracket optimizer.

All statistics are derived exclusively from the historical dataset
(1985–2016, 32 seasons).  No hardcoded seed trivia; everything flows
from cleaned_games.csv and seed_probabilities.json.

Public API
----------
  load_data()                         → (df, probs)
  compute_seed_round_distribution(df) → per-round seed frequency dicts
  compute_optimal_upset_profile(df)   → upset stats + target ranges by round
  generate_optimal_bracket_structure()→ conservative / balanced / upset-heavy
  score_bracket_structure(structure)  → float plausibility score
  generate_multiple_structures(n)     → list of n structures across the spectrum
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
GAMES_FILE   = PROJECT_ROOT / "data" / "processed" / "cleaned_games.csv"
PROBS_FILE   = PROJECT_ROOT / "data" / "processed" / "seed_probabilities.json"
OUTPUT_FILE  = PROJECT_ROOT / "data" / "processed" / "optimal_bracket_structure.json"

# ── Constants ────────────────────────────────────────────────────────────────
ROUND_NUM_TO_NAME = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}

# How many teams are in each bracket slot
BRACKET_SIZES = {
    "round_of_32":   32,
    "sweet_16":      16,
    "elite_8":        8,
    "final_four":     4,
    "championship":   2,
    "champion":       1,
}

# Which preceding round's ADVANCES feed each slot
# e.g. teams in the Sweet 16 = teams that WON their Round of 32 game
SLOT_SOURCE_ROUND = {
    "round_of_32":   "Round of 64",
    "sweet_16":      "Round of 32",
    "elite_8":       "Sweet 16",
    "final_four":    "Elite 8",
    "championship":  "Final Four",
    "champion":      "Championship",
}

# advancement_rates key used when scoring each bracket slot
SLOT_SCORING_KEY = {
    "round_of_32":   "round_64",
    "sweet_16":      "sweet_16",
    "elite_8":       "elite_8",
    "final_four":    "final_four",
    "championship":  "final_four",  # best available proxy (reaching FF ≈ reaching game)
    "champion":      "champion",
}

# Alpha exponents per mode for the power-law bias transformation
MODE_ALPHA = {
    "conservative": 2.0,   # amplifies differences → low seeds dominate
    "balanced":     1.0,   # historical averages, unmodified
    "upset_heavy":  0.5,   # flattens distribution → high seeds get more slots
}


# ════════════════════════════════════════════════════════════════════════════
# 1. Data loading
# ════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, dict]:
    """
    Load and return:
      df    — cleaned_games.csv as a DataFrame
      probs — seed_probabilities.json as a dict
    """
    df = pd.read_csv(GAMES_FILE)
    with open(PROBS_FILE) as f:
        probs = json.load(f)
    return df, probs


# ════════════════════════════════════════════════════════════════════════════
# 2. Seed-round distribution
# ════════════════════════════════════════════════════════════════════════════

def compute_seed_round_distribution(df: pd.DataFrame) -> dict:
    """
    For every round, compute how often each seed appears and how often it
    advances, both expressed as per-year averages.

    Structure:
      {
        "Round of 64": {
          "appearances": {1: 4.0, 2: 4.0, ... 16: 4.0},   ← all 4 appear
          "advances":    {1: 4.0, 2: 3.75, ... 16: 0.01},  ← avg winners
        },
        "Round of 32": { ... },
        ...
      }

    Key relationship:
      dist[round_R]["advances"]  ==  expected seed composition of round R+1
      Their values sum to the bracket size of round R+1.
    """
    n_years = df["year"].nunique()
    dist = {}

    for rnum, rname in ROUND_NUM_TO_NAME.items():
        r = df[df["round"] == rnum]
        appearances: dict[int, float] = {}
        advances:    dict[int, float] = {}

        for seed in range(1, 17):
            n_appear  = int(((r["winning_seed"] == seed) |
                             (r["losing_seed"]  == seed)).sum())
            n_advance = int((r["winning_seed"] == seed).sum())
            appearances[seed] = round(n_appear  / n_years, 4)
            advances[seed]    = round(n_advance / n_years, 4)

        dist[rname] = {"appearances": appearances, "advances": advances}

    return dist


# ════════════════════════════════════════════════════════════════════════════
# 3. Upset profile
# ════════════════════════════════════════════════════════════════════════════

def compute_optimal_upset_profile(df: pd.DataFrame) -> dict:
    """
    Per-round upset statistics across all historical seasons.

    Returns:
      {
        "Round of 64": {
          "average":    6.16,
          "median":     6.0,
          "std":        1.95,
          "target_min": 4,     # floor(avg - 1 std)
          "target_max": 8,     # ceil(avg + 1 std), capped at max games
        },
        ...
      }

    The 8-vs-9 no-upset rule is already encoded in the cleaned data's
    'upset' column, so no special handling is needed here.
    """
    # Maximum possible upsets per round (all games)
    max_games = {1: 32, 2: 16, 3: 8, 4: 4, 5: 2, 6: 1}
    profile = {}

    for rnum, rname in ROUND_NUM_TO_NAME.items():
        per_year = (
            df[df["round"] == rnum]
            .groupby("year")["upset"]
            .sum()
            .tolist()
        )
        if not per_year:
            continue

        avg = statistics.mean(per_year)
        med = statistics.median(per_year)
        std = statistics.stdev(per_year) if len(per_year) > 1 else 0.0

        t_min = max(0,               round(avg - std))
        t_max = min(max_games[rnum], round(avg + std))

        profile[rname] = {
            "average":    round(avg, 2),
            "median":     round(med, 1),
            "std":        round(std, 2),
            "target_min": int(t_min),
            "target_max": int(t_max),
        }

    return profile


# ════════════════════════════════════════════════════════════════════════════
# Internal helpers for bracket generation
# ════════════════════════════════════════════════════════════════════════════

def _apply_mode_bias(advances: dict[int, float], alpha: float) -> dict[int, float]:
    """
    Power-law transformation: rate → rate^alpha.

    alpha > 1 amplifies differences between seeds (conservative/chalky).
    alpha < 1 flattens differences (upset-heavy).
    Seeds with zero historical advances stay at zero regardless of alpha —
    we never invent occurrences that did not happen in the real data.
    """
    return {
        seed: (rate ** alpha if rate > 0 else 0.0)
        for seed, rate in advances.items()
    }


def _largest_remainder(weights: dict[int, float], n_slots: int) -> dict[int, int]:
    """
    Hamilton's apportionment method.
    Allocates exactly n_slots among seeds proportional to weights.
    Ties in the remainder step are broken by seed number (lower seed wins).
    """
    total = sum(weights.values())
    if total == 0:
        return {seed: 0 for seed in weights}

    exact      = {s: w * n_slots / total  for s, w in weights.items()}
    floors     = {s: int(v)               for s, v in exact.items()}
    remainders = {s: exact[s] - floors[s] for s in exact}

    n_remaining = n_slots - sum(floors.values())
    # Sort by remainder descending, seed ascending (lower seed breaks ties)
    priority = sorted(remainders, key=lambda s: (-remainders[s], s))
    for seed in priority[:n_remaining]:
        floors[seed] += 1

    return floors


def _counts_to_list(counts: dict[int, int]) -> list[int]:
    """Expand {seed: count} into a sorted list: {1:2, 5:1} → [1, 1, 5]."""
    out = []
    for seed in range(1, 17):
        out.extend([seed] * counts.get(seed, 0))
    return out


def _upset_target(round_name: str, upset_profile: dict, mode: str) -> int:
    """
    Return the upset-count target for a given round/mode, derived from
    historical statistics.  Clamped to [target_min, target_max].
    """
    stats = upset_profile.get(round_name, {})
    avg   = stats.get("average", 0.0)
    std   = stats.get("std",     0.0)

    # Conservative → below average; upset_heavy → above average
    offsets = {"conservative": -0.5, "balanced": 0.0, "upset_heavy": 0.5}
    raw = avg + offsets[mode] * std

    t_min = stats.get("target_min", 0)
    t_max = stats.get("target_max", 0)
    return int(round(max(t_min, min(t_max, raw))))


def _build_single_structure(
    alpha:         float,
    dist:          dict,
    upset_profile: dict,
    adv_rates:     dict,
    mode:          str,
) -> dict:
    """
    Core generation routine.  Produces one complete bracket structure given
    an alpha bias value and the pre-loaded data objects.
    """
    bracket: dict = {}

    for slot, src_round in SLOT_SOURCE_ROUND.items():
        raw_advances = dist[src_round]["advances"]
        adjusted     = _apply_mode_bias(raw_advances, alpha)
        n_slots      = BRACKET_SIZES[slot]

        if slot == "champion":
            # Single champion: pick seed with highest adjusted rate
            # Falls back to seed 1 if all rates are zero (shouldn't happen)
            valid = {s: v for s, v in adjusted.items() if v > 0}
            bracket["champion"] = int(min(valid, key=lambda s: -valid[s])) \
                if valid else 1
        else:
            counts = _largest_remainder(adjusted, n_slots)
            bracket[slot] = _counts_to_list(counts)

    # Upset targets derived from historical statistics + mode offset
    bracket["upset_profile"] = {
        rname: _upset_target(rname, upset_profile, mode)
        for rname in ROUND_NUM_TO_NAME.values()
        if rname in upset_profile
    }

    bracket["score"] = _score(bracket, adv_rates)
    return bracket


def _score(structure: dict, adv_rates: dict) -> float:
    """
    Compute the mean historical advancement probability across all bracket
    slots.  Each seed's rate for its round is looked up in adv_rates.

    Interpretation:
      Higher score = more seeds that historically reach their assigned round.
      Conservative brackets score higher; upset-heavy brackets score lower.
      Absolute value is less meaningful than relative comparisons.
    """
    all_probs: list[float] = []

    for slot, adv_key in SLOT_SCORING_KEY.items():
        seeds = structure.get(slot, [])
        if isinstance(seeds, int):
            seeds = [seeds]
        for seed in seeds:
            rate = adv_rates.get(str(seed), {}).get(adv_key, 0.0)
            all_probs.append(rate)

    return round(sum(all_probs) / len(all_probs), 4) if all_probs else 0.0


# ════════════════════════════════════════════════════════════════════════════
# 4. Generate the three named bracket structures
# ════════════════════════════════════════════════════════════════════════════

def generate_bracket_structure_from_data(
    df: pd.DataFrame, adv_rates: dict
) -> dict:
    """
    Like generate_optimal_bracket_structure() but accepts an already-loaded
    DataFrame and advancement-rates dict instead of reading from disk.

    Used by the backtest engine to produce a pre-tournament-only structure
    (with df filtered to years before the target tournament).

    Returns the same shape as generate_optimal_bracket_structure():
      { "conservative": {...}, "balanced": {...}, "upset_heavy": {...} }
    """
    dist          = compute_seed_round_distribution(df)
    upset_profile = compute_optimal_upset_profile(df)
    return {
        mode: _build_single_structure(
            alpha=MODE_ALPHA[mode],
            dist=dist,
            upset_profile=upset_profile,
            adv_rates=adv_rates,
            mode=mode,
        )
        for mode in ("conservative", "balanced", "upset_heavy")
    }


def generate_optimal_bracket_structure() -> dict:
    """
    Produce conservative, balanced, and upset-heavy bracket structures.

    Method
    ------
    For each round, the expected seed composition is taken from the historical
    per-year average advancement counts (dist[round]["advances"]).  An alpha
    power transformation biases that distribution:

      conservative (α=2.0) → amplifies seed-quality differences; 1-seeds
                              crowd out upsets
      balanced     (α=1.0) → raw historical averages; matches observed
                              seed distributions
      upset_heavy  (α=0.5) → flattens the distribution; high seeds claim
                              more slots without violating historical zeros

    Hamilton's largest-remainder method converts the continuous distribution
    into an integer list that sums to the exact bracket size for each round.

    Returns
    -------
    {
      "conservative": { "round_of_32": [...], ..., "champion": 1,
                        "upset_profile": {...}, "score": 0.82 },
      "balanced":     { ... },
      "upset_heavy":  { ... },
    }
    """
    df, probs   = load_data()
    dist         = compute_seed_round_distribution(df)
    upset_profile = compute_optimal_upset_profile(df)
    adv_rates    = probs["advancement_rates"]

    return {
        mode: _build_single_structure(
            alpha=MODE_ALPHA[mode],
            dist=dist,
            upset_profile=upset_profile,
            adv_rates=adv_rates,
            mode=mode,
        )
        for mode in ("conservative", "balanced", "upset_heavy")
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. Score a bracket structure (public API)
# ════════════════════════════════════════════════════════════════════════════

def score_bracket_structure(structure: dict) -> float:
    """
    Score a bracket structure using historical advancement probabilities.

    Accepts any dict with the same shape as generate_optimal_bracket_structure
    output: keys round_of_32, sweet_16, elite_8, final_four, championship,
    champion (each a list of seeds or a single seed int).

    Returns the mean advancement probability across all bracket slots.
    Higher = more historically plausible.
    """
    _, probs = load_data()
    return _score(structure, probs["advancement_rates"])


# ════════════════════════════════════════════════════════════════════════════
# 6. Generate n structures across the spectrum
# ════════════════════════════════════════════════════════════════════════════

def generate_multiple_structures(n: int = 10) -> list[dict]:
    """
    Generate n bracket structures spanning the conservative → upset-heavy
    spectrum, sorted by score (most plausible first).

    The three named modes are always included.  Remaining slots are filled
    by interpolating alpha values uniformly across [0.4, 2.5].

    Each entry in the returned list adds a "label" field.
    """
    df, probs   = load_data()
    dist         = compute_seed_round_distribution(df)
    upset_profile = compute_optimal_upset_profile(df)
    adv_rates    = probs["advancement_rates"]

    # Always include the three named modes
    results = []
    for mode in ("conservative", "balanced", "upset_heavy"):
        s = _build_single_structure(MODE_ALPHA[mode], dist, upset_profile,
                                    adv_rates, mode)
        s["label"] = mode
        results.append(s)

    # Fill remaining with interpolated alpha values
    n_extra = max(0, n - 3)
    if n_extra > 0:
        # Spread across the full alpha range, excluding the three fixed points
        import numpy as np
        alphas = [a for a in np.linspace(0.4, 2.5, n_extra + 6)
                  if not any(abs(a - fa) < 0.05
                             for fa in MODE_ALPHA.values())][:n_extra]

        for alpha in alphas:
            # Map alpha to an approximate mode for upset_profile targeting
            if alpha >= 1.5:
                mode_label = "conservative"
            elif alpha <= 0.7:
                mode_label = "upset_heavy"
            else:
                mode_label = "balanced"

            s = _build_single_structure(alpha, dist, upset_profile,
                                        adv_rates, mode_label)
            s["label"] = f"alpha_{round(alpha, 2)}"
            results.append(s)

    # Sort by score descending (most plausible first); cap to n
    return sorted(results, key=lambda x: x["score"], reverse=True)[:n]
