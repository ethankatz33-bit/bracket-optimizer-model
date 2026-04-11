"""
lib/team_selector.py
Team Selection Engine — fills the optimized seed bracket with actual teams.

The 64-team field uses placeholder ratings.  Swap MOCK_TEAMS for a
live data source to connect real season stats.

Pipeline
--------
  1. Score each team (rating + seed quality + champion profile)
  2. For each potential upset, compute:
       win_probability  — historical rate this underdog seed beats this favorite seed
       upset_quality    — win_probability × (underdog_score / favorite_score)
  3. Select upsets that clear both per-mode thresholds:
       UPSET_MIN_WIN_PROB     — minimum historical win rate (hard gate)
       UPSET_MIN_DESIRABILITY — minimum composite quality score (soft gate)
     No fixed count is applied; the number of upsets is determined by how many
     matchups clear the thresholds in each round.
  4. Apply the Sweet 16 composition constraint (enforces DD seed presence)
  5. Simulate every round, attach upset_quality to each upset game result
  6. Report structure compliance — upset counts are informational only;
     the only enforced constraint is Sweet 16 double-digit seed composition
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OPT_FILE     = PROJECT_ROOT / "data" / "processed" / "optimal_bracket_structure.json"
PROBS_FILE   = PROJECT_ROOT / "data" / "processed" / "seed_probabilities.json"
OUTPUT_FILE  = PROJECT_ROOT / "data" / "processed" / "generated_bracket.json"

# ── Bracket layout constants ──────────────────────────────────────────────────

REGIONS = ["East", "West", "South", "Midwest"]

# Standard first-round seed matchups within each region.
# Each tuple is (lo_seed, hi_seed): lo_seed is the favorite, hi_seed the underdog.
R64_MATCHUPS = [(1,16), (8,9), (5,12), (4,13), (6,11), (3,14), (7,10), (2,15)]

# Round-of-32: pair R64 game-slot winners (0,1), (2,3), (4,5), (6,7)
R32_PAIRS = [(0,1), (2,3), (4,5), (6,7)]

# Sweet 16: pair R32 game-slot winners (0,1), (2,3)
S16_PAIRS = [(0,1), (2,3)]

# Final Four: East vs West, South vs Midwest (indices into REGIONS list)
FF_REGION_PAIRS = [(0,1), (2,3)]   # (East,West), (South,Midwest)

# ── Mock team field (replace with live data for a real season) ────────────────
# Ratings are on a 20–96 scale calibrated to seed tiers with intra-tier variance.
# One team per seed per region = 64 teams total.

MOCK_TEAMS: dict[str, dict[int, dict]] = {
    "East": {
        1:  {"name": "Duke",             "rating": 94.0},
        2:  {"name": "Villanova",        "rating": 87.5},
        3:  {"name": "North Carolina",   "rating": 81.5},
        4:  {"name": "Georgetown",       "rating": 76.0},
        5:  {"name": "Connecticut",      "rating": 71.5},
        6:  {"name": "Texas",            "rating": 66.5},
        7:  {"name": "Mississippi St",   "rating": 62.0},
        8:  {"name": "California",       "rating": 57.5},
        9:  {"name": "USC",              "rating": 52.5},
        10: {"name": "Colorado",         "rating": 48.5},
        11: {"name": "Providence",       "rating": 44.5},
        12: {"name": "New Mexico St",    "rating": 41.0},
        13: {"name": "Murray State",     "rating": 37.5},
        14: {"name": "Montana",          "rating": 33.5},
        15: {"name": "UC Santa Barbara", "rating": 29.0},
        16: {"name": "Norfolk State",    "rating": 23.5},
    },
    "West": {
        1:  {"name": "Kansas",           "rating": 93.0},
        2:  {"name": "Michigan State",   "rating": 86.0},
        3:  {"name": "Indiana",          "rating": 82.5},
        4:  {"name": "Purdue",           "rating": 77.5},
        5:  {"name": "Iowa State",       "rating": 72.0},
        6:  {"name": "Clemson",          "rating": 67.5},
        7:  {"name": "Wichita State",    "rating": 63.0},
        8:  {"name": "Penn State",       "rating": 58.0},
        9:  {"name": "Richmond",         "rating": 53.0},
        10: {"name": "BYU",              "rating": 49.5},
        11: {"name": "Marquette",        "rating": 45.5},
        12: {"name": "Missouri State",   "rating": 40.5},
        13: {"name": "Chattanooga",      "rating": 37.0},
        14: {"name": "Mercer",           "rating": 33.0},
        15: {"name": "Oral Roberts",     "rating": 29.5},
        16: {"name": "Hampton",          "rating": 23.0},
    },
    "South": {
        1:  {"name": "Kentucky",         "rating": 92.0},
        2:  {"name": "Arizona",          "rating": 88.0},
        3:  {"name": "Louisville",       "rating": 80.5},
        4:  {"name": "UCLA",             "rating": 78.0},
        5:  {"name": "Wisconsin",        "rating": 72.5},
        6:  {"name": "Georgia Tech",     "rating": 68.0},
        7:  {"name": "Texas Tech",       "rating": 61.5},
        8:  {"name": "Seton Hall",       "rating": 56.5},
        9:  {"name": "TCU",              "rating": 53.5},
        10: {"name": "Oregon",           "rating": 50.0},
        11: {"name": "Loyola Chicago",   "rating": 46.0},
        12: {"name": "Oregon State",     "rating": 42.5},
        13: {"name": "Vermont",          "rating": 36.5},
        14: {"name": "Winthrop",         "rating": 32.5},
        15: {"name": "Jacksonville St",  "rating": 28.0},
        16: {"name": "Abilene Christian","rating": 22.5},
    },
    "Midwest": {
        1:  {"name": "Gonzaga",          "rating": 95.5},
        2:  {"name": "Florida",          "rating": 85.0},
        3:  {"name": "Syracuse",         "rating": 83.0},
        4:  {"name": "Arkansas",         "rating": 75.5},
        5:  {"name": "Iowa",             "rating": 71.0},
        6:  {"name": "Utah",             "rating": 66.0},
        7:  {"name": "Alabama",          "rating": 62.5},
        8:  {"name": "Notre Dame",       "rating": 57.0},
        9:  {"name": "Davidson",         "rating": 51.5},
        10: {"name": "Pittsburgh",       "rating": 47.5},
        11: {"name": "St. Louis",        "rating": 43.0},
        12: {"name": "Cincinnati",       "rating": 39.5},
        13: {"name": "Bucknell",         "rating": 36.0},
        14: {"name": "Bradley",          "rating": 32.0},
        15: {"name": "Pacific",          "rating": 28.5},
        16: {"name": "Texas Southern",   "rating": 22.0},
    },
}


# ════════════════════════════════════════════════════════════════════════════
# Team scoring
# ════════════════════════════════════════════════════════════════════════════

def score_team(team: dict, adv_rates: dict) -> float:
    """
    Composite team quality score in [0, 1].

    Components
    ----------
    rating_score  (60 %) — normalised placeholder rating (20-96 scale)
    seed_quality  (25 %) — linear seed advantage: seed 1 = 1.0, seed 16 = 0.0
    champ_profile (15 %) — historical probability that a team of this seed
                           wins the championship (from advancement_rates)

    The champion-profile term rewards seeds with a realistic track record
    of going deep, slightly boosting teams that are both high-rated AND
    from a historically strong seed tier.
    """
    seed   = team["seed"]
    rating = team["rating"]

    rating_score  = (rating - 20) / 76.0          # normalise to [0, 1]
    seed_quality  = (17 - seed) / 16.0             # seed 1 → 1.0, seed 16 → 0.0625

    # Champion profile: scale seed 1's rate (~0.148) to 1.0
    champ_rate    = adv_rates.get(str(seed), {}).get("champion", 0.0)
    max_rate      = adv_rates.get("1", {}).get("champion", 0.148)
    champ_profile = min(1.0, champ_rate / max_rate) if max_rate > 0 else 0.0

    return round(0.60 * rating_score + 0.25 * seed_quality + 0.15 * champ_profile, 4)


# ════════════════════════════════════════════════════════════════════════════
# Upset-selection thresholds
# ════════════════════════════════════════════════════════════════════════════

# Eligible underdog seed range, by round.
# Early rounds (R64–S16): only double-digit seeds (10–13) are eligible underdogs.
#   Seeds 14–16 win too rarely to pick reliably; seeds 8–9 are excluded entirely.
# Late rounds (E8+): expand to 2–13 so that 2/3/4-seeds can beat 1-seeds.
#   These matchups are historically plausible (2v1 ≈44%, 3v1 ≈31%) and produce
#   realistic Final Four variety.
ROUND_SEED_RANGE: dict[str, tuple[int, int]] = {
    "Round of 64":  (10, 13),
    "Round of 32":  (10, 13),
    "Sweet 16":     (10, 13),
    "Elite 8":      (2,  13),
    "Final Four":   (2,  13),
    "Championship": (2,  13),
}

# Minimum historical win probability for the underdog seed in this matchup type.
# Acts as a hard gate; calibrated from 1985–2016 data (seed_probabilities.json):
#   R64 win rates: 10v7≈39%, 11v6≈36%, 12v5≈36%, 13v4≈20%
# Conservative admits only the top tier (≥28%); balanced and upset_heavy
# also admit the 13v4 tier (≥20%).
UPSET_MIN_WIN_PROB: dict[str, float] = {
    "conservative": 0.28,
    "balanced":     0.22,
    "upset_heavy":  0.22,
}

# Minimum upset desirability score: win_probability × (underdog_score / favorite_score).
# Acts as a quality gate: excludes matchups where the underdog is too weak
# relative to the favorite, even if the seed matchup has historical precedent.
UPSET_MIN_DESIRABILITY: dict[str, float] = {
    "conservative": 0.12,
    "balanced":     0.07,
    "upset_heavy":  0.04,
}

# Per-round relaxation of UPSET_MIN_WIN_PROB[mode] (negative = more permissive).
# Later rounds feature stronger surviving teams; the early-round threshold is
# too strict for matchups like 2v1 or 3v1, which are historically common.
ROUND_PROB_DELTA: dict[str, float] = {
    "Sweet 16":    -0.03,
    "Elite 8":     -0.05,
    "Final Four":  -0.05,
    "Championship": -0.05,
}

# Hard cap on total upset picks per round, by mode.
# Candidates are ranked by desirability; only the top N are selected.
# E8/FF/Championship caps are intentionally loose — the gate thresholds and
# FF balance bias are the primary controls in late rounds.
UPSET_MAX_BY_ROUND: dict[str, dict[str, int]] = {
    "Round of 64":  {"conservative": 5,  "balanced": 7,  "upset_heavy": 9},
    "Round of 32":  {"conservative": 2,  "balanced": 3,  "upset_heavy": 4},
    "Sweet 16":     {"conservative": 1,  "balanced": 1,  "upset_heavy": 2},
    "Elite 8":      {"conservative": 1,  "balanced": 2,  "upset_heavy": 2},
    "Final Four":   {"conservative": 1,  "balanced": 1,  "upset_heavy": 2},
    "Championship": {"conservative": 1,  "balanced": 1,  "upset_heavy": 1},
}

# Per-region caps for Round of 64 only.
# Prevents all three mid-tier upsets (10v7, 11v6, 12v5) from firing in the
# same region, which would cascade into DD-vs-DD paths in later rounds.
UPSET_MAX_PER_REGION_R64      = 2   # total upsets per region
UPSET_MAX_MID_TIER_PER_REGION = 2   # of the three types 10v7 / 11v6 / 12v5

# The three mid-tier R64 matchup types that cause cascade problems when all fire
_MID_TIER_MATCHUPS: frozenset[tuple[int, int]] = frozenset({(5, 12), (6, 11), (7, 10)})

# Soft Final Four balance bias — applied to E8 upset candidates' desirability
# scores (for ranking and gate purposes) based on how many 1-seeds are projected
# to reach the FF by default.  Encourages ~2 1-seeds; discourages 0 or 4.
# Does NOT block any outcome — only shifts the probability of selection.
_FF_BALANCE_MULTIPLIER: dict[int, float] = {
    4: 1.50,   # all 4 games have 1-seed favorites → strongly boost upsets
    3: 1.20,   # 3 of 4 → mild boost
    2: 1.00,   # 2 of 4 → neutral (ideal target state)
    1: 0.70,   # 1 of 4 → mild penalty on further upsets
    0: 0.40,   # 0 of 4 → strong penalty
}

# Soft gate for the second Elite 8 upset in balanced mode only.
# After the first E8 upset is selected, any subsequent candidate must have
# biased desirability (post-FF-balance-multiplier) above this value to be
# selected.  This discourages the second upset without blocking it outright:
# strong matchups (high biased desirability) still pass; weak ones are skipped.
#
# Calibrated from 1990–2016 pre-tournament win rates:
#   biased desirability for 2v1 E8 ranges 0.476–0.597 (mean ≈ 0.530)
#   gate=0.540 → second upset passes in ~9/22 measured years → ~65% 1-seeds
_BALANCED_E8_SECOND_GATE: float = 0.540


# ════════════════════════════════════════════════════════════════════════════
# Game-selection helpers
# ════════════════════════════════════════════════════════════════════════════

def _is_8v9(seed_a: int, seed_b: int) -> bool:
    """8-vs-9 matchups are treated as pick'em; no upset is counted."""
    return {seed_a, seed_b} == {8, 9}


def _upset_rate(fav_seed: int, und_seed: int, win_rates: dict) -> float:
    """
    Historical win rate of the underdog (und_seed) over the favorite (fav_seed).
    Key format matches matchup_win_rates: "{higher}_vs_{lower}".
    Returns 0.0 when there is no historical precedent.
    """
    key = f"{und_seed}_vs_{fav_seed}"
    return win_rates.get(key, 0.0)


def _upset_desirability(team_a: dict, team_b: dict, win_rates: dict) -> float:
    """
    How warranted is an upset pick for this matchup?

      desirability = historical_upset_rate × (underdog_score / favorite_score)

    Determines favorite / underdog from seed numbers internally so the
    function is correct regardless of which team is passed first.
    Returns 0.0 for 8v9 games or matchups with no historical upset record.
    """
    if _is_8v9(team_a["seed"], team_b["seed"]):
        return 0.0
    # Lower seed number = better team = favorite
    if team_a["seed"] < team_b["seed"]:
        fav, und = team_a, team_b
    elif team_b["seed"] < team_a["seed"]:
        fav, und = team_b, team_a
    else:
        return 0.0   # same seed — treat as no upset
    rate = _upset_rate(fav["seed"], und["seed"], win_rates)
    if rate == 0.0:
        return 0.0
    score_ratio = und["score"] / fav["score"] if fav["score"] > 0 else 0.0
    return round(rate * score_ratio, 5)


def _select_upset_indices(
    matchups:   list[tuple[dict, dict]],
    win_rates:  dict,
    mode:       str = "balanced",
    round_name: str = "",
    regions:    list[str] | None = None,
) -> dict[int, float]:
    """
    Select upset picks using per-round gates, seed-range rules, and hard caps.

    Eligibility gates (all must pass)
    ----------------------------------
    1. Underdog seed in ROUND_SEED_RANGE[round_name].
       Early rounds (R64–S16): 10–13 only; seeds 8/9 excluded, 14–16 too rare.
       Late rounds (E8+): 2–13, allowing 2/3/4-seeds to beat 1-seeds.
    2. Historical win probability ≥ UPSET_MIN_WIN_PROB[mode] + ROUND_PROB_DELTA[round].
       Later rounds apply a small relaxation (–0.03 to –0.05) to admit the
       historically common 2v1 and 3v1 matchups without requiring a threshold change.
    3. Composite desirability ≥ UPSET_MIN_DESIRABILITY[mode].

    Ranking and caps
    ----------------
    • Eligible candidates are ranked by desirability (highest first).
    • UPSET_MAX_BY_ROUND[round_name][mode] caps the total picks.
    • Round of 64 only: per-region caps and mid-tier cluster cap enforce
      UPSET_MAX_PER_REGION_R64 and UPSET_MAX_MID_TIER_PER_REGION.

    Final Four soft balance (Elite 8 only)
    ---------------------------------------
    Before ranking, desirability scores for E8 candidates are scaled by
    _FF_BALANCE_MULTIPLIER[projected_1seeds_in_ff].  This is a soft scoring
    adjustment, not a hard constraint: it shifts which upsets rank highest
    when the cap is binding, nudging toward ~2 1-seeds in the Final Four
    without blocking any specific outcome.

    Returns
    -------
    dict mapping matchup index → upset_quality score.
    Indices absent from the dict are resolved as favorite wins.
    """
    base_min_prob = UPSET_MIN_WIN_PROB.get(mode, 0.22)
    min_desir     = UPSET_MIN_DESIRABILITY.get(mode, 0.09)
    prob_delta    = ROUND_PROB_DELTA.get(round_name, 0.0)
    effective_min_prob = max(0.0, base_min_prob + prob_delta)
    seed_min, seed_max = ROUND_SEED_RANGE.get(round_name, (10, 13))

    # ── Build eligible candidates ──────────────────────────────────────────
    candidates: list[tuple[int, float, dict, dict, str]] = []

    for i, (team_a, team_b) in enumerate(matchups):
        # 8/9 games: never count as upsets
        if _is_8v9(team_a["seed"], team_b["seed"]):
            continue

        # Identify favorite and underdog by seed number
        if team_a["seed"] < team_b["seed"]:
            fav, und = team_a, team_b
        elif team_b["seed"] < team_a["seed"]:
            fav, und = team_b, team_a
        else:
            continue   # same seed — no upset possible

        # Gate 0: underdog seed in allowed range for this round
        if not (seed_min <= und["seed"] <= seed_max):
            continue

        # Gate 1: historical win probability (with round-specific relaxation)
        win_prob = _upset_rate(fav["seed"], und["seed"], win_rates)
        if win_prob < effective_min_prob:
            continue

        # Gate 2: composite desirability
        desir = _upset_desirability(team_a, team_b, win_rates)
        if desir < min_desir:
            continue

        region = regions[i] if regions else ""
        candidates.append((i, desir, fav, und, region))

    # ── E8 Final Four soft balance adjustment ─────────────────────────────
    # Count how many E8 games have a 1-seed as the lower-seed (default winner).
    # Apply a desirability multiplier that steers toward ~2 1-seeds in the FF.
    if round_name == "Elite 8" and candidates:
        projected_1seeds = sum(
            1 for a, b in matchups
            if min(a["seed"], b["seed"]) == 1
        )
        ff_bias = _FF_BALANCE_MULTIPLIER.get(projected_1seeds, 1.0)
        candidates = [(i, d * ff_bias, fav, und, r)
                      for i, d, fav, und, r in candidates]

    # ── Rank by (adjusted) desirability descending ────────────────────────
    candidates.sort(key=lambda x: -x[1])

    # ── Greedily select within caps ───────────────────────────────────────
    total_cap = UPSET_MAX_BY_ROUND.get(round_name, {}).get(mode, len(candidates))

    selected:       dict[int, float] = {}
    region_count:   dict[str, int]   = {}
    region_midtier: dict[str, int]   = {}

    for i, desir, fav, und, region in candidates:
        if len(selected) >= total_cap:
            break

        # Soft gate: second E8 upset in balanced mode requires higher quality.
        # desir here is the biased value (post-FF-balance-multiplier), so the
        # gate is applied on the same scale used for ranking.
        if (round_name == "Elite 8" and mode == "balanced"
                and len(selected) >= 1
                and desir < _BALANCED_E8_SECOND_GATE):
            continue

        if round_name == "Round of 64":
            if region_count.get(region, 0) >= UPSET_MAX_PER_REGION_R64:
                continue
            if (fav["seed"], und["seed"]) in _MID_TIER_MATCHUPS:
                if region_midtier.get(region, 0) >= UPSET_MAX_MID_TIER_PER_REGION:
                    continue
                region_midtier[region] = region_midtier.get(region, 0) + 1
            region_count[region] = region_count.get(region, 0) + 1

        selected[i] = round(desir, 5)

    return selected


# Sweet 16 DD target ranges per mode
_S16_DD_TARGETS: dict[str, tuple[int, int]] = {
    "conservative": (0, 1),
    "balanced":     (1, 1),
    "upset_heavy":  (1, 2),
}


def _enforce_s16_dd_constraint(
    matchups:   list[tuple[dict, dict]],
    upset_map:  dict[int, float],
    mode:       str,
    win_rates:  dict,
) -> tuple[dict[int, float], list[str]]:
    """
    Post-process the R32 upset selection so the number of double-digit seeds
    (seed ≥ 10) that advance to the Sweet 16 satisfies the mode's target range.

    Rules
    -----
    conservative : 0 – 1  (trim excess; don't force one in)
    balanced     : exactly 1  (trim excess OR force one in)
    upset_heavy  : 1 – 2  (force one in if 0; trim if >2)

    Strategy
    --------
    • Too many DD winners → remove the lowest-scoring DD winner(s) by
      flipping their game back to the single-digit team.
    • Too few DD winners  → promote the most plausible DD loser (ranked by
      upset desirability then team score) by flipping their game.

    Flipping a game means toggling its index in the upset set so the desired
    team wins.  The upset-count for R32 may shift by ±1 per forced flip —
    structural compliance takes priority over the upset-count target when
    the two conflict.

    Returns
    -------
    adjusted_indices : updated upset set
    notes            : human-readable strings describing each forced change
    """
    notes: list[str] = []

    if mode not in _S16_DD_TARGETS:
        return upset_map, notes

    target_min, target_max = _S16_DD_TARGETS[mode]
    adjusted: dict[int, float] = dict(upset_map)   # index → quality score

    def _sim_winner_loser(i: int) -> tuple[dict, dict]:
        """Simulate game i and return (winner, loser) given current adjusted."""
        team_a, team_b = matchups[i]
        if _is_8v9(team_a["seed"], team_b["seed"]):
            if team_a["score"] >= team_b["score"]:
                return team_a, team_b
            return team_b, team_a
        fav, und = (
            (team_a, team_b) if team_a["seed"] < team_b["seed"]
            else (team_b, team_a)
        )
        return (und, fav) if i in adjusted else (fav, und)

    def _flip_to_win(dd_team: dict, i: int, quality: float) -> None:
        """Toggle i so dd_team wins, recording quality score."""
        team_a, team_b = matchups[i]
        fav = team_a if team_a["seed"] < team_b["seed"] else team_b
        if dd_team["seed"] == fav["seed"]:
            adjusted.pop(i, None)       # dd is fav → normal win (not an upset)
        else:
            adjusted[i] = quality       # dd is und → upset win

    def _flip_to_lose(dd_team: dict, i: int) -> None:
        """Toggle i so dd_team loses."""
        team_a, team_b = matchups[i]
        fav = team_a if team_a["seed"] < team_b["seed"] else team_b
        if dd_team["seed"] == fav["seed"]:
            # dd was winning as favorite; make them lose = now an upset
            q = _upset_desirability(team_a, team_b, win_rates)
            adjusted[i] = q
        else:
            # dd was winning as underdog; remove the upset → favorite wins
            adjusted.pop(i, None)

    # ── Iteratively fix excess DD winners ────────────────────────────────────
    for _ in range(16):
        sim = [_sim_winner_loser(i) for i in range(len(matchups))]
        dd_winners = [(i, w, l) for i, (w, l) in enumerate(sim) if w["seed"] >= 10]
        if len(dd_winners) <= target_max:
            break
        worst_i, worst_dd, replacement = min(dd_winners, key=lambda x: x[1]["score"])
        _flip_to_lose(worst_dd, worst_i)
        notes.append(
            f"S16 constraint ({mode}): removed {worst_dd['name']} "
            f"(seed {worst_dd['seed']}), advanced "
            f"{replacement['name']} (seed {replacement['seed']}) instead"
        )

    # ── Iteratively force DD winners if too few ───────────────────────────────
    for _ in range(16):
        sim = [_sim_winner_loser(i) for i in range(len(matchups))]
        dd_count = sum(1 for w, _ in sim if w["seed"] >= 10)
        if dd_count >= target_min:
            break
        dd_losers = []
        for i, (w, l) in enumerate(sim):
            if l["seed"] < 10:
                continue
            if _is_8v9(matchups[i][0]["seed"], matchups[i][1]["seed"]):
                continue
            d = _upset_desirability(matchups[i][0], matchups[i][1], win_rates)
            dd_losers.append((i, l, w, d))
        if not dd_losers:
            break
        best_i, best_dd, displaced, best_d = max(
            dd_losers, key=lambda x: (x[3], x[1]["score"])
        )
        _flip_to_win(best_dd, best_i, round(best_d, 5))
        notes.append(
            f"S16 constraint ({mode}): forced {best_dd['name']} "
            f"(seed {best_dd['seed']}) to Sweet 16 — displaced "
            f"{displaced['name']} (seed {displaced['seed']})"
        )

    return adjusted, notes


# ════════════════════════════════════════════════════════════════════════════
# Round simulation
# ════════════════════════════════════════════════════════════════════════════

def _play_game(
    team_a:        dict,
    team_b:        dict,
    is_upset_pick: bool,
    round_name:    str,
    region:        str,
    upset_quality: float | None = None,
) -> dict:
    """
    Resolve a single game and return a structured result dict.

    For 8v9: winner is always the higher-rated team; is_upset is forced False.
    For all others: is_upset_pick drives whether the underdog or favorite wins.

    upset_quality, when provided, is attached to the result so the bracket
    output carries a visible ranking of how plausible each upset pick was.
    """
    if _is_8v9(team_a["seed"], team_b["seed"]):
        winner = team_a if team_a["score"] >= team_b["score"] else team_b
        loser  = team_b if winner is team_a else team_a
        is_upset = False
    else:
        if team_a["seed"] < team_b["seed"]:
            fav, und = team_a, team_b
        else:
            fav, und = team_b, team_a

        if is_upset_pick:
            winner, loser, is_upset = und, fav, True
        else:
            winner, loser, is_upset = fav, und, False

    result = {
        "winner":   winner,
        "loser":    loser,
        "is_upset": is_upset,
        "round":    round_name,
        "region":   region,
    }
    if is_upset and upset_quality is not None:
        result["upset_quality"] = upset_quality
    return result


def _simulate_round(
    matchups:      list[tuple[dict, dict, str]],   # (team_a, team_b, region)
    upset_map:     dict[int, float],               # index → upset_quality score
    round_name:    str,
) -> tuple[list[dict], list[dict]]:
    """
    Simulate all games in one round.

    upset_map maps matchup index to upset_quality score for every game
    that should be played as an upset.  Indices absent from the map are
    resolved as favorite wins.

    Returns
    -------
    results  : list of game-result dicts (for bracket output)
    winners  : list of winning team dicts, in matchup order
    """
    results: list[dict] = []
    winners: list[dict] = []

    for i, (team_a, team_b, region) in enumerate(matchups):
        quality = upset_map.get(i)
        result  = _play_game(
            team_a, team_b,
            is_upset_pick=(i in upset_map),
            round_name=round_name,
            region=region,
            upset_quality=quality,
        )
        results.append(result)
        winners.append(result["winner"])

    return results, winners


# ════════════════════════════════════════════════════════════════════════════
# Full bracket simulation
# ════════════════════════════════════════════════════════════════════════════

def simulate_bracket(
    mode: str = "balanced",
    *,
    _teams_override:     dict | None = None,
    _probs_override:     dict | None = None,
    _structure_override: dict | None = None,
) -> dict:
    """
    Simulate a complete NCAA tournament bracket.

    Parameters
    ----------
    mode : "conservative" | "balanced" | "upset_heavy"
           Must match a key in optimal_bracket_structure.json.

    Returns
    -------
    {
      "mode":          str,
      "round_of_64":   [game_result, ...],   # 32 games
      "round_of_32":   [game_result, ...],   # 16 games
      "sweet_16":      [game_result, ...],   # 8 games
      "elite_8":       [game_result, ...],   # 4 games
      "final_four":    [game_result, ...],   # 2 games
      "championship":  game_result,          # 1 game
      "champion":      team_dict,
      "reasoning":     {...},
      "structure_check": {...},
    }
    """
    # ── Load inputs ───────────────────────────────────────────────────────
    # Each input can be overridden (used by the backtest engine to inject
    # pre-tournament-only data without writing to disk).
    if _structure_override is not None:
        opt = _structure_override
    else:
        with open(OPT_FILE) as f: opt = json.load(f)

    if _probs_override is not None:
        probs = _probs_override
    else:
        with open(PROBS_FILE) as f: probs = json.load(f)

    structure     = opt[mode]
    upset_profile = structure["upset_profile"]
    win_rates     = probs["matchup_win_rates"]
    adv_rates     = probs["advancement_rates"]

    # ── Build scored team dicts ───────────────────────────────────────────
    # teams[region][seed] = full team dict including computed score
    teams_source = _teams_override if _teams_override is not None else MOCK_TEAMS
    teams: dict[str, dict[int, dict]] = {}
    for region, seed_map in teams_source.items():
        teams[region] = {}
        for seed, info in seed_map.items():
            t = {**info, "seed": seed, "region": region}
            t["score"] = score_team(t, adv_rates)
            teams[region][seed] = t

    bracket: dict = {
        "mode": mode,
        "round_of_64":  [],
        "round_of_32":  [],
        "sweet_16":     [],
        "elite_8":      [],
        "final_four":   [],
        "championship": None,
        "champion":     None,
    }

    # ── Round of 64 (32 games) ────────────────────────────────────────────
    # Build matchup list preserving region and slot index for later pairing
    r64_matchups: list[tuple[dict, dict, str]] = []
    for region in REGIONS:
        for lo_seed, hi_seed in R64_MATCHUPS:
            r64_matchups.append((teams[region][lo_seed],
                                 teams[region][hi_seed],
                                 region))

    r64_sel = _select_upset_indices(
        [(a, b) for a, b, _ in r64_matchups],
        win_rates=win_rates,
        mode=mode,
        round_name="Round of 64",
        regions=[r for _, _, r in r64_matchups],
    )
    r64_results, r64_winners = _simulate_round(r64_matchups, r64_sel, "Round of 64")
    bracket["round_of_64"] = r64_results

    # Organise R64 winners by region × slot for R32 pairing
    # r64_winners order: 8 per region × 4 regions = 32 entries
    r64_by_region: dict[str, list[dict]] = {r: [] for r in REGIONS}
    for idx, w in enumerate(r64_winners):
        r64_by_region[REGIONS[idx // 8]].append(w)

    # ── Round of 32 (16 games) ────────────────────────────────────────────
    r32_matchups: list[tuple[dict, dict, str]] = []
    for region in REGIONS:
        slots = r64_by_region[region]   # 8 winners, in R64 slot order
        for g1, g2 in R32_PAIRS:
            team_a, team_b = slots[g1], slots[g2]
            r32_matchups.append((team_a, team_b, region))

    r32_sel = _select_upset_indices(
        [(a, b) for a, b, _ in r32_matchups],
        win_rates=win_rates,
        mode=mode,
        round_name="Round of 32",
        regions=[r for _, _, r in r32_matchups],
    )
    r32_sel, s16_notes = _enforce_s16_dd_constraint(
        [(a, b) for a, b, _ in r32_matchups],
        r32_sel,
        mode,
        win_rates,
    )
    r32_results, r32_winners = _simulate_round(r32_matchups, r32_sel, "Round of 32")
    bracket["round_of_32"]       = r32_results
    bracket["s16_constraint_notes"] = s16_notes

    r32_by_region: dict[str, list[dict]] = {r: [] for r in REGIONS}
    for idx, w in enumerate(r32_winners):
        r32_by_region[REGIONS[idx // 4]].append(w)

    # ── Sweet 16 (8 games) ────────────────────────────────────────────────
    s16_matchups: list[tuple[dict, dict, str]] = []
    for region in REGIONS:
        slots = r32_by_region[region]   # 4 winners
        for g1, g2 in S16_PAIRS:
            s16_matchups.append((slots[g1], slots[g2], region))

    s16_sel = _select_upset_indices(
        [(a, b) for a, b, _ in s16_matchups],
        win_rates=win_rates,
        mode=mode,
        round_name="Sweet 16",
        regions=[r for _, _, r in s16_matchups],
    )
    s16_results, s16_winners = _simulate_round(s16_matchups, s16_sel, "Sweet 16")
    bracket["sweet_16"] = s16_results

    s16_by_region: dict[str, list[dict]] = {r: [] for r in REGIONS}
    for idx, w in enumerate(s16_winners):
        s16_by_region[REGIONS[idx // 2]].append(w)

    # ── Elite 8 (4 games, one per region) ────────────────────────────────
    e8_matchups: list[tuple[dict, dict, str]] = []
    for region in REGIONS:
        a, b = s16_by_region[region]
        e8_matchups.append((a, b, region))

    e8_sel = _select_upset_indices(
        [(a, b) for a, b, _ in e8_matchups],
        win_rates=win_rates,
        mode=mode,
        round_name="Elite 8",
        regions=[r for _, _, r in e8_matchups],
    )
    e8_results, e8_winners = _simulate_round(e8_matchups, e8_sel, "Elite 8")
    bracket["elite_8"] = e8_results
    # e8_winners[i] is the champion of REGIONS[i]

    # ── Final Four (2 games) ─────────────────────────────────────────────
    # East vs West, South vs Midwest
    ff_matchups: list[tuple[dict, dict, str]] = []
    for ri, rj in FF_REGION_PAIRS:
        ff_matchups.append((e8_winners[ri], e8_winners[rj], "National"))

    ff_sel = _select_upset_indices(
        [(a, b) for a, b, _ in ff_matchups],
        win_rates=win_rates,
        mode=mode,
        round_name="Final Four",
        regions=[r for _, _, r in ff_matchups],
    )
    ff_results, ff_winners = _simulate_round(ff_matchups, ff_sel, "Final Four")
    bracket["final_four"] = ff_results

    # ── Championship (1 game) ─────────────────────────────────────────────
    champ_matchups = [(ff_winners[0], ff_winners[1], "National")]
    champ_sel = _select_upset_indices(
        [(a, b) for a, b, _ in champ_matchups],
        win_rates=win_rates,
        mode=mode,
        round_name="Championship",
        regions=[r for _, _, r in champ_matchups],
    )
    champ_results, champ_winners = _simulate_round(champ_matchups, champ_sel, "Championship")
    bracket["championship"] = champ_results[0]
    bracket["champion"]     = champ_winners[0]

    # ── Reasoning ────────────────────────────────────────────────────────
    bracket["reasoning"] = _build_reasoning(bracket)

    # ── Structure compliance ──────────────────────────────────────────────
    bracket["structure_check"] = check_structure_compliance(bracket, structure, mode)

    return bracket


# ════════════════════════════════════════════════════════════════════════════
# Reasoning + compliance
# ════════════════════════════════════════════════════════════════════════════

def _build_reasoning(bracket: dict) -> dict:
    """Construct human-readable notes about the bracket."""
    champ    = bracket["champion"]
    runner   = bracket["championship"]["loser"]

    all_games = (
        bracket["round_of_64"] + bracket["round_of_32"] +
        bracket["sweet_16"]    + bracket["elite_8"]     +
        bracket["final_four"]  + [bracket["championship"]]
    )
    upsets = [g for g in all_games if g["is_upset"]]
    notable = [
        f"{g['winner']['name']} (seed {g['winner']['seed']}) over "
        f"{g['loser']['name']} (seed {g['loser']['seed']}) — {g['round']}"
        for g in upsets
        if g["winner"]["seed"] >= 10   # double-digit seed wins only
    ]

    champion_note = (
        f"{champ['name']} (seed {champ['seed']}, rating {champ['rating']}) "
        f"won the championship over "
        f"{runner['name']} (seed {runner['seed']}, rating {runner['rating']}). "
        f"Score differential: {round(champ['score'] - runner['score'], 4):+.4f}. "
        f"Bracket total upsets: {len(upsets)}."
    )
    if champ["seed"] > 2:
        champion_note += (
            f" Cinderella: a seed-{champ['seed']} champion is historically rare."
        )

    return {
        "champion":         champion_note,
        "total_upsets":     len(upsets),
        "notable_upsets":   notable if notable else ["No double-digit seed upsets."],
    }


def check_structure_compliance(
    bracket:   dict,
    structure: dict,
    mode:      str,
) -> dict:
    """
    Compare the generated bracket's seed distribution against the
    optimizer's target structure and upset targets.

    Upset counts are reported as informational only — no pass/fail gate.
    The only enforced structural constraint is Sweet 16 double-digit seed
    composition.  fully_compliant reflects the DD check exclusively.
    """
    from collections import Counter

    def _seeds_in_round(games: list[dict]) -> list[int]:
        """All seeds that WON their game in this round."""
        return [g["winner"]["seed"] for g in games]

    def _upset_count(games: list[dict]) -> int:
        return sum(1 for g in games if g["is_upset"])

    rounds_map = {
        "round_of_32":  ("round_of_64",  "Round of 64"),
        "sweet_16":     ("round_of_32",  "Round of 32"),
        "elite_8":      ("sweet_16",     "Sweet 16"),
        "final_four":   ("elite_8",      "Elite 8"),
        "championship": ("final_four",   "Final Four"),
    }

    upset_compliance: dict[str, dict] = {}
    for bracket_key, (_, round_name) in rounds_map.items():
        source_games = bracket.get(bracket_key.replace("round_of_32", "round_of_64")
                                              .replace("sweet_16",    "round_of_32")
                                              .replace("elite_8",     "sweet_16")
                                              .replace("final_four",  "elite_8")
                                              .replace("championship","final_four"),
                                   [])

    # Simpler: recompute directly from bracket keys
    round_game_keys = {
        "Round of 64": "round_of_64",
        "Round of 32": "round_of_32",
        "Sweet 16":    "sweet_16",
        "Elite 8":     "elite_8",
        "Final Four":  "final_four",
        "Championship": "championship",
    }
    upset_compliance = {}
    for rname, bkey in round_game_keys.items():
        games = bracket.get(bkey, [])
        if isinstance(games, dict):   # championship is a single dict
            games = [games]
        target = structure["upset_profile"].get(rname, "N/A")
        actual = _upset_count(games)
        upset_compliance[rname] = {
            "target":        target,
            "actual":        actual,
            "ok":            True,    # informational only — no pass/fail gate
            "informational": True,
        }

    # Seed distribution compliance per round
    seed_compliance: dict[str, dict] = {}
    for struct_key, bracket_source in [
        ("round_of_32",  "round_of_64"),
        ("sweet_16",     "round_of_32"),
        ("elite_8",      "sweet_16"),
        ("final_four",   "elite_8"),
        ("championship", "final_four"),
    ]:
        target_seeds  = structure.get(struct_key, [])
        games         = bracket.get(bracket_source, [])
        actual_seeds  = sorted(_seeds_in_round(games))
        target_sorted = sorted(target_seeds if isinstance(target_seeds, list)
                               else [target_seeds])
        seed_compliance[struct_key] = {
            "target": target_sorted,
            "actual": actual_seeds,
            "ok":     actual_seeds == target_sorted,
        }

    # Double-digit seed check — mode-aware target range
    s16_teams = [g["winner"]["seed"] for g in bracket["round_of_32"]]
    dd_in_s16 = sum(1 for s in s16_teams if s >= 10)
    t_min, t_max = _S16_DD_TARGETS.get(mode, (0, 16))
    dd_ok        = t_min <= dd_in_s16 <= t_max
    dd_target_str = str(t_min) if t_min == t_max else f"{t_min}–{t_max}"

    return {
        "mode":                    mode,
        "double_digit_in_sweet16": {
            "actual":     dd_in_s16,
            "target":     dd_target_str,
            "target_min": t_min,
            "target_max": t_max,
            "ok":         dd_ok,
        },
        "upset_compliance":  upset_compliance,
        "seed_compliance":   seed_compliance,
        "fully_compliant":   dd_ok,   # upset counts are informational only
    }
