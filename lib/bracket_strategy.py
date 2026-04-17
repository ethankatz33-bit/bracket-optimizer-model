"""
lib/bracket_strategy.py
Champion-value bracket portfolio generator.

Shifts the objective from raw accuracy to expected pool winnings.
Works entirely from simulate_bracket() outputs — does NOT touch the model.

Key concepts
------------
value_score   : win_prob / public_pct
                How much "bang per pick-share" — positive when your win_prob
                exceeds your share of public picks on that team.

composite     : pool-size blended score
                Small pool → weight win_prob (safer to pick likely winner).
                Large pool → weight value_score (differentiation matters more).

champion-first: fix champion, build their E8→FF→Championship path using base
                model picks for opponents, fill all other games from base.

portfolio     : N diverse brackets, each with a different champion pick,
                ranked by composite score for the given pool size.

Public API
----------
  extract_candidates(bracket, public_picks)  → list[ChampionCandidate]
  generate_portfolio(bracket, n, pool_size, public_picks) → list[BracketEntry]
  format_portfolio(entries, pool_size)  → str
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

# ── Optional: import win-probability function from team_ratings ───────────────
try:
    from lib.team_ratings import predict_win_probability as _predict_wp
    _HAS_WP = True
except ImportError:
    _HAS_WP = False


# ── Default public pick % (per individual team, not per seed group) ───────────
# Approximates ESPN bracket challenge champion pick distributions.
# Four 1-seeds collectively draw ~55%; each averages ~14%.
# Override with real data via public_picks argument.
_DEFAULT_PUBLIC_PCT: dict[int, float] = {
    1:  0.140,
    2:  0.055,
    3:  0.025,
    4:  0.012,
    5:  0.006,
    6:  0.004,
    7:  0.003,
    8:  0.002,
    9:  0.002,
    10: 0.001,
    11: 0.001,
    12: 0.0008,
    13: 0.0005,
    14: 0.0003,
    15: 0.0001,
    16: 0.00005,
}

# Historical per-team P(win championship) by seed — long-run average.
# Based on ~35 years: ~60% 1-seeds, ~22% 2-seeds, ~12% 3/4-seeds.
# Adjusted so four 1-seeds sum to ~0.60 total.
_HIST_WIN_PROB: dict[int, float] = {
    1:  0.150,
    2:  0.055,
    3:  0.030,
    4:  0.018,
    5:  0.010,
    6:  0.008,
    7:  0.006,
    8:  0.005,
    9:  0.004,
    10: 0.003,
    11: 0.003,
    12: 0.002,
    13: 0.001,
    14: 0.001,
    15: 0.0005,
    16: 0.0001,
}

# Pool size thresholds (must match lib/pool_strategy.py tier boundaries)
_SMALL_POOL  = 25    # ≤ 25  → heavily favor win probability
_MEDIUM_POOL = 100   # 26–100 → blend
# > 100               → heavily favor value_score / differentiation

# Final Four region pairings (mirrors FF_REGION_PAIRS in team_selector.py).
# (East, West) share one FF game; (South, Midwest) share the other.
_FF_PAIR: dict[str, str] = {
    "East": "West", "West": "East",
    "South": "Midwest", "Midwest": "South",
}

# Which FF game index each region belongs to (0 = East/West, 1 = South/Midwest)
_FF_GAME_IDX: dict[str, int] = {
    "East": 0, "West": 0, "South": 1, "Midwest": 1,
}

# Standard ESPN-style point values by round (used in EV explanation)
_ROUND_POINTS: dict[str, int] = {
    "Round of 64":  1,
    "Round of 32":  2,
    "Sweet 16":     4,
    "Elite 8":      8,
    "Final Four":  16,
    "Championship":32,
}


# ════════════════════════════════════════════════════════════════════════════
# Data classes
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ChampionCandidate:
    """A team being considered as champion pick in a portfolio bracket."""
    name:        str
    seed:        int
    region:      str
    team_rating: float
    cps:         float        # champion_profile_score (0–1)
    win_prob:    float        # estimated P(team wins tournament) — MC title_prob when available
    public_pct:  float        # estimated fraction of public picking this team
    value_score: float        # win_prob / public_pct (>1 = positive value)
    mc_ff_prob:  float = 0.0  # MC Final Four probability (0 when MC not run)
    composite:   float = 0.0  # pool-size blended ranking score (set by portfolio fn)
    in_base_ff:  bool  = False
    in_base_e8:  bool  = False
    team_dict:   dict  = field(default_factory=dict, repr=False)


@dataclass
class BracketEntry:
    """One bracket in the portfolio."""
    index:     int
    champion:  ChampionCandidate
    bracket:   dict
    rationale: str
    ev_note:   str


# ════════════════════════════════════════════════════════════════════════════
# Win-probability helpers
# ════════════════════════════════════════════════════════════════════════════

def _wp(team_a: dict, team_b: dict) -> float:
    """P(team_a beats team_b), using model if available else 0.5."""
    if _HAS_WP and "team_rating" in team_a and "team_rating" in team_b:
        return float(_predict_wp(team_a, team_b)["team_a"])
    # Seed-based fallback: lower seed is the favourite
    s_a, s_b = team_a.get("seed", 8), team_b.get("seed", 8)
    if s_a == s_b:
        return 0.5
    return 0.62 if s_a < s_b else 0.38


def _path_win_prob(candidate: ChampionCandidate, base_bracket: dict) -> float:
    """
    Estimate P(candidate wins championship) as the product:
        P(win E8) × P(win FF) × P(win Championship)

    Opponents at each stage are taken from the base bracket:
    - E8 opp  = whoever is in their regional E8 game
    - FF opp  = E8 winner of the paired region
    - Champ opp = expected opponent from the other FF half
                  (weighted by their P(reach championship))
    """
    team   = candidate.team_dict
    region = candidate.region

    # Fallback: no team_rating data
    if not team or "team_rating" not in team:
        return _HIST_WIN_PROB.get(candidate.seed, 0.005)

    # ── E8 ────────────────────────────────────────────────────────────────
    e8_game = next(
        (g for g in base_bracket.get("elite_8", []) if g.get("region") == region),
        None,
    )
    if e8_game is None:
        return _HIST_WIN_PROB.get(candidate.seed, 0.005)

    if e8_game["winner"]["name"] == candidate.name:
        e8_opp = e8_game["loser"]
    elif e8_game["loser"]["name"] == candidate.name:
        e8_opp = e8_game["winner"]
    else:
        e8_opp = e8_game["winner"]   # not in E8; pretend they face current winner

    p_e8 = _wp(team, e8_opp)

    # ── Final Four ────────────────────────────────────────────────────────
    paired_region = _FF_PAIR.get(region)
    ff_e8 = next(
        (g for g in base_bracket.get("elite_8", []) if g.get("region") == paired_region),
        None,
    )
    ff_opp = ff_e8["winner"] if ff_e8 else None
    p_ff   = _wp(team, ff_opp) if ff_opp else 0.5

    # ── Championship ──────────────────────────────────────────────────────
    # Other half = the two regions NOT in candidate's FF game
    ff_idx   = _FF_GAME_IDX.get(region, 0)
    regions  = ["East", "West", "South", "Midwest"]
    # Indices of regions in the OTHER FF game
    other_indices = [0, 1] if ff_idx == 1 else [2, 3]
    other_regions = [regions[i] for i in other_indices]

    other_e8_winners = [
        next((g["winner"] for g in base_bracket.get("elite_8", [])
              if g.get("region") == r), None)
        for r in other_regions
    ]
    other_e8_winners = [t for t in other_e8_winners if t is not None]

    if len(other_e8_winners) == 2:
        # P(opponent A makes championship) × P(we beat A)
        # + P(opponent B makes championship) × P(we beat B)
        p_a_champ = _wp(other_e8_winners[0], other_e8_winners[1])
        p_win_vs_a = _wp(team, other_e8_winners[0])
        p_win_vs_b = _wp(team, other_e8_winners[1])
        p_champ    = p_a_champ * p_win_vs_a + (1 - p_a_champ) * p_win_vs_b
    elif len(other_e8_winners) == 1:
        p_champ = _wp(team, other_e8_winners[0])
    else:
        p_champ = 0.5

    return round(p_e8 * p_ff * p_champ, 5)


# ════════════════════════════════════════════════════════════════════════════
# Candidate extraction
# ════════════════════════════════════════════════════════════════════════════

def _e8_participants(base_bracket: dict) -> list[dict]:
    """Return the 8 teams that played in the Elite 8 (4 winners + 4 losers)."""
    teams = []
    for game in base_bracket.get("elite_8", []):
        teams.append(game["winner"])
        teams.append(game["loser"])
    return teams


def _region_of(team_name: str, base_bracket: dict) -> str | None:
    """Find which region a team played their E8 game in."""
    for game in base_bracket.get("elite_8", []):
        if game["winner"]["name"] == team_name or game["loser"]["name"] == team_name:
            return game.get("region")
    return None


def _find_team_in_bracket(bracket: dict, name: str) -> dict | None:
    """Search all bracket rounds for a team dict by name."""
    for rnd_key in ("round_of_64", "round_of_32", "sweet_16", "elite_8", "final_four"):
        games = bracket.get(rnd_key, [])
        if isinstance(games, dict):
            games = [games]
        for game in (games or []):
            for side in ("winner", "loser"):
                t = game.get(side, {})
                if t.get("name") == name:
                    return t
    cg = bracket.get("championship")
    if cg:
        for side in ("winner", "loser"):
            t = cg.get(side, {})
            if t.get("name") == name:
                return t
    return None


def extract_candidates(
    base_bracket:  dict,
    public_picks:  dict[str, float] | None = None,
    mc_results=None,
) -> list[ChampionCandidate]:
    """
    Build the pool of champion candidates.

    Base pool: E8 participants (8 teams).
    If mc_results provided: also include any team with MC title_prob ≥ threshold
    that was eliminated before the E8.  Win probabilities use MC title_prob when
    available, falling back to the path-based estimate.

    Parameters
    ----------
    base_bracket  : output of simulate_bracket()
    public_picks  : optional {team_name: fraction} override for public pick %.
    mc_results    : optional MCResults from run_monte_carlo().

    Returns
    -------
    List of ChampionCandidate sorted by win_prob descending.
    """
    ff_names = {
        t["name"]
        for game in base_bracket.get("final_four", [])
        for t in (game["winner"], game["loser"])
    }
    e8_winner_names = {g["winner"]["name"] for g in base_bracket.get("elite_8", [])}

    # ── Collect team dicts: E8 first, then MC-only additions ─────────────
    team_pool: list[dict] = []
    seen: set[str] = set()

    for team in _e8_participants(base_bracket):
        if team["name"] not in seen:
            seen.add(team["name"])
            team_pool.append(team)

    if mc_results is not None:
        try:
            from lib.monte_carlo import MC_CANDIDATE_MIN_TITLE_PROB
        except ImportError:
            MC_CANDIDATE_MIN_TITLE_PROB = 0.025
        for mc_r in mc_results.candidates(MC_CANDIDATE_MIN_TITLE_PROB):
            if mc_r.name not in seen:
                team = _find_team_in_bracket(base_bracket, mc_r.name)
                if team is not None:
                    seen.add(mc_r.name)
                    team_pool.append(team)

    # ── Build ChampionCandidate for each team ─────────────────────────────
    candidates: list[ChampionCandidate] = []

    for team in team_pool:
        name        = team["name"]
        seed        = team.get("seed", 16)
        region      = _region_of(name, base_bracket) or team.get("region", "Unknown")
        team_rating = float(team.get("team_rating", 0.0))
        cps         = float(team.get("champion_profile_score",
                                     team.get("profile_score", 0.5)))

        pub_pct = (
            public_picks.get(name)
            if public_picks and name in public_picks
            else _DEFAULT_PUBLIC_PCT.get(seed, 0.001)
        )

        # Win probability: prefer MC title_prob over path-based estimate
        if mc_results is not None:
            mc_tp = mc_results.title_prob(name)
            win_prob = mc_tp if mc_tp > 0 else _path_win_prob(
                ChampionCandidate(
                    name=name, seed=seed, region=region,
                    team_rating=team_rating, cps=cps,
                    win_prob=0.0, public_pct=pub_pct, value_score=0.0,
                    team_dict=team,
                ),
                base_bracket,
            )
        else:
            win_prob = _path_win_prob(
                ChampionCandidate(
                    name=name, seed=seed, region=region,
                    team_rating=team_rating, cps=cps,
                    win_prob=0.0, public_pct=pub_pct, value_score=0.0,
                    team_dict=team,
                ),
                base_bracket,
            )

        value_score = round(win_prob / max(pub_pct, 0.0001), 3)
        mc_ff_prob  = round(mc_results.ff_prob(name), 4) if mc_results is not None else 0.0

        candidates.append(ChampionCandidate(
            name=name,
            seed=seed,
            region=region,
            team_rating=team_rating,
            cps=cps,
            win_prob=round(win_prob, 4),
            public_pct=round(pub_pct, 4),
            value_score=value_score,
            mc_ff_prob=mc_ff_prob,
            in_base_ff=name in ff_names,
            in_base_e8=name in e8_winner_names,
            team_dict=team,
        ))

    return sorted(candidates, key=lambda c: c.win_prob, reverse=True)


# ════════════════════════════════════════════════════════════════════════════
# Composite score (pool-size-aware ranking)
# ════════════════════════════════════════════════════════════════════════════

def _composite_score(candidate: ChampionCandidate, pool_size: int) -> float:
    """
    Blend win_prob, value_score, and (when available) MC Final Four probability.

    When mc_ff_prob > 0 the formula uses three components:
      Small pool  (≤ 25)   : 72% win_prob + 18% value_score + 10% ff_prob
      Medium pool (26–100) : 45% win_prob + 45% value_score + 10% ff_prob
      Large pool  (> 100)  : 18% win_prob + 72% value_score + 10% ff_prob

    Without mc_ff_prob (MC not run) the original two-component formula is used:
      Small  : 80% / 20%    Medium : 50% / 50%    Large  : 20% / 80%

    All components are normalised to [0,1] before blending.
    win_prob    → already in [0,1]
    value_score → normalised by dividing by a ceiling of 10.0
    mc_ff_prob  → already in [0,1]
    """
    norm_wp = candidate.win_prob
    norm_vs = min(candidate.value_score / 10.0, 1.0)

    if candidate.mc_ff_prob > 0:
        norm_ff = candidate.mc_ff_prob
        if pool_size <= _SMALL_POOL:
            w_wp, w_vs, w_ff = 0.72, 0.18, 0.10
        elif pool_size <= _MEDIUM_POOL:
            w_wp, w_vs, w_ff = 0.45, 0.45, 0.10
        else:
            w_wp, w_vs, w_ff = 0.18, 0.72, 0.10
        return round(w_wp * norm_wp + w_vs * norm_vs + w_ff * norm_ff, 5)

    if pool_size <= _SMALL_POOL:
        w_wp, w_vs = 0.80, 0.20
    elif pool_size <= _MEDIUM_POOL:
        w_wp, w_vs = 0.50, 0.50
    else:
        w_wp, w_vs = 0.20, 0.80

    return round(w_wp * norm_wp + w_vs * norm_vs, 5)


# ════════════════════════════════════════════════════════════════════════════
# Champion-first bracket construction
# ════════════════════════════════════════════════════════════════════════════

def _make_game(winner: dict, loser: dict, round_name: str, region: str) -> dict:
    """Construct a game result dict that matches simulate_bracket() output structure."""
    return {
        "winner":       winner,
        "loser":        loser,
        "round":        round_name,
        "region":       region,
        "is_upset":     winner.get("seed", 0) > loser.get("seed", 0),
        "upset_quality": None,
    }


def build_champion_first_bracket(
    base_bracket: dict,
    candidate:    ChampionCandidate,
) -> dict:
    """
    Construct a bracket where candidate wins the championship.

    Changes from base bracket
    -------------------------
    1. E8 (candidate's region): candidate wins (flipped if necessary)
    2. FF (candidate's game):   candidate beats E8 winner of paired region
    3. Championship:            candidate beats winner of other FF game
    4. All R64 / R32 / S16 / other-region E8 games: unchanged

    If candidate is not in the base E8, they are substituted in as the winner
    of their regional E8 (displacing whoever was there).
    """
    bracket = copy.deepcopy(base_bracket)
    team    = candidate.team_dict
    region  = candidate.region

    # ── E8: force candidate to win ────────────────────────────────────────
    e8_game = next(
        (g for g in bracket["elite_8"] if g.get("region") == region), None
    )
    if e8_game is None:
        return bracket  # safety guard

    if e8_game["winner"]["name"] == candidate.name:
        # Already wins; identify opponent for path labelling
        pass
    elif e8_game["loser"]["name"] == candidate.name:
        # Was predicted to lose — flip
        e8_game["winner"], e8_game["loser"] = team, e8_game["winner"]
        e8_game["is_upset"] = team.get("seed", 0) > e8_game["loser"].get("seed", 0)
    else:
        # Not in E8 (eliminated earlier) — substitute in as winner
        original_winner         = e8_game["winner"]
        e8_game["winner"]       = team
        e8_game["loser"]        = original_winner
        e8_game["is_upset"]     = team.get("seed", 0) > original_winner.get("seed", 0)

    # ── FF: candidate vs E8 winner of paired region ───────────────────────
    paired_region = _FF_PAIR.get(region, "")
    ff_e8_game    = next(
        (g for g in bracket["elite_8"] if g.get("region") == paired_region), None
    )
    ff_opp  = ff_e8_game["winner"] if ff_e8_game else None
    ff_idx  = _FF_GAME_IDX.get(region, 0)

    if ff_opp and ff_idx < len(bracket["final_four"]):
        bracket["final_four"][ff_idx] = _make_game(team, ff_opp, "Final Four", "National")

    # ── Championship: candidate vs winner of the other FF game ────────────
    other_ff_idx = 1 - ff_idx
    if other_ff_idx < len(bracket["final_four"]):
        other_ff_winner = bracket["final_four"][other_ff_idx]["winner"]
        bracket["championship"] = _make_game(
            team, other_ff_winner, "Championship", "National"
        )
        bracket["champion"] = team

    return bracket


# ════════════════════════════════════════════════════════════════════════════
# Portfolio generator
# ════════════════════════════════════════════════════════════════════════════

def generate_portfolio(
    base_bracket: dict,
    n:            int  = 5,
    pool_size:    int  = 100,
    public_picks: dict[str, float] | None = None,
    mc_results=None,
) -> list[BracketEntry]:
    """
    Generate a portfolio of N diverse champion-first brackets.

    Parameters
    ----------
    base_bracket  : output of simulate_bracket()
    n             : number of brackets in portfolio (capped at # of E8 teams = 8)
    pool_size     : estimated number of entries in the pool
    public_picks  : optional {team_name: float} override for champion pick %.
    mc_results    : optional MCResults — expands candidate pool and uses MC
                    title probabilities for win_prob when available.

    Returns
    -------
    List of BracketEntry sorted by composite score, best first.
    Brackets are guaranteed to have distinct champions.
    """
    candidates = extract_candidates(base_bracket, public_picks, mc_results)

    # Attach composite scores and re-sort
    for c in candidates:
        c.composite = _composite_score(c, pool_size)
    candidates.sort(key=lambda c: c.composite, reverse=True)

    # Take top N (at most len(candidates))
    selected = candidates[:min(n, len(candidates))]

    # Determine pool strategy label
    if pool_size <= _SMALL_POOL:
        strategy = "small-pool (win-probability weighted)"
    elif pool_size <= _MEDIUM_POOL:
        strategy = "medium-pool (balanced)"
    else:
        strategy = "large-pool (leverage / value weighted)"

    entries: list[BracketEntry] = []
    for i, cand in enumerate(selected, 1):
        bracket   = build_champion_first_bracket(base_bracket, cand)
        rationale = _build_rationale(cand, pool_size, strategy)
        ev_note   = _build_ev_note(cand, pool_size)

        entries.append(BracketEntry(
            index=i,
            champion=cand,
            bracket=bracket,
            rationale=rationale,
            ev_note=ev_note,
        ))

    return entries


# ════════════════════════════════════════════════════════════════════════════
# Rationale and EV explanation builders
# ════════════════════════════════════════════════════════════════════════════

def _build_rationale(c: ChampionCandidate, pool_size: int, strategy: str) -> str:
    ff_src = f"  MC FF prob: {c.mc_ff_prob:.1%}" if c.mc_ff_prob > 0 else ""
    lines = [
        f"{c.name} — seed {c.seed}, {c.region}",
        f"Strategy:  {strategy}",
        f"Title prob:{c.win_prob:>7.1%}   "
        f"Public pick: {c.public_pct:.1%}   "
        f"Value score: {c.value_score:.2f}x   "
        f"Composite: {c.composite:.4f}" + ff_src,
    ]
    if c.value_score >= 2.0:
        lines.append("Leverage:  HIGH — win probability substantially exceeds pick share")
    elif c.value_score >= 1.2:
        lines.append("Leverage:  MODERATE — win probability slightly exceeds pick share")
    elif c.value_score >= 0.8:
        lines.append("Leverage:  NEUTRAL — priced fairly by public")
    else:
        lines.append("Leverage:  LOW — public is over-weighting this team")
    if c.in_base_ff:
        lines.append("Path:      model predicts this team reaches the Final Four")
    elif c.in_base_e8:
        lines.append("Path:      model predicts this team wins their region (E8 winner)")
    else:
        lines.append("Path:      model does NOT predict this team to reach the Final Four"
                     " — speculative pick")
    return "\n".join(lines)


def _build_ev_note(c: ChampionCandidate, pool_size: int) -> str:
    champ_pts = _ROUND_POINTS["Championship"]
    ev_relative = c.win_prob / max(c.public_pct, 0.0001)

    if pool_size > 1:
        # Simplified EV: E[points from champion pick] ∝ P(correct) / P(others also correct)
        # Here we use value_score as the relative EV multiplier
        ev_note = (
            f"Champion pick contributes {champ_pts} pts if correct.\n"
            f"Expected relative return vs consensus champion: {ev_relative:.2f}x\n"
        )
        if ev_relative > 1.5:
            ev_note += (
                f"In a {pool_size}-person pool, picking {c.name} gives you "
                f"~{ev_relative:.1f}x the expected return per point of pick-share "
                f"vs the consensus favourite."
            )
        elif ev_relative < 0.7:
            ev_note += (
                f"Warning: {c.name} is over-picked relative to their win probability. "
                f"If they win, you'll share the prize with many other entries."
            )
        else:
            ev_note += (
                f"{c.name} is fairly priced — "
                f"win probability closely matches their public pick share."
            )
    else:
        ev_note = f"Single-entry pool — maximize win probability. {champ_pts} pts for correct champion."

    return ev_note


# ════════════════════════════════════════════════════════════════════════════
# Formatted output
# ════════════════════════════════════════════════════════════════════════════

def format_portfolio(entries: list[BracketEntry], pool_size: int) -> str:
    """Return a human-readable string summarising the full portfolio."""
    W     = 72
    lines = []
    lines.append("=" * W)
    lines.append(f"  BRACKET PORTFOLIO  ({len(entries)} brackets, pool size ≈ {pool_size})")
    lines.append("=" * W)

    # Summary table — add MC FF% column when MC data is available
    has_mc = any(e.champion.mc_ff_prob > 0 for e in entries)
    if has_mc:
        lines.append(f"\n  {'#':>2}  {'Champion':<22} {'s':>2}  {'Region':<8}  "
                     f"{'Title%':>6}  {'FF%':>5}  {'Pick%':>6}  {'Value':>6}  {'Comp':>6}  Path")
        lines.append("  " + "─" * (W - 2))
        for e in entries:
            c = e.champion
            path = "FF" if c.in_base_ff else ("E8" if c.in_base_e8 else "spec")
            lines.append(
                f"  {e.index:>2}  {c.name:<22} {c.seed:>2}  {c.region:<8}  "
                f"{c.win_prob:>6.1%}  {c.mc_ff_prob:>5.1%}  {c.public_pct:>6.2%}  "
                f"{c.value_score:>6.2f}x  {c.composite:>6.4f}  {path}"
            )
    else:
        lines.append(f"\n  {'#':>2}  {'Champion':<22} {'s':>2}  {'Region':<8}  "
                     f"{'Win%':>5}  {'Pick%':>6}  {'Value':>6}  {'Comp':>6}  Path")
        lines.append("  " + "─" * (W - 2))
        for e in entries:
            c = e.champion
            path = "FF" if c.in_base_ff else ("E8" if c.in_base_e8 else "spec")
            lines.append(
                f"  {e.index:>2}  {c.name:<22} {c.seed:>2}  {c.region:<8}  "
                f"{c.win_prob:>5.1%}  {c.public_pct:>6.2%}  {c.value_score:>6.2f}x  "
                f"{c.composite:>6.4f}  {path}"
            )

    # Detail for each bracket
    for e in entries:
        c = e.champion
        lines.append(f"\n{'─' * W}")
        lines.append(f"  BRACKET {e.index}: {c.name} (seed {c.seed}, {c.region})")
        lines.append(f"{'─' * W}")

        # Show the champion's path in this bracket
        bracket = e.bracket
        champ_games = _champion_path_games(bracket, c.name)
        if champ_games:
            lines.append(f"  Champion path:")
            for rnd, opp, opp_seed in champ_games:
                lines.append(f"    {rnd:<16}  beats {opp} (seed {opp_seed})")

        lines.append(f"\n  {e.rationale}")
        lines.append(f"\n  Expected value:")
        lines.append(f"  " + e.ev_note.replace("\n", "\n  "))

    lines.append(f"\n{'=' * W}")
    lines.append("  END OF PORTFOLIO")
    lines.append("=" * W)
    return "\n".join(lines)


def _champion_path_games(bracket: dict, champion_name: str) -> list[tuple[str, str, int]]:
    """
    Return list of (round_name, opponent_name, opponent_seed) for all games
    the champion won, from E8 onwards.
    """
    rounds = [
        ("elite_8",      bracket.get("elite_8", [])),
        ("final_four",   bracket.get("final_four", [])),
        ("championship", [bracket["championship"]] if bracket.get("championship") else []),
    ]
    path = []
    for _, games in rounds:
        if isinstance(games, dict):
            games = [games]
        for game in games:
            if game.get("winner", {}).get("name") == champion_name:
                opp = game["loser"]
                path.append((game["round"], opp["name"], opp.get("seed", "?")))
    return path
