"""
lib/monte_carlo.py
Monte Carlo tournament simulation layer.

Runs the bracket N times using probabilistic game resolution
(instead of deterministic upset-selection logic) to estimate
round-advancement and title probabilities for each team.

Public API
----------
  run_monte_carlo(teams_override, n_sims, seed)  →  MCResults
  format_mc_summary(results, top_n)              →  str
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
PROBS_FILE   = PROJECT_ROOT / "data" / "processed" / "seed_probabilities.json"

# Optional team-ratings win-probability model
try:
    from lib.team_ratings import predict_win_probability as _predict_wp
    _HAS_WP = True
except ImportError:
    _HAS_WP = False

# Bracket layout (mirrors team_selector.py constants exactly)
REGIONS         = ["East", "West", "South", "Midwest"]
R64_MATCHUPS    = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
R32_PAIRS       = [(0,1),(2,3),(4,5),(6,7)]
S16_PAIRS       = [(0,1),(2,3)]
FF_REGION_PAIRS = [(0,1),(2,3)]  # (East,West), (South,Midwest)

# Stage indices stored per-team in one simulation run.
# A team that loses in R64 is NOT recorded (stage never set → 0 count).
#   0 = won R64  (reached Round of 32, then lost)
#   1 = won R32  (reached Sweet 16, then lost)
#   2 = won S16  (reached Elite 8, then lost)
#   3 = won E8   (reached Final Four, then lost)
#   4 = won FF   (reached Championship, then lost)
#   5 = Champion
_STAGE_R32   = 0
_STAGE_S16   = 1
_STAGE_E8    = 2
_STAGE_FF    = 3
_STAGE_CG    = 4
_STAGE_CHAMP = 5

# Minimum title probability to include a team as a MC-based champion candidate
MC_CANDIDATE_MIN_TITLE_PROB: float = 0.025


# ════════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MCTeamResult:
    """Advancement probability summary for one team across N simulations."""
    name:       str
    seed:       int
    region:     str
    sims:       int
    # Counts: how many sims this team reached at least this stage
    r32:        int = 0   # won R64
    s16:        int = 0   # won R32
    e8:         int = 0   # won S16
    ff:         int = 0   # won E8
    champ_game: int = 0   # won FF
    champion:   int = 0   # won Championship

    # ── Probability properties ────────────────────────────────────────────
    @property
    def title_prob(self)      -> float: return self.champion   / self.sims
    @property
    def champ_game_prob(self) -> float: return self.champ_game / self.sims
    @property
    def ff_prob(self)         -> float: return self.ff         / self.sims
    @property
    def e8_prob(self)         -> float: return self.e8         / self.sims
    @property
    def s16_prob(self)        -> float: return self.s16        / self.sims
    @property
    def r32_prob(self)        -> float: return self.r32        / self.sims

    def to_dict(self) -> dict:
        return {
            "name":   self.name,
            "seed":   self.seed,
            "region": self.region,
            "sims":   self.sims,
            "probs": {
                "title":       round(self.title_prob,      4),
                "champ_game":  round(self.champ_game_prob, 4),
                "final_four":  round(self.ff_prob,         4),
                "elite_8":     round(self.e8_prob,         4),
                "sweet_16":    round(self.s16_prob,        4),
                "round_of_32": round(self.r32_prob,        4),
            },
        }


@dataclass
class MCResults:
    """Aggregated Monte Carlo results across all teams."""
    n_sims:  int
    results: list[MCTeamResult]
    _index:  dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {r.name: r for r in self.results}

    def get(self, name: str) -> Optional[MCTeamResult]:
        return self._index.get(name)

    def title_prob(self, name: str) -> float:
        r = self._index.get(name)
        return r.title_prob if r else 0.0

    def ff_prob(self, name: str) -> float:
        r = self._index.get(name)
        return r.ff_prob if r else 0.0

    def top_by_title(self, n: int = 10) -> list[MCTeamResult]:
        return sorted(self.results, key=lambda r: r.title_prob, reverse=True)[:n]

    def top_by_ff(self, n: int = 10) -> list[MCTeamResult]:
        return sorted(self.results, key=lambda r: r.ff_prob, reverse=True)[:n]

    def candidates(self, min_prob: float = MC_CANDIDATE_MIN_TITLE_PROB) -> list[MCTeamResult]:
        """Teams with title_prob ≥ min_prob, sorted descending."""
        return sorted(
            [r for r in self.results if r.title_prob >= min_prob],
            key=lambda r: r.title_prob,
            reverse=True,
        )

    def to_dict(self) -> dict:
        by_title = sorted(self.results, key=lambda r: r.title_prob, reverse=True)
        return {
            "n_sims": self.n_sims,
            "teams":  [r.to_dict() for r in by_title],
        }


# ════════════════════════════════════════════════════════════════════════════
# Game simulation helpers
# ════════════════════════════════════════════════════════════════════════════

def _load_win_rates() -> dict:
    with open(PROBS_FILE) as f:
        return json.load(f)["matchup_win_rates"]


def _game_wp(
    team_a:   dict,
    team_b:   dict,
    win_rates: dict,
    cache:    dict,
) -> float:
    """
    P(team_a beats team_b).  Results are cached by (name_a, name_b).

    Priority:
    1. team_rating-based model  (if both teams have team_rating)
    2. historical seed matchup  (if exact seed pair is in win_rates)
    3. linear seed-gap fallback (always available)
    """
    na, nb = team_a["name"], team_b["name"]
    key = (na, nb)
    if key in cache:
        return cache[key]
    rkey = (nb, na)
    if rkey in cache:
        p = 1.0 - cache[rkey]
        cache[key] = p
        return p

    # 1. Rating-based
    if _HAS_WP and "team_rating" in team_a and "team_rating" in team_b:
        p = float(_predict_wp(team_a, team_b)["team_a"])

    else:
        sa, sb = int(team_a.get("seed", 8)), int(team_b.get("seed", 8))
        if sa == sb:
            p = 0.5
        elif sa < sb:
            # team_a is seed-favourite
            und_rate = win_rates.get(f"{sb}_vs_{sa}", 0.0)
            p = 1.0 - und_rate if und_rate > 0 else max(0.55, 0.5 + (sb - sa) * 0.025)
        else:
            und_rate = win_rates.get(f"{sa}_vs_{sb}", 0.0)
            p = und_rate if und_rate > 0 else max(0.05, 0.5 - (sa - sb) * 0.025)
        p = max(0.05, min(0.95, p))

    cache[key] = p
    return p


# ════════════════════════════════════════════════════════════════════════════
# Single-tournament simulation
# ════════════════════════════════════════════════════════════════════════════

def _simulate_one(
    teams:     dict,
    win_rates: dict,
    rng:       random.Random,
    wp_cache:  dict,
) -> dict[str, int]:
    """
    Simulate one complete 64-team tournament.

    Returns {team_name: stage} only for teams that WON their first game
    (i.e. advanced past R64).  Teams eliminated in R64 are absent from
    the dict — their count stays at 0 for all stages.

    Stage values: 0=R32, 1=S16, 2=E8, 3=FF, 4=CG, 5=Champion.
    """
    result: dict[str, int] = {}

    def play(a: dict, b: dict) -> tuple[dict, dict]:
        p = _game_wp(a, b, win_rates, wp_cache)
        if rng.random() < p:
            return a, b   # (winner, loser)
        return b, a

    # ── Round of 64 ───────────────────────────────────────────────────────
    r32: dict[str, list[dict]] = {}
    for region in REGIONS:
        rw = []
        for lo, hi in R64_MATCHUPS:
            w, _ = play(teams[region][lo], teams[region][hi])
            rw.append(w)
        r32[region] = rw

    # ── Round of 32 ───────────────────────────────────────────────────────
    s16: dict[str, list[dict]] = {}
    for region in REGIONS:
        sw = []
        for i, j in R32_PAIRS:
            w, lo = play(r32[region][i], r32[region][j])
            result[lo["name"]] = _STAGE_R32   # reached R32
            sw.append(w)
        s16[region] = sw

    # ── Sweet 16 ──────────────────────────────────────────────────────────
    e8: dict[str, list[dict]] = {}
    for region in REGIONS:
        ew = []
        for i, j in S16_PAIRS:
            w, lo = play(s16[region][i], s16[region][j])
            result[lo["name"]] = _STAGE_S16   # reached S16
            ew.append(w)
        e8[region] = ew

    # ── Elite 8 ───────────────────────────────────────────────────────────
    ff: list[dict] = []
    for region in REGIONS:
        w, lo = play(e8[region][0], e8[region][1])
        result[lo["name"]] = _STAGE_E8        # reached E8
        ff.append(w)

    # ── Final Four ────────────────────────────────────────────────────────
    cg: list[dict] = []
    for a_idx, b_idx in FF_REGION_PAIRS:
        w, lo = play(ff[a_idx], ff[b_idx])
        result[lo["name"]] = _STAGE_FF        # reached FF
        cg.append(w)

    # ── Championship ──────────────────────────────────────────────────────
    w, lo = play(cg[0], cg[1])
    result[lo["name"]] = _STAGE_CG            # reached Championship
    result[w["name"]]  = _STAGE_CHAMP         # Champion

    return result


# ════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(
    teams_override: dict,
    n_sims: int = 1000,
    rand_seed: int | None = None,
) -> MCResults:
    """
    Run n_sims full bracket simulations.

    Parameters
    ----------
    teams_override : {region: {seed: team_dict}} — the 64-team field.
                     Team dicts should include team_rating for model-based
                     win probabilities; seed-based fallback used otherwise.
    n_sims         : number of simulation runs (default 1000, typical 5000–10000)
    rand_seed      : optional RNG seed for reproducibility

    Returns
    -------
    MCResults with per-team advancement counts and probabilities.
    """
    win_rates = _load_win_rates()
    rng       = random.Random(rand_seed)
    wp_cache: dict = {}

    # Build flat team lookup for initialising counts
    all_teams: list[dict] = [
        t
        for region in REGIONS
        for t in teams_override.get(region, {}).values()
    ]

    # Initialise a counter object per team
    counters: dict[str, MCTeamResult] = {
        t["name"]: MCTeamResult(
            name   = t["name"],
            seed   = int(t.get("seed", 0)),
            region = str(t.get("region", "")),
            sims   = n_sims,
        )
        for t in all_teams
    }

    # ── Run simulations ────────────────────────────────────────────────────
    for _ in range(n_sims):
        result = _simulate_one(teams_override, win_rates, rng, wp_cache)

        for name, stage in result.items():
            c = counters.get(name)
            if c is None:
                continue
            if stage >= _STAGE_R32:   c.r32        += 1
            if stage >= _STAGE_S16:   c.s16        += 1
            if stage >= _STAGE_E8:    c.e8         += 1
            if stage >= _STAGE_FF:    c.ff         += 1
            if stage >= _STAGE_CG:    c.champ_game += 1
            if stage == _STAGE_CHAMP: c.champion   += 1

    return MCResults(n_sims=n_sims, results=list(counters.values()))


# ════════════════════════════════════════════════════════════════════════════
# Display
# ════════════════════════════════════════════════════════════════════════════

def format_mc_summary(results: MCResults, top_n: int = 10) -> str:
    W   = 72
    SEP = "=" * W
    lines: list[str] = []

    lines.append(SEP)
    lines.append(f"  MONTE CARLO RESULTS  ({results.n_sims:,} simulations)".center(W))
    lines.append(SEP)

    # ── Title probability table ───────────────────────────────────────────
    lines.append(f"\n  TOP {top_n} BY TITLE PROBABILITY")
    lines.append(
        f"  {'#':<3} {'Name':<22} {'s':>2}  {'Region':<8}  "
        f"{'Title':>6}  {'CG':>6}  {'FF':>6}  {'E8':>6}  {'S16':>6}"
    )
    lines.append("  " + "─" * 68)
    for rank, r in enumerate(results.top_by_title(top_n), 1):
        lines.append(
            f"  {rank:<3} {r.name:<22} {r.seed:>2}  {r.region:<8}  "
            f"{r.title_prob:>6.1%}  {r.champ_game_prob:>6.1%}  "
            f"{r.ff_prob:>6.1%}  {r.e8_prob:>6.1%}  {r.s16_prob:>6.1%}"
        )

    # ── Final Four probability table ──────────────────────────────────────
    lines.append(f"\n  TOP {top_n} BY FINAL FOUR PROBABILITY")
    lines.append(
        f"  {'#':<3} {'Name':<22} {'s':>2}  {'Region':<8}  "
        f"{'FF%':>6}  {'Title%':>7}"
    )
    lines.append("  " + "─" * 54)
    for rank, r in enumerate(results.top_by_ff(top_n), 1):
        lines.append(
            f"  {rank:<3} {r.name:<22} {r.seed:>2}  {r.region:<8}  "
            f"{r.ff_prob:>6.1%}  {r.title_prob:>7.1%}"
        )

    lines.append("\n" + SEP)
    return "\n".join(lines)
