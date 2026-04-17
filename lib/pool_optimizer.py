"""
lib/pool_optimizer.py
Pool-Size Optimizer — post-processing layer over simulate_bracket().

Does NOT re-run bracket simulation.  Takes an existing bracket dict and
re-scores the champion and Final Four picks using pool-size-adjusted weights
that shift between probability-maximizing (small pools) and
leverage-maximizing (large pools).

Formulas
--------
  quality  = 0.70 × win_prob + 0.30 × champion_profile_score
  leverage = win_prob − public_pick_pct   (value_score)
  pool_score = q_weight × quality + (1 − q_weight) × leverage

Pool size thresholds and quality weights (q_weight):
  small  (≤ 20) : 0.85  — win probability dominates, few opponents to beat
  medium (21–75): 0.60  — balanced
  large  (> 75) : 0.30  — leverage dominates, need differentiated picks
"""

from __future__ import annotations

# Optional import — falls back to seed-only scoring when absent.
try:
    from lib.team_ratings import predict_win_probability as _predict_win_probability
    _HAS_RATINGS = True
except ImportError:
    _HAS_RATINGS = False

# Seed-based public pick percentages (mirrors _SEED_PUBLIC_PICK_LATE in team_selector).
_SEED_PUBLIC_PICK: dict[int, float] = {
    1: 0.370, 2: 0.190, 3: 0.110, 4: 0.075, 5: 0.045,
    6: 0.030, 7: 0.020, 8: 0.015, 9: 0.015,
    10: 0.010, 11: 0.010, 12: 0.005, 13: 0.002,
    14: 0.001, 15: 0.000, 16: 0.000,
}


# ── Strategy tiers ──────────────────────────────────────────────────────────

def _strategy(pool_size: int) -> dict:
    """Return strategy name, quality weight, and description for a pool size."""
    if pool_size <= 20:
        return {
            "name":        "small",
            "q_weight":    0.85,
            "description": (
                f"Small pool ({pool_size} entries): prioritize win probability. "
                "With few opponents you mainly need correct outcomes, not "
                "differentiated picks. Champion and Final Four choices lean "
                "heavily toward the most likely winners."
            ),
        }
    elif pool_size <= 75:
        return {
            "name":        "medium",
            "q_weight":    0.60,
            "description": (
                f"Medium pool ({pool_size} entries): balanced strategy. "
                "Mix of probability and leverage — take the chalk pick when it "
                "is clearly better, but lean toward the value pick when the "
                "win-probability gap is small."
            ),
        }
    else:
        return {
            "name":        "large",
            "q_weight":    0.30,
            "description": (
                f"Large pool ({pool_size} entries): maximize leverage. "
                "Most entries will pick the heavy favorites. To win you need "
                "differentiated picks. Champion and Final Four choices lean "
                "toward teams whose win probability significantly exceeds "
                "public ownership."
            ),
        }


# ── Per-team scoring ────────────────────────────────────────────────────────

def _public_pick_pct(team: dict, opponent: dict) -> float:
    """Seed-based public pick estimate normalized to the specific matchup."""
    if "public_pick_pct" in team and team["public_pick_pct"] is not None:
        return float(team["public_pick_pct"])
    w_t = _SEED_PUBLIC_PICK.get(team["seed"], 0.001)
    w_o = _SEED_PUBLIC_PICK.get(opponent["seed"], 0.001)
    total = w_t + w_o
    return w_t / total if total > 0 else 0.5


def _team_pool_score(
    team:      dict,
    opponent:  dict,
    q_weight:  float,
) -> tuple[float, float, float, float]:
    """
    Compute pool-adjusted score for `team` vs `opponent`.

    Returns (pool_score, quality, leverage, win_prob).
    """
    has_data = _HAS_RATINGS and "team_rating" in team and "team_rating" in opponent

    if has_data:
        wp  = float(_predict_win_probability(team, opponent)["team_a"])
        cps = float(team.get("champion_profile_score", team.get("profile_score", 0.5)))
        pub = _public_pick_pct(team, opponent)
    else:
        # Seed-based fallback
        seed_t = team["seed"]
        seed_o = opponent["seed"]
        # Rough win prob from seed advantage
        wp  = max(0.05, min(0.95, (seed_o - seed_t + 16) / 32.0))
        cps = max(0.0, (16 - seed_t) / 15.0)
        pub = _public_pick_pct(team, opponent)

    quality  = 0.70 * wp + 0.30 * cps
    leverage = wp - pub
    score    = q_weight * quality + (1.0 - q_weight) * leverage

    return round(score, 4), round(quality, 4), round(leverage, 4), round(wp, 4)


# ── Final Four team profiles ─────────────────────────────────────────────────

def _ff_team_profile(team: dict, q_weight: float) -> dict:
    """
    Build a pool-score profile for a Final Four team.

    Uses a seed-1 equivalent as the implicit "opponent" so quality and
    leverage are measured against the baseline best competitor.
    """
    # Build a synthetic top-1-seed opponent for the leverage calculation
    seed1_proxy: dict = {"seed": 1, "rating": 96.0}
    score, quality, leverage, wp = _team_pool_score(team, seed1_proxy, q_weight)

    # Simpler public_pick for standalone FF teams (based on seed only)
    pub = _SEED_PUBLIC_PICK.get(team["seed"], 0.001)

    # Ownership label
    if pub >= 0.25:
        ownership = "heavy chalk"
    elif pub >= 0.10:
        ownership = "moderate chalk"
    elif pub >= 0.04:
        ownership = "moderate value"
    else:
        ownership = "high leverage"

    return {
        "name":      team["name"],
        "seed":      team["seed"],
        "pool_score": score,
        "quality":   quality,
        "leverage":  leverage,
        "win_prob_vs_1": wp,
        "est_public_pct": round(pub, 3),
        "ownership_label": ownership,
    }


# ── Main entry point ─────────────────────────────────────────────────────────

def optimize_for_pool(bracket: dict, pool_size: int) -> dict:
    """
    Apply pool-size strategy to an existing simulated bracket.

    Parameters
    ----------
    bracket   : dict — output from simulate_bracket()
    pool_size : int  — number of entries in your contest pool

    Returns
    -------
    {
        "pool_size":             int,
        "strategy":              str,   "small" | "medium" | "large"
        "strategy_description":  str,
        "champion":              dict,  team dict
        "champion_rationale":    str,
        "champion_changed":      bool,  True if optimizer flipped the model pick
        "final_four":            list[dict],  4 teams, sorted by pool_score desc
        "final_four_rationale":  str,
    }
    """
    strat    = _strategy(pool_size)
    q_weight = strat["q_weight"]

    # ── Championship re-scoring ───────────────────────────────────────────
    champ_game = bracket.get("championship", {})
    model_champ = bracket.get("champion", {})
    runner      = champ_game.get("loser", {})

    if model_champ and runner:
        score_a, qual_a, lev_a, wp_a = _team_pool_score(model_champ, runner, q_weight)
        score_b, qual_b, lev_b, wp_b = _team_pool_score(runner, model_champ, q_weight)

        if score_a >= score_b:
            opt_champ   = model_champ
            opt_score   = score_a
            opt_qual    = qual_a
            opt_lev     = lev_a
            opt_wp      = wp_a
            changed     = False
            alt         = runner
            alt_score   = score_b
        else:
            opt_champ   = runner
            opt_score   = score_b
            opt_qual    = qual_b
            opt_lev     = lev_b
            opt_wp      = wp_b
            changed     = True
            alt         = model_champ
            alt_score   = score_a

        # Build rationale
        gap = round(abs(opt_score - alt_score), 4)
        if changed:
            flip_reason = (
                f"Model picked {alt['name']} (seed {alt['seed']}) "
                f"but pool-adjusted score favors {opt_champ['name']} "
                f"(seed {opt_champ['seed']}) by {gap:.4f}."
            )
        else:
            flip_reason = (
                f"Model pick {opt_champ['name']} (seed {opt_champ['seed']}) "
                f"confirmed — pool-adjusted margin {gap:.4f} over "
                f"{alt['name']} (seed {alt['seed']})."
            )

        champ_rationale = (
            f"{opt_champ['name']} (seed {opt_champ['seed']}) | "
            f"pool_score={opt_score:.4f}  quality={opt_qual:.4f}  "
            f"leverage={opt_lev:.4f}  win_prob={opt_wp:.3f}. "
            f"{flip_reason}"
        )
    else:
        opt_champ       = model_champ
        changed         = False
        champ_rationale = "Championship data unavailable — using model champion."

    # ── Final Four re-ranking ────────────────────────────────────────────
    # The 4 FF teams are the E8 winners — already in bracket["final_four"]
    # as the two game winners, plus we reconstruct from the champion/runner path.
    ff_teams_raw: list[dict] = []
    for game in bracket.get("final_four", []):
        ff_teams_raw.append(game.get("winner", {}))
        ff_teams_raw.append(game.get("loser", {}))
    # Deduplicate by name while preserving order
    seen: set[str] = set()
    ff_teams: list[dict] = []
    for t in ff_teams_raw:
        if t and t.get("name") and t["name"] not in seen:
            seen.add(t["name"])
            ff_teams.append(t)

    ff_profiles = [_ff_team_profile(t, q_weight) for t in ff_teams]
    ff_profiles.sort(key=lambda p: p["pool_score"], reverse=True)

    # Build FF rationale
    chalk    = [p for p in ff_profiles if "chalk"  in p["ownership_label"]]
    leverage = [p for p in ff_profiles if "value"  in p["ownership_label"]
                                       or "leverage" in p["ownership_label"]]

    ff_rationale_parts = []
    if strat["name"] == "large" and leverage:
        ff_rationale_parts.append(
            f"Large-pool strategy: lead with high-leverage picks — "
            + ", ".join(f"{p['name']} (s{p['seed']}, {p['ownership_label']})"
                        for p in leverage)
            + "."
        )
        if chalk:
            ff_rationale_parts.append(
                "Chalk picks still appear but ranked lower: "
                + ", ".join(f"{p['name']} (s{p['seed']})" for p in chalk)
                + "."
            )
    elif strat["name"] == "small" and chalk:
        ff_rationale_parts.append(
            f"Small-pool strategy: prioritize chalk — "
            + ", ".join(f"{p['name']} (s{p['seed']}, {p['ownership_label']})"
                        for p in chalk)
            + "."
        )
    else:
        ff_rationale_parts.append(
            "Balanced Final Four: "
            + ", ".join(
                f"{p['name']} (s{p['seed']}, score={p['pool_score']:.4f})"
                for p in ff_profiles
            )
            + "."
        )

    ff_rationale = "  ".join(ff_rationale_parts)

    return {
        "pool_size":            pool_size,
        "strategy":             strat["name"],
        "strategy_description": strat["description"],
        "champion":             opt_champ,
        "champion_rationale":   champ_rationale,
        "champion_changed":     changed,
        "final_four":           ff_profiles,
        "final_four_rationale": ff_rationale,
    }
