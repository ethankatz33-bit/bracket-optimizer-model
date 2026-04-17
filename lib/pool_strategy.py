"""
lib/pool_strategy.py
Pool-size-aware champion recommendation engine.

Translates Monte Carlo probabilities and candidate scores into a
concrete, actionable recommendation for a specific pool's size tier.

Four official tiers
-------------------
  small_pool  : 10–25 entrants   → maximize win probability, 1 bracket
  medium_pool : 26–100 entrants  → value-positive pick, 1–2 brackets
  large_pool  : 101–1000 entrants → max leverage, 2–4 brackets
  mega_pool   : 1001+ entrants   → max differentiation, 5–10 brackets

Per-tier logic
--------------
  Small:  primary = safest (highest title prob)
  Medium: primary = first candidate with value_score ≥ 1.3 AND seed ≤ 3
          (ties broken by composite); falls back to safest
  Large:  primary = best value (highest value_score with title_prob ≥ 2.5%)
  Mega:   primary = most contrarian viable (highest value_score with
          public_pct < 5% AND title_prob ≥ 2.5%);
          falls back to best value if no contrarian exists

Public API
----------
  classify_pool(pool_size)               → str  (tier key)
  build_recommendation(candidates, pool_size) → Recommendation
  format_recommendation(rec)             → str
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── Tier boundaries ───────────────────────────────────────────────────────────

SMALL_MIN  = 10
SMALL_MAX  = 25
MEDIUM_MIN = 26
MEDIUM_MAX = 100
LARGE_MIN  = 101
LARGE_MAX  = 1000
# MEGA      = 1001+

_TIER_LABELS: dict[str, str] = {
    "small_pool":  f"Small pool  ({SMALL_MIN}–{SMALL_MAX} entrants)",
    "medium_pool": f"Medium pool ({MEDIUM_MIN}–{MEDIUM_MAX} entrants)",
    "large_pool":  f"Large pool  ({LARGE_MIN}–{LARGE_MAX} entrants)",
    "mega_pool":   "Mega pool   (1001+ entrants)",
}

_TIER_STRATEGY_NOTES: dict[str, str] = {
    "small_pool":  (
        "Small pool: everyone has a real shot at the pot. Maximize your chance "
        "of being right — picking the most likely champion outperforms contrarian "
        "picks when the field is shallow."
    ),
    "medium_pool": (
        "Medium pool: one well-differentiated champion beats chasing upsets. "
        "Pick a team with positive value (win prob > pick share) — ideally a "
        "1–3 seed the public is under-valuing."
    ),
    "large_pool": (
        "Large pool: the consensus champion is nearly worthless EV. Picking the "
        "same team as 150 other entries means you split the pot even if they win. "
        "Own your champion pick with maximum leverage — high win prob, low public share."
    ),
    "mega_pool": (
        "Mega pool: only a diversified portfolio gives you meaningful expected "
        "value. Each bracket must have a different champion. Include at least one "
        "pick the public is heavily ignoring — even a seed-5 team at 1–2% public "
        "pick share can produce a 50× payout."
    ),
}

# Minimum title probability for any pick to be considered "viable"
_MIN_VIABLE_PROB     = 0.025
_CONTRARIAN_PUB_CAP  = 0.05   # public_pct ceiling for contrarian classification
_MEDIUM_VALUE_FLOOR  = 1.3    # value_score threshold for medium-pool primary
_MEDIUM_SEED_CAP     = 3      # seed cap for medium-pool value pick

W = 72


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    """
    Pool-size-aware champion recommendation with primary and alternative picks.
    """
    pool_size:           int
    tier:                str    # e.g. "large_pool"
    tier_label:          str    # e.g. "Large pool  (101–1000 entrants)"
    n_brackets:          int    # recommended portfolio size
    strategy_note:       str    # one-paragraph pool advice

    primary:             object = None   # ChampionCandidate | None
    primary_reason:      str    = ""

    safest_alt:          object = None   # ChampionCandidate | None
    value_alt:           object = None   # ChampionCandidate | None

    def to_dict(self) -> dict:
        def _c(cand) -> dict | None:
            if cand is None:
                return None
            return {
                "name":        cand.name,
                "seed":        cand.seed,
                "region":      cand.region,
                "title_prob":  round(cand.win_prob, 4),
                "ff_prob":     round(cand.mc_ff_prob, 4),
                "public_pct":  round(cand.public_pct, 4),
                "value_score": round(cand.value_score, 3),
            }
        return {
            "pool_size":       self.pool_size,
            "tier":            self.tier,
            "tier_label":      self.tier_label,
            "n_brackets":      self.n_brackets,
            "strategy_note":   self.strategy_note,
            "primary":         _c(self.primary),
            "primary_reason":  self.primary_reason,
            "safest_alt":      _c(self.safest_alt),
            "value_alt":       _c(self.value_alt),
        }


# ── Tier classification ───────────────────────────────────────────────────────

def classify_pool(pool_size: int) -> str:
    """Return the tier key for a given pool size."""
    if pool_size <= SMALL_MAX:
        return "small_pool"
    if pool_size <= MEDIUM_MAX:
        return "medium_pool"
    if pool_size <= LARGE_MAX:
        return "large_pool"
    return "mega_pool"


def _n_brackets(tier: str, pool_size: int, primary_is_safest: bool) -> int:
    if tier == "small_pool":
        return 1
    if tier == "medium_pool":
        return 1 if primary_is_safest else 2
    if tier == "large_pool":
        if pool_size <= 300:
            return 2
        if pool_size <= 600:
            return 3
        return 4
    # mega_pool
    if pool_size <= 3000:
        return 5
    if pool_size <= 7000:
        return 7
    return 10


# ── Pick selectors ────────────────────────────────────────────────────────────

def _safest(candidates: list) -> object | None:
    viable = [c for c in candidates if c.win_prob >= _MIN_VIABLE_PROB]
    return max(viable, key=lambda c: c.win_prob) if viable else None


def _best_value(candidates: list) -> object | None:
    viable = [c for c in candidates if c.win_prob >= _MIN_VIABLE_PROB]
    return max(viable, key=lambda c: c.value_score) if viable else None


def _most_contrarian(candidates: list) -> object | None:
    pool = [
        c for c in candidates
        if c.win_prob >= _MIN_VIABLE_PROB and c.public_pct < _CONTRARIAN_PUB_CAP
    ]
    return max(pool, key=lambda c: c.value_score) if pool else None


def _medium_primary(candidates: list, safest_pick) -> object | None:
    """
    Best candidate with value_score ≥ threshold AND seed ≤ cap.
    Falls back to safest if none qualifies.
    """
    value_picks = [
        c for c in candidates
        if (c.win_prob >= _MIN_VIABLE_PROB
            and c.value_score >= _MEDIUM_VALUE_FLOOR
            and c.seed <= _MEDIUM_SEED_CAP)
    ]
    if value_picks:
        return max(value_picks, key=lambda c: c.composite if c.composite else c.value_score)
    return safest_pick


# ── Reason strings ────────────────────────────────────────────────────────────

def _reason(tier: str, cand, pool_size: int) -> str:
    if cand is None:
        return "No viable candidates found."
    name = cand.name
    tp   = cand.win_prob
    pub  = cand.public_pct
    vs   = cand.value_score
    ff   = cand.mc_ff_prob

    ff_note = f", {ff:.1%} Final Four probability" if ff > 0 else ""

    if tier == "small_pool":
        return (
            f"{name} has the highest title probability in the field ({tp:.1%})"
            f"{ff_note}. "
            f"In a shallow pool, picking the most likely winner maximises "
            f"your chance of taking the pot outright."
        )
    if tier == "medium_pool":
        if vs >= _MEDIUM_VALUE_FLOOR:
            ev_x = round(vs, 1)
            return (
                f"{name} carries a {tp:.1%} title probability against only "
                f"{pub:.1%} public pick share — {ev_x}× value{ff_note}. "
                f"In a {pool_size}-person pool this is the sweet spot: "
                f"meaningful win probability AND differentiation from the field."
            )
        return (
            f"{name} has the highest title probability ({tp:.1%}) and is the "
            f"safest single-bracket choice for your pool size."
        )
    if tier == "large_pool":
        splits = round(pool_size * pub)
        return (
            f"{name}: {tp:.1%} title probability, {pub:.1%} public pick share "
            f"({vs:.2f}× value{ff_note}). "
            f"Estimated ~{splits} other entries in your pool will pick this team — "
            f"but at {vs:.1f}× value, this is the highest-leverage champion available."
        )
    # mega_pool
    splits = max(1, round(pool_size * pub))
    return (
        f"{name}: {tp:.1%} title probability, {pub:.1%} public pick share "
        f"({vs:.2f}× value{ff_note}). "
        f"Only ~{splits} other entries in a {pool_size:,}-person pool pick this team. "
        f"If they win, the payout per surviving entry is maximised."
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def build_recommendation(
    candidates: list,
    pool_size:  int,
) -> Recommendation:
    """
    Build a pool-size-aware Recommendation from a list of ChampionCandidates.

    Parameters
    ----------
    candidates : list[ChampionCandidate], sorted descending by win_prob.
    pool_size  : number of entrants in the pool.

    Returns
    -------
    Recommendation with primary pick, alternatives, portfolio size advice.
    """
    tier  = classify_pool(pool_size)
    label = _TIER_LABELS[tier]
    note  = _TIER_STRATEGY_NOTES[tier]

    safest_pick = _safest(candidates)
    value_pick  = _best_value(candidates)
    contrarian  = _most_contrarian(candidates)

    if tier == "small_pool":
        primary = safest_pick
    elif tier == "medium_pool":
        primary = _medium_primary(candidates, safest_pick)
    elif tier == "large_pool":
        primary = value_pick
    else:  # mega_pool
        primary = contrarian or value_pick

    primary_is_safest = (
        primary is not None
        and safest_pick is not None
        and primary.name == safest_pick.name
    )

    n = _n_brackets(tier, pool_size, primary_is_safest)
    reason = _reason(tier, primary, pool_size)

    # Alternatives: show safest and value even if same as primary
    return Recommendation(
        pool_size       = pool_size,
        tier            = tier,
        tier_label      = label,
        n_brackets      = n,
        strategy_note   = note,
        primary         = primary,
        primary_reason  = reason,
        safest_alt      = safest_pick,
        value_alt       = value_pick,
    )


# ── Formatter ─────────────────────────────────────────────────────────────────

def format_recommendation(rec: Recommendation) -> str:
    lines: list[str] = []
    lines.append("=" * W)
    lines.append(
        f"  POOL RECOMMENDATION — {rec.tier_label.upper()}".center(W)
    )
    lines.append("=" * W)
    lines.append(f"\n  Pool size   : {rec.pool_size:,} entrants")
    lines.append(f"  Tier        : {rec.tier_label}")

    # Strategy note (word-wrapped at W-4 chars, uniform 2-space indent)
    lines.append(f"\n  Strategy:")
    for part in _wrap(rec.strategy_note, W - 4).splitlines():
        lines.append(f"  {part}")

    # Primary recommendation
    if rec.primary:
        c = rec.primary
        ff_str = f"   FF prob: {c.mc_ff_prob:.1%}" if c.mc_ff_prob > 0 else ""
        lines.append(f"\n  {'─' * (W - 2)}")
        lines.append(f"  RECOMMENDED CHAMPION")
        lines.append(f"  {'─' * (W - 2)}")
        lines.append(
            f"  {c.name}  (#{c.seed} {c.region})"
        )
        lines.append(
            f"  Title prob: {c.win_prob:.1%}   "
            f"Public: {c.public_pct:.1%}   "
            f"Value: {c.value_score:.2f}×{ff_str}"
        )
        lines.append(f"\n  Why:")
        for part in _wrap(rec.primary_reason, W - 4).splitlines():
            lines.append(f"  {part}")
        lines.append(
            f"\n  Recommended portfolio size: {rec.n_brackets} bracket"
            + ("s" if rec.n_brackets != 1 else "")
        )
    else:
        lines.append("\n  No viable champion candidate found.")

    # Alternatives
    lines.append(f"\n  {'─' * (W - 2)}")
    lines.append(f"  ALTERNATIVES")
    lines.append(f"  {'─' * (W - 2)}")
    _alt_line(lines, "Safest pick    ", rec.safest_alt, rec.primary)
    _alt_line(lines, "Best value pick", rec.value_alt,  rec.primary)

    lines.append("\n" + "=" * W)
    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _wrap(text: str, width: int) -> str:
    """Word-wrap text to width; return newline-joined string."""
    words = text.split()
    lines: list[str] = []
    line  = ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = line + " " + word if line else word
    if line:
        lines.append(line)
    return "\n".join(lines)


def _alt_line(lines: list, label: str, cand, primary) -> None:
    if cand is None:
        lines.append(f"  {label} : —")
        return
    same = primary is not None and cand.name == primary.name
    ff_str = f", {cand.mc_ff_prob:.1%} FF" if cand.mc_ff_prob > 0 else ""
    tag    = "  ← same as recommended" if same else ""
    lines.append(
        f"  {label} : {cand.name} (#{cand.seed} {cand.region})  "
        f"— {cand.win_prob:.1%} title, {cand.public_pct:.1%} public, "
        f"{cand.value_score:.2f}× value{ff_str}{tag}"
    )
