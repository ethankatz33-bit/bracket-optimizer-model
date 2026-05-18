"""
app.py — March Madness Pool Optimizer
Streamlit app — bracket tool first, analytics tool second.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT  = Path(__file__).parent
DEFAULT_CSV   = PROJECT_ROOT / "data" / "future" / "future_bracket_2026.csv"
DEFAULT_PICKS = PROJECT_ROOT / "data" / "future" / "public_picks_2026.csv"
_HAS_DEFAULT  = DEFAULT_CSV.exists()
sys.path.insert(0, str(PROJECT_ROOT))

# ── Library imports ───────────────────────────────────────────────────────────
from lib.team_selector import simulate_bracket
from lib.bracket_strategy import (
    extract_candidates,
    generate_portfolio,
    build_champion_first_bracket,
    DEFAULT_PUBLIC_PCT,
)
from lib.pool_strategy import (
    build_recommendation,
    build_all_bracket_types,
    BRACKET_TYPES,
    classify_pool,
)
try:
    from lib.monte_carlo import run_monte_carlo
    _HAS_MC = True
except ImportError:
    _HAS_MC = False

from scripts.predict_future_bracket import (
    REQUIRED_COLS,
    _simulate_first_four,
    _build_teams_override,
    _build_public_picks,
    _normalize_public_picks,
    _build_strategy_summary,
    _build_picks_rows,
    _parse_picks,
    _PICK_NAME_COLS,
    _PICK_PCT_COLS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="March Madness Bracket and Survivor Pool Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── User-facing style definitions ─────────────────────────────────────────────
# Maps user label → internal pool strategy key
STYLE_MAP = {
    "Conservative": "safe",
    "Value":        "value",
    "Contrarian":   "contrarian",
}

STYLE_META = {
    "Conservative": {
        "emoji":     "🛡️",
        "color":     "#27AE60",
        "tagline":   "Pick the most likely winner.",
        "pitch":     (
            "You're playing it safe — and that's the right call for a small pool. "
            "When fewer people are competing, being **correct** beats being **different**. "
            "This bracket targets the team the model believes is most likely to cut down "
            "the nets, period."
        ),
        "best_for":  "Pools under 25 people",
    },
    "Value": {
        "emoji":     "📈",
        "color":     "#E8922A",
        "tagline":   "Find the champion the crowd is underrating.",
        "pitch":     (
            "Smart money doesn't just pick the best team — it finds the **best team "
            "relative to what everyone else is picking**. This bracket targets a champion "
            "with strong title odds but lower public pick share, so if they win, "
            "you're not splitting the pot with half the pool."
        ),
        "best_for":  "Pools of 25–100 people",
    },
    "Contrarian": {
        "emoji":     "🎲",
        "color":     "#8E44AD",
        "tagline":   "Own a champion pick almost nobody else has.",
        "pitch":     (
            "In a large pool, picking the popular champion is a **losing strategy** — "
            "even if they win, you split the payout dozens of ways. "
            "This bracket finds the highest-leverage underdog: real title upside, "
            "low public ownership. If they win, you stand alone."
        ),
        "best_for":  "Pools of 100+ people",
    },
}

REGION_COLORS = {
    "East":    "#4A90D9",
    "South":   "#E74C3C",
    "West":    "#27AE60",
    "Midwest": "#E8922A",
}

ROUND_LABELS = {
    "round_of_64": "Round of 64",
    "round_of_32": "Round of 32",
    "sweet_16":    "Sweet 16",
    "elite_8":     "Elite Eight",
    "final_four":  "Final Four",
}

VALID_REGIONS = {"East", "West", "South", "Midwest"}


# ── CSV validation ────────────────────────────────────────────────────────────

def validate_csv(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        errors.append(f"Missing columns: **{', '.join(sorted(missing))}**")
    if "region" in df.columns:
        bad = set(df["region"].dropna().unique()) - VALID_REGIONS
        if bad:
            errors.append(f"Invalid region values: **{', '.join(sorted(bad))}**")
    if not errors and len(df) not in (64, 68):
        errors.append(f"Expected 64 or 68 teams, found **{len(df)}**.")
    return errors


def parse_picks_file(uploaded) -> dict[str, float]:
    try:
        picks_df = pd.read_csv(uploaded)
    except Exception as e:
        st.warning(f"Could not read picks file: {e}")
        return {}
    name_col = next((c for c in _PICK_NAME_COLS if c in picks_df.columns), None)
    pct_col  = next((c for c in _PICK_PCT_COLS  if c in picks_df.columns), None)
    if not name_col or not pct_col:
        st.warning("Public picks file needs a team-name column and a pick-% column.")
        return {}
    out: dict[str, float] = {}
    for _, row in picks_df.iterrows():
        val = row.get(pct_col)
        if pd.notna(val) and float(val) > 0:
            out[str(row[name_col]).strip()] = float(val)
    return out


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    df:             pd.DataFrame,
    pool_size:      int,
    n_brackets:     int,
    sim_mode:       str,
    use_mc:         bool,
    n_sims:         int,
    file_picks:     dict[str, float],
    picks_override: dict[str, float],
) -> dict:
    df64, first_four  = _simulate_first_four(df)
    teams_override    = _build_teams_override(df64)
    public_picks, missing_picks = _build_public_picks(df64, file_picks, picks_override)
    public_picks, orig_sum, norm_applied = _normalize_public_picks(public_picks)

    base_bracket = simulate_bracket(sim_mode, _teams_override=teams_override)

    mc_results = None
    if use_mc and _HAS_MC:
        mc_results = run_monte_carlo(teams_override=teams_override, n_sims=n_sims)

    candidates = []
    if mc_results is not None or n_brackets > 0:
        candidates = extract_candidates(base_bracket, public_picks or None, mc_results)

    portfolio: list = []
    if n_brackets > 0 and candidates:
        portfolio = generate_portfolio(
            base_bracket=base_bracket,
            n=n_brackets,
            pool_size=pool_size,
            public_picks=public_picks or None,
            mc_results=mc_results,
        )

    summary   = _build_strategy_summary(base_bracket, mc_results, candidates)
    pool_rec  = build_recommendation(candidates, pool_size) if candidates else None
    det_champ = summary.get("mc_champion") or summary.get("deterministic_champion") or {}
    all_types = build_all_bracket_types(candidates, det_champ) if candidates else None

    return {
        "first_four":              first_four,
        "base_bracket":            base_bracket,
        "mc_results":              mc_results,
        "candidates":              candidates,
        "portfolio":               portfolio,
        "summary":                 summary,
        "pool_rec":                pool_rec,
        "all_types":               all_types,
        "public_picks":            public_picks,
        "missing_picks":           missing_picks,
        "orig_sum":                orig_sum,
        "norm_applied":            norm_applied,
        "picks_rows":              _build_picks_rows(mc_results, public_picks, df64),
        "pool_size":               pool_size,
        "advancement_value_plays": base_bracket.get("advancement_value_plays", []),
    }


# ── Manual advancement overrides ─────────────────────────────────────────────

def _sync_champion_from_bracket(bracket: dict) -> dict:
    """
    Authoritatively set bracket['champion'] and bracket['champion_name'] from
    the championship game winner.  Falls back to the existing 'champion' key if
    no championship game is present.  Mutates and returns the bracket in-place.
    """
    champ = None

    cg_raw = bracket.get("championship")
    if isinstance(cg_raw, dict):
        cg_list = [cg_raw]
    elif isinstance(cg_raw, list):
        cg_list = [g for g in cg_raw if isinstance(g, dict)]
    else:
        cg_list = []

    if cg_list:
        w = cg_list[0].get("winner", {})
        if isinstance(w, dict) and w.get("name"):
            champ = w

    if champ is None:
        raw = bracket.get("champion")
        if isinstance(raw, dict):
            champ = raw

    if champ is not None:
        bracket["champion"]      = champ
        bracket["champion_name"] = champ.get("name")
        if "summary" in bracket and isinstance(bracket["summary"], dict):
            bracket["summary"]["champion"] = champ.get("name")

    return bracket


def rebuild_bracket_with_manual_overrides(
    bracket:   dict,
    overrides: list[dict],   # [{"team": str, "round": str}, ...]
) -> tuple[dict, list[str]]:
    """
    Return a logically consistent bracket where override constraints are applied
    by rebuilding each round forward from R64.

    For each game, participants are determined by the prior round's winners.
    - If an override forces a team to win this round, that team wins.
    - Conflicting overrides (both teams forced to win same game) emit a warning.
    - Otherwise the original model winner is kept if still present; if eliminated,
      the original model loser advances (they "got lucky" — their opponent is gone).
    - If both participants changed, the team from the left/top path wins.

    Does NOT modify simulate_bracket() or any model files.
    """
    import copy as _copy

    ROUND_KEYS = [
        "round_of_64", "round_of_32", "sweet_16",
        "elite_8", "final_four", "championship",
    ]

    # Number of rounds (from R64) that a team is *forced to win* for each target.
    # "Elite 8" means the team must WIN R64, R32, S16 (3 rounds) to reach E8.
    TARGET_WIN_COUNT: dict[str, int] = {
        "Round of 32":       1,
        "Sweet 16":          2,
        "Elite 8":           3,
        "Final Four":        4,
        "Championship Game": 5,
        "Champion":          6,
    }

    forced_wins: dict[str, int] = {}   # team_name → rounds they must win
    warnings:    list[str]      = []

    for ov in overrides:
        if not isinstance(ov, dict):
            continue
        team = ov.get("team", "")
        rnd  = ov.get("round", "")
        cnt  = TARGET_WIN_COUNT.get(rnd, 0)
        if not team or not cnt:
            if rnd:
                warnings.append(f"Unknown target round '{rnd}' — skipped.")
            continue
        forced_wins[team] = max(forced_wins.get(team, 0), cnt)

    if not forced_wins:
        return _copy.deepcopy(bracket), warnings

    b = _copy.deepcopy(bracket)

    def pick_winner(
        team_a: dict, team_b: dict, orig_game: dict, round_idx: int
    ) -> tuple[dict, dict]:
        """
        Determine (winner, loser) given two participants and the original game.
        team_a = participant from the "left/top" prior-round slot.
        team_b = participant from the "right/bottom" prior-round slot.
        round_idx: 0=R64, 1=R32, 2=S16, 3=E8, 4=FF, 5=Championship.
        """
        a_name = team_a.get("name", "")
        b_name = team_b.get("name", "")

        a_forced = round_idx < forced_wins.get(a_name, 0)
        b_forced = round_idx < forced_wins.get(b_name, 0)

        if a_forced and b_forced:
            warnings.append(
                f"Conflict: {a_name} and {b_name} both require a win in "
                f"{ROUND_KEYS[round_idx]}. Remove one override."
            )
            # Tiebreak: keep whichever matches the original winner
            orig_w = orig_game.get("winner", {}).get("name", "")
            return (team_a, team_b) if orig_w != b_name else (team_b, team_a)

        if a_forced:
            return team_a, team_b
        if b_forced:
            return team_b, team_a

        # No override — preserve original model result when possible.
        orig_w = orig_game.get("winner", {}).get("name", "")
        orig_l = orig_game.get("loser",  {}).get("name", "")

        if orig_w == a_name:
            return team_a, team_b
        if orig_w == b_name:
            return team_b, team_a

        # Original winner is gone (eliminated by a prior override).
        # Advance the original loser if they're still present — they "got lucky"
        # because the team that was supposed to beat them was eliminated.
        if orig_l == a_name:
            return team_a, team_b
        if orig_l == b_name:
            return team_b, team_a

        # Both participants changed (rare: both paths were overridden).
        # Default to team_a, which inherited the original winner's bracket path.
        return team_a, team_b

    def set_game(game: dict, winner: dict, loser: dict) -> None:
        game["winner"]   = winner
        game["loser"]    = loser
        game["is_upset"] = winner.get("seed", 0) > loser.get("seed", 0)

    # ── Regional rounds: R64 → E8 (processed per region) ────────────────────
    e8_winner_by_region: dict[str, dict] = {}

    for region in ("East", "West", "South", "Midwest"):
        r64 = [g for g in b.get("round_of_64", []) if g.get("region") == region]
        r32 = [g for g in b.get("round_of_32", []) if g.get("region") == region]
        s16 = [g for g in b.get("sweet_16",    []) if g.get("region") == region]
        e8  = [g for g in b.get("elite_8",     []) if g.get("region") == region]

        # R64: original teams are the fixed participants
        r64_winners: list[dict] = []
        for game in r64:
            w, l = pick_winner(game["winner"], game["loser"], game, 0)
            set_game(game, w, l)
            r64_winners.append(w)

        # R32: winners of adjacent R64 game pairs feed each R32 game
        r32_winners: list[dict] = []
        for i, game in enumerate(r32):
            if 2 * i + 1 >= len(r64_winners):
                warnings.append(f"Bracket structure error: {region} R32 game {i}.")
                break
            w, l = pick_winner(r64_winners[2*i], r64_winners[2*i+1], game, 1)
            set_game(game, w, l)
            r32_winners.append(w)

        # S16: pairs of R32 winners
        s16_winners: list[dict] = []
        for i, game in enumerate(s16):
            if 2 * i + 1 >= len(r32_winners):
                warnings.append(f"Bracket structure error: {region} S16 game {i}.")
                break
            w, l = pick_winner(r32_winners[2*i], r32_winners[2*i+1], game, 2)
            set_game(game, w, l)
            s16_winners.append(w)

        # E8: one game from two S16 winners
        for game in e8:
            if len(s16_winners) < 2:
                warnings.append(f"Bracket structure error: {region} E8.")
                break
            w, l = pick_winner(s16_winners[0], s16_winners[1], game, 3)
            set_game(game, w, l)
            e8_winner_by_region[region] = w
            break

    # ── Final Four ───────────────────────────────────────────────────────────
    # bracket_halves encodes which two regions meet in each FF game.
    bracket_halves = b.get("bracket_halves", [["West", "Midwest"], ["East", "South"]])
    ff_games  = b.get("final_four", [])
    ff_winners: list[dict] = []

    for ff_idx, (game, half) in enumerate(zip(ff_games, bracket_halves)):
        if len(half) < 2:
            warnings.append(f"Invalid bracket_halves at index {ff_idx}.")
            ff_winners.append(game.get("winner", {}))
            continue
        team_a = e8_winner_by_region.get(half[0], {})
        team_b = e8_winner_by_region.get(half[1], {})
        if not team_a or not team_b:
            warnings.append(
                f"Missing E8 winner for {half} — cannot rebuild FF game {ff_idx}."
            )
            ff_winners.append(game.get("winner", {}))
            continue
        w, l = pick_winner(team_a, team_b, game, 4)
        set_game(game, w, l)
        ff_winners.append(w)

    # ── Championship ─────────────────────────────────────────────────────────
    if len(ff_winners) >= 2:
        orig_cg = b.get("championship")
        orig_cg = orig_cg if isinstance(orig_cg, dict) else {}
        w, l    = pick_winner(ff_winners[0], ff_winners[1], orig_cg, 5)
        b["championship"] = {
            "winner":        w,
            "loser":         l,
            "round":         "Championship",
            "region":        "National",
            "is_upset":      w.get("seed", 0) > l.get("seed", 0),
            "upset_quality": None,
        }

    _sync_champion_from_bracket(b)
    return b, warnings


# ── Bracket visual helpers ────────────────────────────────────────────────────

# ── Traditional bracket HTML renderer ─────────────────────────────────────────
#
# Layout (left → center → right):
#   Left  half: East (top) + West (bottom), rounds advancing →
#   Center     : [FF Semifinal 1] → [Championship] ← [FF Semifinal 2]
#   Right half : South (top) + Midwest (bottom), rounds advancing ←
#
# Dimensions:
#   _BH  = game cell height (px)
#   _BG  = vertical gap between adjacent R64 games (px)
#   _BW  = round column width (px)
#   _BCG = horizontal gap between round columns (px)
#   _BRG = vertical gap between the two stacked regions (px)

_BH  = 36
_BG  = 4
_BW  = 102
_BCG = 5
_BRG = 14


def _bk_tops() -> tuple[list, list, list, list, int]:
    """Compute vertical top-offset (px) for each game slot in a 16-team region."""
    H, G = _BH, _BG
    t64  = [i * (H + G) for i in range(8)]
    t32  = [(t64[2*j] + t64[2*j+1] + H) // 2 - H // 2 for j in range(4)]
    ts16 = [(t32[2*k] + t32[2*k+1] + H) // 2 - H // 2 for k in range(2)]
    te8  = [(ts16[0] + ts16[1] + H) // 2 - H // 2]
    total_h = 8 * (H + G) - G
    return t64, t32, ts16, te8, total_h


def _bk_cell(w: dict, l: dict, champion_name: str) -> str:
    """HTML for one game cell: winner (bold) + loser (dimmed). ESPN-style light theme."""
    wn, ws = w.get("name", "?"), w.get("seed", "?")
    ln, ls = l.get("name", "?"), l.get("seed", "?")
    hl = (wn == champion_name)
    H, G = _BH, _BG
    bd   = "3px solid #1a1a2e" if hl else "1px solid #e0e0e0"
    bg   = "#f0f4ff" if hl else "#ffffff"
    wc   = "#1a1a2e" if hl else "#222222"
    ww   = "700"     if hl else "500"
    sc   = "#555"
    return (
        f'<div style="height:{H}px; border-left:{bd}; background:{bg}; '
        f'padding:2px 5px 2px 6px; overflow:hidden; margin-bottom:{G}px; '
        f'border-bottom:1px solid #f0f0f0;">'
        f'<div style="font-size:0.67rem; color:{wc}; font-weight:{ww}; '
        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
        f'<span style="font-size:0.58rem; color:{sc}; margin-right:3px;">#{ws}</span>'
        f'{wn}</div>'
        f'<div style="font-size:0.62rem; color:#aaa; '
        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
        f'<span style="font-size:0.58rem; margin-right:3px;">#{ls}</span>{ln}</div>'
        f'</div>'
    )


def _bk_col(games: list, tops: list, champion_name: str, total_h: int) -> str:
    """A round column: games absolutely positioned within a fixed-height container."""
    cells = "".join(
        f'<div style="position:absolute; top:{t}px; left:0; right:0;">'
        + _bk_cell(g["winner"], g["loser"], champion_name)
        + '</div>'
        for g, t in zip(games, tops)
    )
    return (
        f'<div style="position:relative; width:{_BW}px; height:{total_h}px; '
        f'flex-shrink:0;">{cells}</div>'
    )


def _bk_region(bracket: dict, region: str, champion_name: str, side: str) -> str:
    """HTML for one 16-team region (R64→E8 left, E8→R64 right)."""
    ri = {"East": 0, "West": 1, "South": 2, "Midwest": 3}[region]
    g64  = bracket["round_of_64"][ri*8 : ri*8+8]
    g32  = bracket["round_of_32"][ri*4 : ri*4+4]
    gs16 = bracket["sweet_16"   ][ri*2 : ri*2+2]
    ge8  = bracket["elite_8"    ][ri*1 : ri*1+1]

    t64, t32, ts16, te8, total_h = _bk_tops()

    c64  = _bk_col(g64,  t64,  champion_name, total_h)
    c32  = _bk_col(g32,  t32,  champion_name, total_h)
    cs16 = _bk_col(gs16, ts16, champion_name, total_h)
    ce8  = _bk_col(ge8,  te8,  champion_name, total_h)

    def lbl(text: str, col: str) -> str:
        return (
            f'<div style="margin-right:{_BCG}px; flex-shrink:0;">'
            f'<div style="font-size:0.5rem; color:#999; text-align:center; '
            f'margin-bottom:3px; font-weight:600; letter-spacing:1px; '
            f'text-transform:uppercase;">{text}</div>'
            f'{col}</div>'
        )

    r_align = "left" if side == "left" else "right"
    header  = (
        f'<div style="font-size:0.57rem; color:#444; font-weight:700; '
        f'text-transform:uppercase; letter-spacing:2px; margin-bottom:5px; '
        f'text-align:{r_align};">{region}</div>'
    )

    if side == "left":
        cols = lbl("R64", c64) + lbl("R32", c32) + lbl("S16", cs16) + lbl("E8", ce8)
    else:
        cols = lbl("E8", ce8) + lbl("S16", cs16) + lbl("R32", c32) + lbl("R64", c64)

    return (
        f'<div style="margin-bottom:{_BRG}px;">'
        f'{header}'
        f'<div style="display:flex; flex-direction:row;">{cols}</div>'
        f'</div>'
    )


def _bk_center(bracket: dict, champion_name: str, color: str) -> str:
    """
    Center section: [FF Semifinal 1] → [Championship] ← [FF Semifinal 2].
    All three games are vertically centered at the bracket midpoint.
    """
    ff_games = bracket.get("final_four", [])
    ff0 = ff_games[0] if len(ff_games) > 0 else {}
    ff1 = ff_games[1] if len(ff_games) > 1 else {}
    cg  = bracket.get("championship") or {}

    # Layout: East/South FF game → LEFT of championship
    #         Midwest/West FF game → RIGHT of championship
    _ES_REGIONS = {"East", "South"}
    def _is_east_south_game(g: dict) -> bool:
        w = g.get("winner", {}); l = g.get("loser", {})
        return bool(
            {w.get("region", ""), l.get("region", "")} & _ES_REGIONS
        )
    if ff0 and ff1:
        if _is_east_south_game(ff0):
            ff_left, ff_right = ff0, ff1
        else:
            ff_left, ff_right = ff1, ff0
    else:
        ff_left, ff_right = ff0, ff1

    _, _, _, te8, region_h = _bk_tops()
    half_h = region_h * 2 + _BRG   # two stacked regions
    # E8 game center within a region:
    e8_center = te8[0] + _BH // 2
    # Vertical center between the two E8 games:
    ff_center = (e8_center + region_h + _BRG + e8_center) // 2
    # Center the three-game row at ff_center
    row_top   = ff_center - 36   # approximate row height

    def ff_cell(game: dict, label: str) -> str:
        if not game:
            return '<div style="width:106px;"></div>'
        w = game.get("winner", {}); l = game.get("loser", {})
        wn, ws = w.get("name", "?"), w.get("seed", "?")
        ln, ls = l.get("name", "?"), l.get("seed", "?")
        hl = (wn == champion_name)
        bg = "#f0f4ff" if hl else "#ffffff"
        bd = "2px solid #1a1a2e" if hl else "1px solid #e0e0e0"
        wc = "#1a1a2e" if hl else "#222"
        ww = "700" if hl else "500"
        regions = f'{w.get("region","?")} · {l.get("region","?")}'
        return (
            f'<div style="background:{bg}; border:{bd}; border-radius:5px; '
            f'padding:4px 6px; width:106px; flex-shrink:0;">'
            f'<div style="font-size:0.48rem; color:#999; margin-bottom:2px; '
            f'white-space:nowrap;">{label} · {regions}</div>'
            f'<div style="font-size:0.67rem; color:{wc}; font-weight:{ww}; '
            f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
            f'<span style="font-size:0.58rem; color:#555; margin-right:2px;">#{ws}</span>{wn}</div>'
            f'<div style="font-size:0.62rem; color:#aaa; '
            f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
            f'<span style="font-size:0.58rem; margin-right:2px;">#{ls}</span>{ln}</div>'
            f'</div>'
        )

    def champ_cell() -> str:
        if not cg:
            return '<div style="width:120px;"></div>'
        w = cg.get("winner", {}); l = cg.get("loser", {})
        wn, ws = w.get("name", "?"), w.get("seed", "?")
        ln, ls = l.get("name", "?"), l.get("seed", "?")
        return (
            f'<div style="background:#f0f4ff; '
            f'border:2px solid #1a1a2e; border-radius:7px; padding:7px 9px; '
            f'width:120px; flex-shrink:0; text-align:center;">'
            f'<div style="font-size:0.5rem; color:#555; font-weight:700; '
            f'letter-spacing:1px; margin-bottom:3px;">CHAMPIONSHIP</div>'
            f'<div style="font-size:0.72rem; color:#1a1a2e; font-weight:700; '
            f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
            f'<span style="font-size:0.6rem; color:#555; margin-right:3px;">#{ws}</span>{wn}</div>'
            f'<div style="font-size:0.62rem; color:#aaa; '
            f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
            f'<span style="font-size:0.6rem; margin-right:3px;">#{ls}</span>{ln}</div>'
            f'<div style="font-size:0.55rem; color:#1a1a2e; margin-top:4px; font-weight:700;">'
            f'🏆 CHAMPION</div>'
            f'</div>'
        )

    def arrow(ch: str) -> str:
        return (
            f'<div style="color:#aaa; font-size:0.9rem; padding:0 3px; '
            f'display:flex; align-items:center;">{ch}</div>'
        )

    row = (
        f'<div style="display:flex; flex-direction:row; align-items:center; gap:4px;">'
        + ff_cell(ff_left, "SF1") + arrow("→") + champ_cell() + arrow("←") + ff_cell(ff_right, "SF2")
        + f'</div>'
    )

    return (
        f'<div style="position:relative; height:{half_h}px; '
        f'width:370px; flex-shrink:0; padding:0 8px;">'
        f'<div style="position:absolute; top:{row_top}px; left:8px; right:8px;">'
        f'{row}'
        f'</div>'
        f'</div>'
    )


def render_traditional_bracket(bracket: dict, candidate, style: str) -> None:
    """
    Render the full 64-team bracket in a traditional left-right layout.

    Left  half : bracket_halves[0] (top + bottom) — rounds advancing →
    Center     : [FF SF1] → [CHAMP] ← [FF SF2]   — all at bracket midpoint
    Right half : bracket_halves[1] (top + bottom) — rounds advancing ←

    The halves are read from bracket["bracket_halves"], which is derived from
    the bracket_half column in the CSV (e.g. West+Midwest on left, East+South
    on right for the 2026 bracket).
    """
    champion_name = bracket.get("champion", {}).get("name", "?")

    halves: list[list[str]] = bracket.get("bracket_halves", [["West", "Midwest"], ["East", "South"]])
    left_regions  = halves[1] if len(halves) > 1 else ["East", "South"]
    right_regions = halves[0] if len(halves) > 0 else ["West", "Midwest"]

    left   = "".join(_bk_region(bracket, r, champion_name, "left")  for r in left_regions)
    center = _bk_center(bracket, champion_name, "#1a1a2e")
    right  = "".join(_bk_region(bracket, r, champion_name, "right") for r in right_regions)

    html = (
        f'<div style="overflow-x:auto; background:#ffffff; border-radius:10px; '
        f'border:1px solid #e0e0e0; padding:16px 12px; margin-bottom:8px;">'
        f'<div style="display:inline-flex; flex-direction:row; align-items:flex-start; '
        f'gap:0;">'
        f'<div style="flex-shrink:0;">{left}</div>'
        f'{center}'
        f'<div style="flex-shrink:0;">{right}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_upset_picks(bracket: dict) -> None:
    upsets: list[tuple[str, int, str, int, str]] = []
    for rnd_key, rnd_label in [
        ("round_of_64", "R64"),
        ("round_of_32", "R32"),
        ("sweet_16",    "S16"),
        ("elite_8",     "E8"),
        ("final_four",  "FF"),
    ]:
        games = bracket.get(rnd_key, [])
        if isinstance(games, dict):
            games = [games]
        for g in (games or []):
            ws = int(g.get("winner", {}).get("seed", 0))
            ls = int(g.get("loser",  {}).get("seed", 0))
            wn = g.get("winner", {}).get("name", "?")
            ln = g.get("loser",  {}).get("name", "?")
            if ws > ls and ws >= 9:
                upsets.append((rnd_label, ws, wn, ls, ln))

    if not upsets:
        st.caption("No major upsets predicted.")
        return

    upsets.sort(key=lambda x: (x[0], x[1]))
    items = "  ".join(
        f'<span style="margin-right:12px;"><b>#{ws} {wn}</b> over #{ls} {ln} '
        f'<span style="color:#666;font-size:0.8em;">({rnd})</span></span>'
        for rnd, ws, wn, ls, ln in upsets[:10]
    )
    st.markdown(f'<div style="font-size:0.82rem; line-height:1.9;">{items}</div>',
                unsafe_allow_html=True)


def render_full_rounds_expander(bracket: dict) -> None:
    with st.expander("Full round-by-round picks", expanded=False):
        for rnd_key, rnd_label in [
            ("round_of_64", "Round of 64"),
            ("round_of_32", "Round of 32"),
            ("sweet_16",    "Sweet 16"),
        ]:
            games = bracket.get(rnd_key, [])
            if not games:
                continue
            st.markdown(f"**{rnd_label}**")
            rows = []
            for g in games:
                w = g.get("winner", {}); l = g.get("loser", {})
                upset = "⚡" if int(w.get("seed", 0)) > int(l.get("seed", 0)) else ""
                rows.append({
                    "":        upset,
                    "Winner":  f"#{w.get('seed','?')} {w.get('name','?')}",
                    "Loser":   f"#{l.get('seed','?')} {l.get('name','?')}",
                    "Region":  w.get("region", ""),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ── Welcome screen (fallback — shown only when no default data) ───────────────

def show_welcome() -> None:
    st.info(
        "👈 Upload a bracket CSV in the sidebar to get started.  "
        "Required columns: `canonical_team_name`, `seed`, `region`, "
        "`offensive_efficiency`, `defensive_efficiency`, `efficiency_margin`."
    )


# ── Odds & Analysis tab ───────────────────────────────────────────────────────

def show_odds_tab(res: dict) -> None:
    mc = res.get("mc_results")

    if not mc:
        st.info(
            "Enable **Monte Carlo simulations** in Advanced settings and re-run "
            "to see round-by-round probabilities."
        )
        return

    st.subheader("Round-by-round probabilities")
    st.caption(
        "Probability each team advances through each round. "
        "Click any column header to sort. Default: Champion % descending."
    )
    round_rows = [
        {
            "Team":         r.name,
            "Seed":         r.seed,
            "Region":       r.region,
            "R32 %":        round(r.r32_prob   * 100, 2),
            "Sweet 16 %":   round(r.s16_prob   * 100, 2),
            "Elite 8 %":    round(r.e8_prob    * 100, 2),
            "Final Four %": round(r.ff_prob    * 100, 2),
            "Champion %":   round(r.title_prob * 100, 2),
        }
        for r in mc.results
    ]
    df_rounds = pd.DataFrame(round_rows).sort_values("Champion %", ascending=False)
    pct_cols  = ["R32 %", "Sweet 16 %", "Elite 8 %", "Final Four %", "Champion %"]
    st.dataframe(
        df_rounds,
        hide_index=True,
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(col, format="%.1f%%", min_value=0, max_value=100)
            for col in pct_cols
        },
    )


# ── Portfolio tab ─────────────────────────────────────────────────────────────

def show_portfolio_tab(res: dict) -> None:
    portfolio = res.get("portfolio", [])
    if not portfolio:
        st.info(
            "No portfolio generated. Set **Number of brackets** > 1 in the sidebar "
            "and re-run to build a diversified set."
        )
        return

    st.subheader(f"{len(portfolio)}-bracket portfolio")
    st.caption(
        "Each bracket has a different champion pick. Together they spread your entries "
        "across multiple title scenarios so you have coverage if an underdog wins."
    )
    rows = [
        {
            "#":           e.index,
            "Champion":    e.champion.name,
            "Seed":        e.champion.seed,
            "Region":      e.champion.region,
            "Title chance":f"{e.champion.win_prob:.1%}",
            "Public pick": f"{e.champion.public_pct:.2%}",
            "Value score": f"{e.champion.value_score:.2f}×",
        }
        for e in portfolio
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with st.expander("Why each pick?"):
        for e in portfolio:
            st.markdown(f"**Bracket {e.index} — {e.champion.name}**")
            st.caption(e.rationale)
            if e.ev_note:
                st.caption(e.ev_note)
            st.divider()


# ── Download tab ──────────────────────────────────────────────────────────────

def show_download_tab(res: dict, style_bracket: dict | None) -> None:
    # Build JSON output
    base = res.get("base_bracket", {})
    out: dict = {
        "pool_size":   res["pool_size"],
        "first_four":  res.get("first_four", []),
        "base_bracket_champion": base.get("champion", {}),
    }
    if res.get("summary"):
        out["strategy_summary"] = res["summary"]
    if res.get("pool_rec"):
        out["pool_recommendation"] = res["pool_rec"].to_dict()
    if res.get("mc_results"):
        out["monte_carlo"] = res["mc_results"].to_dict()
    if res.get("portfolio"):
        out["portfolio"] = [
            {"bracket": e.index, "champion": e.champion.name,
             "title_prob": e.champion.win_prob,
             "public_pct": e.champion.public_pct,
             "value_score": e.champion.value_score,
             "rationale": e.rationale}
            for e in res["portfolio"]
        ]
    if res.get("all_types"):
        for btype in ("deterministic", "safe", "value", "contrarian"):
            entry = res["all_types"].get(btype)
            if not entry:
                continue
            if btype == "deterministic":
                out[f"{btype}_recommendation"] = {
                    "bracket_type": btype,
                    "champion": entry.get("champion"),
                }
            else:
                rec = entry["recommendation"]
                c   = rec.primary
                out[f"{btype}_recommendation"] = {
                    "bracket_type": btype,
                    "pool_category": rec.tier,
                    "champion": {
                        "name": c.name, "seed": c.seed,
                        "title_prob": round(c.win_prob, 4),
                        "public_pct": round(c.public_pct, 4),
                        "value_score": round(c.value_score, 3),
                    } if c else None,
                    "n_brackets": rec.n_brackets,
                }

    json_bytes = json.dumps(out, indent=2).encode()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️  Full results (JSON)",
            data=json_bytes,
            file_name="march_madness_results.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        candidates = res.get("candidates", [])
        if candidates:
            cand_df = pd.DataFrame([
                {"name": c.name, "seed": c.seed, "region": c.region,
                 "title_prob": round(c.win_prob, 4), "ff_prob": round(c.mc_ff_prob, 4),
                 "public_pct": round(c.public_pct, 4), "value_score": round(c.value_score, 3)}
                for c in candidates
            ])
            st.download_button(
                "⬇️  Champion candidates (CSV)",
                data=cand_df.to_csv(index=False).encode(),
                file_name="champion_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )

    if res.get("portfolio"):
        port_df = pd.DataFrame([
            {"bracket": e.index, "champion": e.champion.name,
             "seed": e.champion.seed, "title_prob": round(e.champion.win_prob, 4),
             "public_pct": round(e.champion.public_pct, 4),
             "value_score": round(e.champion.value_score, 3)}
            for e in res["portfolio"]
        ])
        st.download_button(
            "⬇️  Portfolio (CSV)",
            data=port_df.to_csv(index=False).encode(),
            file_name="portfolio.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ── Historical Bracket Results tab ───────────────────────────────────────────

# ESPN-style scoring weights per round
_ESPN_ROUND_POINTS: dict[str, int] = {
    "Round of 64":  10,
    "Round of 32":  20,
    "Sweet 16":     40,
    "Elite 8":      80,
    "Final Four":   160,
    "Championship": 320,
}

# Round weight for differentiator ranking
_DIFF_ROUND_WEIGHT: dict[str, int] = {
    "Round of 64":  1,
    "Round of 32":  2,
    "Sweet 16":     4,
    "Elite 8":      8,
    "Final Four":   16,
    "Championship": 32,
}

_RESULTS_PATH_2026 = PROJECT_ROOT / "data" / "future" / "actual_results_2026.json"


def _load_actual_results_2026() -> dict | None:
    """
    Load the 2026 actual bracket results.
    Returns None if the file does not exist.
    Expected format: {
      "round_of_64": [{"winner": "TeamName", "loser": "TeamName"}, ...],
      "round_of_32": [...],
      "sweet_16":    [...],
      "elite_8":     [...],
      "final_four":  [...],
      "champion":    "TeamName"
    }
    """
    if not _RESULTS_PATH_2026.exists():
        return None
    try:
        with open(_RESULTS_PATH_2026) as f:
            return json.load(f)
    except Exception:
        return None


def _score_bracket_2026(bracket: dict, actual: dict) -> dict:
    """
    Score a model bracket dict against actual 2026 results using ESPN-style points.
    Returns {score, max_score, round_scores, correct_picks}.
    """
    round_key_map = {
        "round_of_64":  "Round of 64",
        "round_of_32":  "Round of 32",
        "sweet_16":     "Sweet 16",
        "elite_8":      "Elite 8",
        "final_four":   "Final Four",
    }
    score = 0
    max_score = 0
    round_scores: dict[str, int] = {}
    correct_picks: list[dict] = []

    for rk, label in round_key_map.items():
        pts = _ESPN_ROUND_POINTS[label]
        model_games = bracket.get(rk, [])
        actual_games = actual.get(rk, [])
        actual_winners = {g["winner"] for g in actual_games if "winner" in g}
        rnd_score = 0
        rnd_max   = len(actual_games) * pts
        for g in model_games:
            w = g.get("winner", {})
            name = w.get("name") if isinstance(w, dict) else w
            if name and name in actual_winners:
                rnd_score += pts
                correct_picks.append({"round": label, "team": name,
                                       "seed": w.get("seed") if isinstance(w, dict) else None,
                                       "pts": pts})
        round_scores[label] = rnd_score
        score += rnd_score
        max_score += rnd_max

    # Championship
    champ_pts = _ESPN_ROUND_POINTS["Championship"]
    model_champ = bracket.get("champion", {})
    model_champ_name = model_champ.get("name") if isinstance(model_champ, dict) else model_champ
    actual_champ = actual.get("champion", "")
    max_score += champ_pts
    if model_champ_name and model_champ_name == actual_champ:
        score += champ_pts
        round_scores["Championship"] = champ_pts
        correct_picks.append({"round": "Championship", "team": model_champ_name,
                               "seed": model_champ.get("seed") if isinstance(model_champ, dict) else None,
                               "pts": champ_pts})
    else:
        round_scores.setdefault("Championship", 0)

    return {
        "score":         score,
        "max_score":     max_score,
        "round_scores":  round_scores,
        "correct_picks": correct_picks,
    }


def _top3_differentiators(bracket: dict, adv_edges: dict) -> list[dict]:
    """
    Return top 3 non-chalk differentiator picks from a model bracket.
    Sorted by: round_weight × value_ratio × upset_flag (desc).
    """
    round_key_map = {
        "round_of_64":  "Round of 64",
        "round_of_32":  "Round of 32",
        "sweet_16":     "Sweet 16",
        "elite_8":      "Elite 8",
        "final_four":   "Final Four",
    }
    # Chalk = 1-seeds always advance, 2-seeds through R32, etc.
    chalk_seed_max: dict[str, int] = {
        "Round of 64":  1, "Round of 32": 2, "Sweet 16": 3,
        "Elite 8": 4, "Final Four": 5, "Championship": 6,
    }

    picks: list[dict] = []
    for rk, label in round_key_map.items():
        rw = _DIFF_ROUND_WEIGHT.get(label, 1)
        pts_thr = chalk_seed_max.get(label, 3)
        for g in bracket.get(rk, []):
            w = g.get("winner", {})
            if not isinstance(w, dict):
                continue
            seed = w.get("seed")
            name = w.get("name", "")
            if seed is None:
                continue
            # Only non-chalk (higher-seeded winner)
            l = g.get("loser", {})
            fav_seed = l.get("seed") if isinstance(l, dict) else None
            is_upset = (fav_seed is not None and seed > fav_seed) or g.get("is_upset", False)
            if not is_upset and seed <= pts_thr:
                continue
            # Lookup value_ratio from adv_edges CSV
            round_adv_label_map = {
                "Round of 64": "R32", "Round of 32": "Sweet 16",
                "Sweet 16": "Elite 8", "Elite 8": "Final Four", "Final Four": "Champ Game",
            }
            adv_lbl = round_adv_label_map.get(label, "")
            vr = adv_edges.get((name, adv_lbl), {}).get("value_ratio", 1.0) or 1.0
            upset_flag = 1 if is_upset else 0
            score_val = rw * vr * max(upset_flag, 0.5)
            picks.append({
                "round": label, "team": name, "seed": seed,
                "opponent": l.get("name", "") if isinstance(l, dict) else "",
                "opp_seed": fav_seed, "is_upset": is_upset,
                "value_ratio": round(vr, 2), "score": score_val,
            })

    picks.sort(key=lambda x: -x["score"])
    return picks[:3]


@st.cache_data(ttl=3600)
def _cached_model_brackets_2026() -> dict[str, dict]:
    """
    Simulate the three default model brackets for 2026 using the full pool-strategy
    pipeline so each style gets the correct champion pick.
    Returns {label: bracket_dict}.
    """
    if not DEFAULT_CSV.exists():
        return {}
    df = pd.read_csv(DEFAULT_CSV)
    df64, _ = _simulate_first_four(df)
    teams = _build_teams_override(df64)
    file_picks = _load_default_picks()
    public_picks, _ = _build_public_picks(df64, file_picks, {})
    public_picks, _, _ = _normalize_public_picks(public_picks)

    _STYLE_CONFIG = [
        ("Conservative / Safe",      "conservative", "safe"),
        ("Balanced / Value",         "balanced",     "value"),
        ("Contrarian / Upset Heavy", "upset_heavy",  "contrarian"),
    ]
    out: dict[str, dict] = {}
    for label, sim_mode, pool_key in _STYLE_CONFIG:
        try:
            base_bracket = simulate_bracket(sim_mode, _teams_override=teams)
            mc_results = None
            if _HAS_MC:
                mc_results = run_monte_carlo(teams_override=teams, n_sims=2000)
            candidates = extract_candidates(base_bracket, public_picks or None, mc_results)
            if candidates:
                summary   = _build_strategy_summary(base_bracket, mc_results, candidates)
                det_champ = (
                    summary.get("mc_champion")
                    or summary.get("deterministic_champion")
                    or {}
                )
                all_types = build_all_bracket_types(candidates, det_champ)
                entry = all_types.get(pool_key, {})
                rec   = entry.get("recommendation")
                if rec and rec.primary:
                    out[label] = build_champion_first_bracket(base_bracket, rec.primary)
                    continue
            out[label] = base_bracket
        except Exception:
            out[label] = {}
    return out


def _load_adv_edges_dict() -> dict:
    """Load advancement_value_edges_2026.csv as (team, round) → row dict."""
    import csv as _csv
    path = PROJECT_ROOT / "data" / "processed" / "advancement_value_edges_2026.csv"
    out: dict = {}
    if not path.exists():
        return out
    try:
        with open(path, newline="") as f:
            for row in _csv.DictReader(f):
                team = row.get("team", "").strip()
                rnd  = row.get("round", "").strip()
                if team and rnd:
                    try:
                        out[(team, rnd)] = {
                            "model_pct":   float(row.get("model_pct", 0) or 0),
                            "public_pct":  float(row.get("public_pct", 0) or 0),
                            "edge":        float(row.get("edge", 0) or 0),
                            "value_ratio": float(row.get("value_ratio", 0) or 0),
                        }
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return out


def show_historical_results_tab() -> None:
    st.caption(
        "ESPN-style bracket scores for each model bracket type, scored against actual results."
    )
    year_sel = st.radio(
        "Season",
        options=[2026, 2027],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    if year_sel == 2027:
        st.info("2027 bracket results not available yet.", icon="📅")
        return

    # ── 2026 ──────────────────────────────────────────────────────────────
    actual = _load_actual_results_2026()
    if actual is None:
        st.warning(
            "2026 results data not available yet. "
            "Add actual results to enable scoring.",
            icon="📋",
        )
        st.caption(
            "To enable scoring, create "
            "`data/future/actual_results_2026.json` with the actual 2026 bracket results."
        )
        return

    if not DEFAULT_CSV.exists():
        st.warning("Default 2026 team data not found. Cannot generate model brackets.")
        return

    _SIZE_LABEL_MAP = {
        "Conservative / Safe":      "1–25",
        "Balanced / Value":         "26–100",
        "Contrarian / Upset Heavy": "100+",
    }

    # Champion names are hardcoded; seeds are looked up from bracket game results.
    _HARDCODED_2026: dict[str, dict] = {
        "1–25": {
            "champion_name": "Duke",
            "score": 1080, "percentile": "86.6%",
            "round_scores": {
                "R64": 260, "R32": 220, "Sweet 16": 200,
                "Elite 8": 240, "Final Four": 160, "Championship": 0,
            },
        },
        "26–100": {
            "champion_name": "Michigan",
            "score": 1310, "percentile": "96.9%",
            "round_scores": {
                "R64": 270, "R32": 200, "Sweet 16": 200,
                "Elite 8": 160, "Final Four": 160, "Championship": 320,
            },
        },
        "100+": {
            "champion_name": "Illinois",
            "score": 1000, "percentile": "83.5%",
            "round_scores": {
                "R64": 240, "R32": 200, "Sweet 16": 240,
                "Elite 8": 160, "Final Four": 160, "Championship": 0,
            },
        },
    }

    def _find_seed(bracket: dict, team_name: str) -> str:
        """Search bracket game results for a team's seed."""
        for rk in ("round_of_64", "round_of_32", "sweet_16", "elite_8", "final_four", "championship"):
            raw = bracket.get(rk)
            games = [raw] if isinstance(raw, dict) and ("winner" in raw or "loser" in raw) \
                    else (raw if isinstance(raw, list) else [])
            for game in games:
                if not isinstance(game, dict):
                    continue
                for role in ("winner", "loser"):
                    t = game.get(role, {})
                    if isinstance(t, dict) and t.get("name") == team_name:
                        return str(t.get("seed", "?"))
        return "?"

    with st.spinner("Loading 2026 model brackets…"):
        brackets = _cached_model_brackets_2026()

    if not brackets:
        st.error("Could not generate model brackets for 2026.")
        return

    # ── Main scores table ─────────────────────────────────────────────────
    rows = []
    for label, bracket in brackets.items():
        if not bracket:
            continue
        size_lbl   = _SIZE_LABEL_MAP.get(label, label)
        data       = _HARDCODED_2026.get(size_lbl, {})
        champ_name = data.get("champion_name", "—")
        champ_sd   = _find_seed(bracket, champ_name)
        rows.append({
            "Bracket Size":  size_lbl,
            "Score":         data.get("score", "—"),
            "Percentile":    data.get("percentile", "—"),
            "Champion Pick": f"#{champ_sd} {champ_name}",
        })

    st.subheader("2026 Bracket Scores")

    def _champ_cell_html(pick: str) -> str:
        if "Michigan" in pick:
            return f'<span style="color:#28a745;font-weight:700">{pick}</span>'
        if "Duke" in pick or "Illinois" in pick:
            return f'<span style="color:#dc3545;font-weight:700">{pick}</span>'
        return pick

    _score_tbl = (
        '<table style="width:100%;border-collapse:collapse;font-size:0.9rem;margin-bottom:1rem">'
        '<thead><tr style="border-bottom:2px solid #e0e6ef;background:#f8f9fc">'
        '<th style="text-align:left;padding:8px 14px;color:#555;font-weight:600">Bracket Size</th>'
        '<th style="text-align:right;padding:8px 14px;color:#555;font-weight:600">Score</th>'
        '<th style="text-align:right;padding:8px 14px;color:#555;font-weight:600">Percentile</th>'
        '<th style="text-align:left;padding:8px 14px;color:#555;font-weight:600">Champion Pick</th>'
        '</tr></thead><tbody>'
    )
    for _ri, _r in enumerate(rows):
        _bg = "#ffffff" if _ri % 2 == 0 else "#fafbfd"
        _score_tbl += (
            f'<tr style="border-bottom:1px solid #edf0f7;background:{_bg}">'
            f'<td style="padding:9px 14px;font-weight:600">{_r["Bracket Size"]}</td>'
            f'<td style="padding:9px 14px;text-align:right">{_r["Score"]}</td>'
            f'<td style="padding:9px 14px;text-align:right">{_r["Percentile"]}</td>'
            f'<td style="padding:9px 14px">{_champ_cell_html(_r["Champion Pick"])}</td>'
            f'</tr>'
        )
    _score_tbl += '</tbody></table>'
    st.markdown(_score_tbl, unsafe_allow_html=True)

    # ── Round-by-round breakdown — one expander per bracket size ─────────
    _ROUND_DISPLAY = [
        ("R64",          "Round of 64"),
        ("R32",          "Round of 32"),
        ("Sweet 16",     "Sweet 16"),
        ("Elite 8",      "Elite 8"),
        ("Final Four",   "Final Four"),
        ("Championship", "Championship"),
    ]

    st.subheader("Round-by-Round Breakdown")
    for label, bracket in brackets.items():
        if not bracket:
            continue
        size_lbl = _SIZE_LABEL_MAP.get(label, label)
        data     = _HARDCODED_2026.get(size_lbl, {})
        rs       = data.get("round_scores", {})
        total    = data.get("score", 0)
        with st.expander(size_lbl):
            exp_rows = [
                {
                    "Round":               display,
                    "Actual Score":        rs.get(key, 0),
                    "Possible Max Score":  320,
                }
                for key, display in _ROUND_DISPLAY
            ]
            exp_rows.append({
                "Round":              "Total",
                "Actual Score":       total,
                "Possible Max Score": 1920,
            })
            st.dataframe(
                pd.DataFrame(exp_rows),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Actual Score":       st.column_config.NumberColumn("Actual Score",       format="%d"),
                    "Possible Max Score": st.column_config.NumberColumn("Possible Max Score", format="%d"),
                },
            )


# ── Main results view (bracket tab only) ─────────────────────────────────────

def show_results(res: dict, selected_style: str) -> None:
    """Render champion header + bracket visual + portfolio + download."""
    all_types = res.get("all_types")
    internal  = STYLE_MAP[selected_style]
    meta      = STYLE_META[selected_style]
    color     = meta["color"]

    # ── Resolve the style-specific champion and bracket ───────────────────
    candidate = None
    style_bracket = res["base_bracket"]

    if all_types and internal in all_types:
        entry = all_types[internal]
        rec   = entry.get("recommendation")
        if rec and rec.primary:
            candidate = rec.primary
            style_bracket = build_champion_first_bracket(
                res["base_bracket"], candidate
            )

    # ── Apply manual advancement overrides ───────────────────────────────
    _active_overrides = st.session_state.get("manual_adv_overrides", [])
    if _active_overrides:
        style_bracket, _ovr_warns = rebuild_bracket_with_manual_overrides(
            style_bracket, _active_overrides
        )
        for _w in _ovr_warns:
            if "Conflict" in _w or "conflict" in _w:
                st.error(_w)
            else:
                st.warning(_w)
        _ovr_notice = "  ·  ".join(
            f"**{o['team']}** → {o['round']}" for o in _active_overrides
        )
        st.info(
            f"Manual overrides active ({len(_active_overrides)}): {_ovr_notice}",
            icon="⚙️",
        )

    # Stale-state guard
    if not style_bracket.get("bracket_halves"):
        st.warning("⚠️ Stale bracket data — please reload the page.")

    # ── Champion + style header ───────────────────────────────────────────
    # Always read champion from the (possibly override-mutated) bracket.
    # Never fall back to candidate.name, which reflects the pre-override pick.
    champ = style_bracket.get("champion", {})
    cname = champ.get("name", "?")
    cseed = champ.get("seed", "?")
    creg  = champ.get("region", "?")

    prob_parts: list[str] = []
    if candidate and not _active_overrides:
        prob_parts.append(f"{candidate.win_prob:.1%} title probability")
        if candidate.mc_ff_prob > 0:
            prob_parts.append(f"{candidate.mc_ff_prob:.1%} FF")
        if candidate.public_pct > 0:
            prob_parts.append(f"{candidate.public_pct:.1%} public picks")
    prob_txt = "  ·  ".join(prob_parts)

    st.markdown(
        f'<div style="padding:12px 18px; background:{color}18; border-left:4px solid {color}; '
        f'border-radius:6px; margin-bottom:12px; display:flex; align-items:center; gap:14px;">'
        f'<span style="font-size:2rem;">🏆</span>'
        f'<div>'
        f'<div style="font-size:0.7rem; color:{color}; font-weight:700; text-transform:uppercase; '
        f'letter-spacing:1px; margin-bottom:3px;">{meta["emoji"]} {selected_style} · {meta["best_for"]}</div>'
        f'<div style="font-size:1.1rem; font-weight:700; color:#fff;">{cname}'
        f'<span style="font-size:0.78rem; color:#888; font-weight:400; margin-left:10px;">#{cseed} · {creg}</span></div>'
        f'<div style="font-size:0.75rem; color:#777; margin-top:2px;">{prob_txt}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Bracket visual ────────────────────────────────────────────────────
    render_traditional_bracket(style_bracket, candidate, selected_style)

    portfolio = res.get("portfolio", [])
    if len(portfolio) > 1:
        with st.expander(f"📋 Portfolio ({len(portfolio)} brackets)", expanded=False):
            show_portfolio_tab(res)

    with st.expander("⬇️ Download", expanded=False):
        show_download_tab(res, style_bracket)


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_default_picks() -> dict[str, float]:
    """Load public_picks_2026.csv if it exists, else return empty dict."""
    if not DEFAULT_PICKS.exists():
        return {}
    try:
        pp_df    = pd.read_csv(DEFAULT_PICKS)
        name_col = next((c for c in _PICK_NAME_COLS if c in pp_df.columns), None)
        pct_col  = next((c for c in _PICK_PCT_COLS  if c in pp_df.columns), None)
        if not name_col or not pct_col:
            return {}
        out: dict[str, float] = {}
        for _, row in pp_df.iterrows():
            val = row.get(pct_col)
            if pd.notna(val) and float(val) > 0:
                out[str(row[name_col]).strip()] = float(val)
        return out
    except Exception:
        return {}


def _run_and_store(
    df:             pd.DataFrame,
    pool_size:      int,
    n_brackets:     int,
    sim_mode:       str,
    use_mc:         bool,
    n_sims:         int,
    file_picks:     dict[str, float],
    picks_override: dict[str, float],
    selected_style: str,
) -> None:
    """Run the pipeline and write results to session state."""
    res = run_pipeline(
        df             = df,
        pool_size      = pool_size,
        n_brackets     = n_brackets,
        sim_mode       = sim_mode,
        use_mc         = use_mc,
        n_sims         = n_sims,
        file_picks     = file_picks,
        picks_override = picks_override,
    )
    for _k in ["results", "selected_style", "run_ok"]:
        st.session_state.pop(_k, None)
    st.session_state["results"]        = res
    st.session_state["selected_style"] = selected_style
    st.session_state["run_ok"]         = True


_STYLE_TO_SIM_MODE = {
    "Conservative": "conservative",
    "Value":        "balanced",
    "Contrarian":   "upset_heavy",
}

_WATCHED_FILES = [
    PROJECT_ROOT / "data" / "future"     / "future_bracket_2026.csv",
    PROJECT_ROOT / "data" / "future"     / "public_picks_2026.csv",
    PROJECT_ROOT / "data" / "future"     / "espn_advancement_2026.csv",
    PROJECT_ROOT / "data" / "processed"  / "advancement_value_edges_2026.csv",
]

def _data_mtimes() -> tuple:
    return tuple(f.stat().st_mtime if f.exists() else 0 for f in _WATCHED_FILES)


def main() -> None:
    # Clear any stale cached data so mode changes always recompute.
    st.cache_data.clear()
    st.cache_resource.clear()

    # ── Global CSS ────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* ── Hide default Streamlit header padding so our hero has full control ── */
        header[data-testid="stHeader"] { background: transparent; }
        #root > div:first-child { padding-top: 0 !important; }

        /* ── Page background ── */
        .stApp {
            background: linear-gradient(160deg, #f0f2f8 0%, #eef1f9 50%, #f4f5fb 100%);
        }
        @media (prefers-color-scheme: dark) {
            .stApp { background: linear-gradient(160deg, #0d1120 0%, #101525 100%); }
        }

        /* ── Layout ── */
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        /* ── Hero header ── */
        .site-hero {
            background: linear-gradient(135deg, #0f1628 0%, #1a1a2e 60%, #16213e 100%);
            border-radius: 0 0 18px 18px;
            padding: 28px 32px 22px 32px;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.18);
        }
        .site-hero-title {
            font-size: 1.45rem;
            font-weight: 900;
            color: #ffffff;
            letter-spacing: -0.5px;
            line-height: 1.2;
            white-space: nowrap;
        }
        .site-hero-sub {
            font-size: 0.78rem;
            color: #8899cc;
            margin-top: 4px;
            font-weight: 500;
        }
        .substack-btn {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            background: #ff6719;
            color: #fff !important;
            text-decoration: none !important;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.3px;
            padding: 8px 16px;
            border-radius: 8px;
            transition: background 0.15s, transform 0.1s;
            white-space: nowrap;
            box-shadow: 0 2px 8px rgba(255,103,25,0.35);
        }
        .substack-btn:hover { background: #e55a10; transform: translateY(-1px); }

        /* ── Pick cards ── */
        .pick-card {
            background: #ffffff;
            border: 1px solid #e0e4ef;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 6px;
            box-shadow: 0 2px 8px rgba(26,26,46,0.06);
            transition: box-shadow 0.15s;
        }
        .pick-card:hover { box-shadow: 0 4px 16px rgba(26,26,46,0.12); }
        @media (prefers-color-scheme: dark) {
            .pick-card { background: #1e2235; border-color: #2e3450; }
        }

        /* ── Section card ── */
        .section-card {
            background: #fff;
            border: 1px solid #e8ecf6;
            border-radius: 14px;
            padding: 20px 22px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(26,26,46,0.05);
        }
        @media (prefers-color-scheme: dark) {
            .section-card { background: #1a1f35; border-color: #2a3050; }
        }

        /* ── Bracket container ── */
        .bracket-wrap {
            background: #fafbff;
            border: 1px solid #e0e4ef;
            border-radius: 14px;
            padding: 18px 14px;
            overflow-x: auto;
            box-shadow: 0 2px 10px rgba(26,26,46,0.06);
        }

        /* ── Tables ── */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(26,26,46,0.06);
        }

        /* ── Tab bar ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1px;
            border-bottom: 2px solid #d8dced;
            margin-bottom: 1.25rem;
            flex-wrap: nowrap;
            overflow-x: auto;
            padding-bottom: 0;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 7px 11px;
            border-radius: 8px 8px 0 0;
            background: transparent;
            transition: background 0.15s;
            white-space: nowrap;
            flex-shrink: 0;
        }
        .stTabs [data-baseweb="tab"]:hover { background: #f0f2fa; }
        .stTabs [aria-selected="true"] {
            background: #eef1fb !important;
        }

        /* ── Tab label text — strong selectors to override Streamlit defaults ── */
        div[data-testid="stTabs"] button[role="tab"] p {
            color: #111827 !important;
            font-weight: 800 !important;
            font-size: 0.87rem !important;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            color: #111827 !important;
            font-weight: 800 !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] p {
            color: #000000 !important;
            font-weight: 900 !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            border-bottom: 3px solid #111827 !important;
        }

        /* ── Expander ── */
        .streamlit-expanderHeader { font-size: 0.82rem; font-weight: 600; color: #444; }
        details summary { border-radius: 8px !important; }

        /* ── Buttons ── */
        .stButton > button {
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.82rem;
            transition: transform 0.1s, box-shadow 0.1s;
        }
        .stButton > button:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
        .stButton > button:active { transform: scale(0.98); }

        /* ── Download buttons ── */
        .stDownloadButton > button {
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.82rem;
            background: #f4f6ff;
            border: 1px solid #c5cce8;
            color: #1a1a2e;
            transition: background 0.15s;
        }
        .stDownloadButton > button:hover { background: #e4e9f8; }

        /* ── Alerts ── */
        .stAlert { border-radius: 12px; }

        /* ── Dividers ── */
        hr { border-color: #e8ecf6 !important; margin: 1rem 0 !important; }

        /* ── Headings ── */
        h2 { font-size: 1.15rem !important; font-weight: 800 !important; color: #1a1a2e !important; }
        h3 { font-size: 1rem !important; font-weight: 700 !important; color: #1a1a2e !important; }

        /* ── Coming soon ── */
        .coming-soon {
            text-align: center;
            padding: 80px 20px;
            color: #bbb;
            font-size: 1.2rem;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        /* ── Mobile ── */
        @media (max-width: 768px) {
            .site-hero { padding: 20px 16px 16px 16px; border-radius: 0 0 12px 12px; }
            .site-hero-title { font-size: 1.2rem; white-space: normal; }
            .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
            .stTabs [data-baseweb="tab"] { padding: 5px 8px; }
            div[data-testid="stTabs"] button[role="tab"] p { font-size: 0.72rem !important; }
            .section-card { padding: 14px 14px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Hero header (title + Substack link) ──────────────────────────────
    st.markdown(
        '<div class="site-hero">'
        '  <div>'
        '    <div class="site-hero-title">🏀 March Madness Bracket and Survivor Pool Predictor</div>'
        '  </div>'
        '  <a class="substack-btn" href="https://substack.com/@ecbk" target="_blank" rel="noopener">'
        '    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" '
        '         xmlns="http://www.w3.org/2000/svg">'
        '      <path d="M22.539 8.242H1.46V5.406h21.08v2.836zM1.46 10.812V24L12 18.11 22.54 24V10.812H1.46z"/>'
        '      <path d="M22.539 0H1.46v2.836h21.08V0z"/>'
        '    </svg>'
        '    Substack'
        '  </a>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Top-level tabs ────────────────────────────────────────────────────
    bracket_tab, odds_tab, value_tab, history_tab, survivor_tab, mock_tab, about_tab = st.tabs([
        "Bracket Predictions",
        "Odds & Analysis",
        "Top Value Plays",
        "Historical Bracket Results",
        "Optimal Survivor Path",
        "Mock Brackets",
        "About",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # BRACKET PREDICTIONS TAB
    # ══════════════════════════════════════════════════════════════════════
    with bracket_tab:
        st.caption(
            "Generate bracket recommendations based on pool size, team strength, "
            "public pick trends, and model simulations."
        )
        # ── Pool options ──────────────────────────────────────────────────
        _POOL_OPTIONS = [
            ("1–25",   25,  "Conservative"),
            ("26–100", 100, "Value"),
            ("100+",   500, "Contrarian"),
        ]
        pool_labels   = [o[0] for o in _POOL_OPTIONS]
        pool_defaults = [o[1] for o in _POOL_OPTIONS]
        pool_styles   = [o[2] for o in _POOL_OPTIONS]

        col_year, col_pool, col_meta = st.columns([1, 2, 3])

        with col_year:
            tournament_year = st.selectbox(
                "Tournament year",
                options=[2026, 2027],
                index=0,
            )

        with col_pool:
            pool_idx = st.radio(
                "Pool size",
                options=list(range(len(_POOL_OPTIONS))),
                index=1,
                format_func=lambda i: pool_labels[i],
                horizontal=True,
                label_visibility="visible",
            )

        pool_size      = pool_defaults[pool_idx]
        selected_style = pool_styles[pool_idx]
        sim_mode       = _STYLE_TO_SIM_MODE[selected_style]
        meta           = STYLE_META[selected_style]
        with col_meta:
            st.markdown(
                f'<div style="padding:10px 14px; background:linear-gradient(135deg,#f7f9ff,#eef1fb); '
                f'border-radius:10px; border-left:4px solid {meta["color"]}; margin-top:18px; '
                f'box-shadow:0 2px 8px rgba(26,26,46,0.06);">'
                f'<div style="font-size:0.68rem; color:{meta["color"]}; font-weight:700; '
                f'text-transform:uppercase; letter-spacing:1px;">'
                f'{meta["emoji"]} {selected_style} · {meta["best_for"]}</div>'
                f'<div style="font-size:0.74rem; color:#444; margin-top:3px; font-weight:500;">'
                f'{meta["tagline"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Hardcoded simulation settings (not exposed in UI) ────────────
        use_mc = _HAS_MC
        n_sims = 5_000

        # ── Advanced settings ─────────────────────────────────────────────
        with st.expander("⚙️ Advanced settings", expanded=False):
            n_brackets = st.number_input(
                "Portfolio size",
                min_value=1, max_value=20, value=1, step=1,
                help="Generate multiple brackets with different champion picks.",
            )

            st.divider()
            st.markdown("**Override data** *(optional — 2026 bracket pre-loaded by default)*")
            uploaded_csv = st.file_uploader(
                "Bracket CSV",
                type=["csv"],
                help=(
                    "Required columns: canonical_team_name, seed, region, "
                    "offensive_efficiency, defensive_efficiency, efficiency_margin."
                ),
            )
            picks_file = st.file_uploader(
                "Public picks CSV",
                type=["csv"],
                help="CSV with canonical_team_name + public_pick_pct columns.",
            )
            run_custom = st.button(
                "▶  Build from custom data",
                type="primary",
                use_container_width=True,
                disabled=(uploaded_csv is None),
            )

            st.divider()

            # ── Manual advancement overrides ──────────────────────────────
            st.markdown("**Manual Advancement Overrides** *(optional)*")
            st.caption(
                "Force a team to reach a target round. Applied on top of the model "
                "bracket. Click **Apply** to update. Later overrides supersede earlier "
                "ones if paths conflict."
            )

            _adv_round_opts = [
                "Round of 32", "Sweet 16", "Elite 8",
                "Final Four", "Championship Game", "Champion",
            ]
            _ovr_team_opts: list[str] = []
            if DEFAULT_CSV.exists():
                try:
                    _ovr_teams_df = pd.read_csv(DEFAULT_CSV)
                    _ovr_team_opts = sorted(
                        _ovr_teams_df["canonical_team_name"].dropna().tolist()
                    )
                except Exception:
                    pass

            _ovc1, _ovc2, _ovc3 = st.columns([2, 2, 1])
            with _ovc1:
                _sel_team = st.selectbox(
                    "Team",
                    ["— select —"] + _ovr_team_opts,
                    key="ovr_team_sel",
                    label_visibility="collapsed",
                )
            with _ovc2:
                _sel_rnd = st.selectbox(
                    "Round",
                    _adv_round_opts,
                    key="ovr_round_sel",
                    label_visibility="collapsed",
                )
            with _ovc3:
                _add_ovr_btn = st.button("Add", key="ovr_add_btn", use_container_width=True)

            if _add_ovr_btn and _sel_team and _sel_team != "— select —":
                _ovr_list: list = st.session_state.setdefault("manual_adv_overrides", [])
                _found_existing = False
                for _oe in _ovr_list:
                    if _oe["team"] == _sel_team:
                        _oe["round"] = _sel_rnd
                        _found_existing = True
                        break
                if not _found_existing:
                    _ovr_list.append({"team": _sel_team, "round": _sel_rnd})

            _ovr_current: list = st.session_state.get("manual_adv_overrides", [])
            if _ovr_current:
                st.markdown("**Active overrides:**")
                for _oi, _oe in enumerate(_ovr_current):
                    _orl, _orr = st.columns([5, 1])
                    with _orl:
                        st.caption(f"{_oe['team']} → {_oe['round']}")
                    with _orr:
                        if st.button("✕", key=f"rm_ovr_{_oi}", help="Remove override"):
                            _ovr_current.pop(_oi)
                            st.session_state["manual_adv_overrides"] = _ovr_current
                            st.rerun()

            _apb1, _apb2 = st.columns(2)
            with _apb1:
                if st.button(
                    "⚙️ Apply / Refresh Bracket",
                    key="apply_overrides_btn",
                    use_container_width=True,
                    type="primary",
                ):
                    st.cache_data.clear()
                    for _sk in list(st.session_state.keys()):
                        if any(x in _sk.lower() for x in
                               ("bracket", "result", "pipeline", "run_ok")):
                            del st.session_state[_sk]
                    st.rerun()
            with _apb2:
                if st.button(
                    "✕ Reset Overrides",
                    key="reset_overrides_btn",
                    use_container_width=True,
                ):
                    st.session_state.pop("manual_adv_overrides", None)
                    st.cache_data.clear()
                    for _sk in list(st.session_state.keys()):
                        if any(x in _sk.lower() for x in
                               ("bracket", "result", "pipeline", "run_ok")):
                            del st.session_state[_sk]
                    st.rerun()

        # ── 2027 placeholder ──────────────────────────────────────────────
        if tournament_year == 2027:
            st.info(
                "**2027 bracket data is not available yet.** Please use 2026.",
                icon="📅",
            )
            st.stop()

        # ── Auto-recompute when any input or data file changes ────────────
        _pipeline_key = (
            pool_size, selected_style, sim_mode,
            use_mc, n_sims, int(n_brackets),
            *_data_mtimes(),
        )
        if st.session_state.get("pipeline_key") != _pipeline_key:
            st.session_state.pop("run_ok",  None)
            st.session_state.pop("results", None)
            st.session_state["pipeline_key"] = _pipeline_key

        # ── Auto-run with default 2026 data ───────────────────────────────
        if _HAS_DEFAULT and not st.session_state.get("run_ok") and not run_custom:
            with st.spinner(f"Building {selected_style} bracket + Monte Carlo…"):
                try:
                    df_default = pd.read_csv(DEFAULT_CSV)
                    errs = validate_csv(df_default)
                    if errs:
                        st.error("Default CSV has validation errors: " + "; ".join(errs))
                    else:
                        _run_and_store(
                            df             = df_default,
                            pool_size      = int(pool_size),
                            n_brackets     = int(n_brackets),
                            sim_mode       = sim_mode,
                            use_mc         = use_mc and _HAS_MC,
                            n_sims         = int(n_sims),
                            file_picks     = _load_default_picks(),
                            picks_override = {},
                            selected_style = selected_style,
                        )
                except Exception as e:
                    st.error(f"Failed to load 2026 data: {e}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())

        # ── Run with custom uploaded data ─────────────────────────────────
        if run_custom and uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
            except Exception as e:
                st.error(f"Cannot read CSV: {e}")
                st.stop()

            errs = validate_csv(df)
            if errs:
                st.error("**CSV validation failed**")
                for e in errs:
                    st.markdown(f"- {e}")
                st.stop()

            file_picks     = parse_picks_file(picks_file) if picks_file else {}
            picks_override = {}

            label = f"Building {selected_style} bracket + Monte Carlo…"
            with st.spinner(label):
                try:
                    _run_and_store(
                        df             = df,
                        pool_size      = int(pool_size),
                        n_brackets     = int(n_brackets),
                        sim_mode       = sim_mode,
                        use_mc         = use_mc and _HAS_MC,
                        n_sims         = int(n_sims),
                        file_picks     = file_picks,
                        picks_override = picks_override,
                        selected_style = selected_style,
                    )
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())
                    st.session_state["run_ok"] = False

        # ── Render bracket results ─────────────────────────────────────────
        if st.session_state.get("run_ok") and "results" in st.session_state:
            _res = st.session_state["results"]
            _sty = st.session_state.get("selected_style", selected_style)

            show_results(_res, _sty)
        else:
            show_welcome()

    # ══════════════════════════════════════════════════════════════════════
    # ODDS & ANALYSIS TAB
    # ══════════════════════════════════════════════════════════════════════
    with odds_tab:
        st.caption(
            "View advancement probabilities, title chances, and team-level model odds."
        )
        _odds_year = st.radio(
            "Season",
            options=[2026, 2027],
            index=0,
            horizontal=True,
            key="odds_year_sel",
            label_visibility="collapsed",
        )
        if _odds_year == 2027:
            st.info(
                "2027 bracket data is not available yet. Please use 2026.",
                icon="📅",
            )
        elif st.session_state.get("run_ok") and "results" in st.session_state:
            show_odds_tab(st.session_state["results"])
        else:
            st.info("Generate a bracket in the **Bracket Predictions** tab first.")

    # ══════════════════════════════════════════════════════════════════════
    # TOP VALUE PLAYS TAB
    # ══════════════════════════════════════════════════════════════════════
    with value_tab:
        st.caption(
            "Highlights teams where the model sees more advancement upside than the public."
        )
        _value_year = st.radio(
            "Season",
            options=[2026, 2027],
            index=0,
            horizontal=True,
            key="value_year_sel",
            label_visibility="collapsed",
        )
        if _value_year == 2027:
            st.info(
                "2027 bracket data is not available yet. Please use 2026.",
                icon="📅",
            )
        else:
            _adv_csv = PROJECT_ROOT / "data" / "processed" / "advancement_value_edges_2026.csv"
            if _adv_csv.exists():
                try:
                    _adv_df = pd.read_csv(_adv_csv)
                    _adv_df = _adv_df[
                        (_adv_df["edge"].notna()) &
                        (_adv_df["model_pct"].notna()) &
                        (_adv_df["public_pct"].notna()) &
                        (_adv_df["edge"] > 0)
                    ].copy()

                    def _value_tier(edge: float) -> str:
                        if edge >= 0.15:
                            return "★★ Major"
                        if edge >= 0.08:
                            return "★ Strong"
                        return ""
                    _adv_df["Value"] = _adv_df["edge"].apply(_value_tier)

                    _adv_df = _adv_df.sort_values("edge", ascending=False)
                    _adv_disp = pd.DataFrame({
                        "Team":         _adv_df["team"].values,
                        "Advancing To": _adv_df["round"].values,
                        "Seed":    _adv_df["seed"].astype(int).values,
                        "Model %":  [f"{v:.1%}" for v in _adv_df["model_pct"]],
                        "Public %": [f"{v:.1%}" for v in _adv_df["public_pct"]],
                        "Edge":    [f"{v:+.1%}" for v in _adv_df["edge"]],
                        "Ratio":   [f"{v:.2f}x" for v in _adv_df["value_ratio"]],
                        "Value":   _adv_df["Value"].values,
                    })
                    st.dataframe(_adv_disp, hide_index=True, use_container_width=True)
                except Exception:
                    st.caption("Could not load advancement value data.")
            else:
                st.caption("No advancement value data available.")

    # ══════════════════════════════════════════════════════════════════════
    # HISTORICAL BRACKET RESULTS TAB
    # ══════════════════════════════════════════════════════════════════════
    with history_tab:
        show_historical_results_tab()

    # ══════════════════════════════════════════════════════════════════════
    # OPTIMAL SURVIVOR PATH TAB
    # ══════════════════════════════════════════════════════════════════════
    with survivor_tab:
        st.caption(
            "Coming soon: survivor-style pick recommendations using win probability, "
            "ownership, and future path value."
        )
        st.markdown(
            '<div class="coming-soon">🏗️ Coming Soon!</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # MOCK BRACKETS TAB
    # ══════════════════════════════════════════════════════════════════════
    with mock_tab:
        st.caption(
            "Coming soon: projected brackets during the season using current team "
            "ratings and projected seeds."
        )
        st.markdown(
            '<div class="coming-soon">🏗️ Coming Soon!</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # ABOUT TAB
    # ══════════════════════════════════════════════════════════════════════
    with about_tab:
        st.caption(
            "Learn how the model works, what data it uses, and how to interpret the outputs."
        )
        st.header("About This Model")
        st.write(
            "This bracket model combines pre-tournament team strength, historical tournament patterns, "
            "public pick data, and pool-size strategy to generate bracket recommendations."
        )

        st.subheader("Team Strength & Data Inputs")
        st.write(
            "The model draws on publicly available and subscriber-accessed college basketball "
            "analytics data — including KenPom, BartTorvik, and ESPN public bracket trends — "
            "to estimate team quality and matchup win probabilities. "
            "All model outputs and strategy logic are proprietary."
        )

        st.subheader("Public Pick Value")
        st.write(
            "The model compares its advancement probabilities against public bracket "
            "pick trends to identify teams that may be underpicked or overpicked."
        )

        st.subheader("Pool Size Strategy")
        st.write(
            "Small pools favor safer, higher-probability picks. Medium pools balance win "
            "probability and value. Large pools lean more into underpicked teams and "
            "higher-upside paths."
        )

        st.subheader("Upsets")
        st.write(
            "Upsets are evaluated using seed, model win probability, historical matchup "
            "rates, and public value. The model does not pick upsets randomly; it selects "
            "them when they are plausible and strategically useful."
        )

        st.subheader("Simulations")
        st.write(
            "For each bracket, the model runs 5,000 simulations to estimate advancement "
            "probabilities and title chances across all 64 teams."
        )

        st.subheader("Advanced Settings")
        st.write(
            "The Advanced Settings panel (found in the sidebar under ⚙️ Advanced settings) "
            "gives you additional control over how the bracket is generated and displayed."
        )
        adv_items = [
            (
                "Manual Advancement Overrides",
                "Force a specific team to advance to a chosen round — for example, "
                "Texas → Sweet 16 or Michigan → Champion. Overrides are layered on top "
                "of the model bracket and only take effect after clicking "
                "**Apply / Refresh Bracket**. Use **Reset Overrides** to remove all "
                "manual selections and return to the model's default output.",
            ),
            (
                "Apply / Refresh Bracket",
                "Reruns the bracket using the current settings and any active manual "
                "overrides. Also clears cached results so the latest model output is used.",
            ),
            (
                "Reset Overrides",
                "Clears all manual advancement overrides and regenerates the bracket "
                "using only the model's predictions.",
            ),
            (
                "Portfolio Size",
                "Controls how many bracket variants are generated. Useful if you are "
                "entering multiple brackets in the same pool.",
            ),
            (
                "Build from Custom Data",
                "Upload your own bracket CSV and optional public-picks CSV to run the "
                "model on a different year or a custom set of teams.",
            ),
        ]
        for title, desc in adv_items:
            st.markdown(f"**{title}** — {desc}")

        st.divider()
        st.subheader("Important Note")
        st.info(
            "This tool is designed to support bracket strategy, not guarantee results. "
            "March Madness outcomes are inherently unpredictable."
        )

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-top:3rem; padding:12px 0; border-top:1px solid #e8e8e8; '
        'text-align:center;">'
        '<span style="font-size:0.68rem; color:#aaa;">'
        'This tool is an independent bracket strategy model. '
        'It is not affiliated with or endorsed by the NCAA, ESPN, KenPom, BartTorvik, '
        'or any other data provider.'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
