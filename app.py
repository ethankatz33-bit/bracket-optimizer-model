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
    page_title="March Madness Bracket Predictor",
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
        + ff_cell(ff0, "SF1") + arrow("→") + champ_cell() + arrow("←") + ff_cell(ff1, "SF2")
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

    # Stale-state guard
    if not style_bracket.get("bracket_halves"):
        st.warning("⚠️ Stale bracket data — please reload the page.")

    # ── Champion + style header ───────────────────────────────────────────
    champ = style_bracket.get("champion", {})
    cname = champ.get("name", "?")
    cseed = champ.get("seed", "?")
    creg  = champ.get("region", "?")

    prob_parts: list[str] = []
    if candidate:
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
            font-size: 1.65rem;
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
            gap: 2px;
            border-bottom: 2px solid #e0e4ef;
            margin-bottom: 1.25rem;
            flex-wrap: wrap;
            padding-bottom: 0;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.80rem;
            font-weight: 600;
            letter-spacing: 0.4px;
            padding: 8px 18px;
            border-radius: 8px 8px 0 0;
            color: #888;
            background: transparent;
            transition: color 0.15s, background 0.15s;
        }
        .stTabs [data-baseweb="tab"]:hover { color: #1a1a2e; background: #f0f2fa; }
        .stTabs [aria-selected="true"] {
            color: #1a1a2e !important;
            background: #eef1fb !important;
            border-bottom: 2px solid #1a1a2e !important;
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
            .stTabs [data-baseweb="tab"] { padding: 6px 10px; font-size: 0.72rem; }
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
        '    <div class="site-hero-title">🏀 March Madness Bracket Predictor</div>'
        '    <div class="site-hero-sub">AI-powered pool strategy · 2026 Tournament</div>'
        '  </div>'
        '  <a class="substack-btn" href="https://substack.com/@ecbk" target="_blank" rel="noopener">'
        '    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" '
        '         xmlns="http://www.w3.org/2000/svg">'
        '      <path d="M22.539 8.242H1.46V5.406h21.08v2.836zM1.46 10.812V24L12 18.11 22.54 24V10.812H1.46z"/>'
        '      <path d="M22.539 0H1.46v2.836h21.08V0z"/>'
        '    </svg>'
        '    Read on Substack'
        '  </a>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Top-level tabs ────────────────────────────────────────────────────
    bracket_tab, odds_tab, value_tab, survivor_tab, mock_tab, about_tab = st.tabs([
        "Bracket Predictions",
        "Odds & Analysis",
        "Value Plays by Round",
        "Optimal Survivor Path",
        "Mock Brackets",
        "About",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # BRACKET PREDICTIONS TAB
    # ══════════════════════════════════════════════════════════════════════
    with bracket_tab:
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
            manual_picks = st.text_input(
                "Manual pick % overrides",
                placeholder="Duke=0.22, Kansas=0.15",
                help="Comma-separated Name=fraction pairs. Overrides everything else.",
            )
            run_custom = st.button(
                "▶  Build from custom data",
                type="primary",
                use_container_width=True,
                disabled=(uploaded_csv is None),
            )

        st.divider()

        # ── Refresh button ────────────────────────────────────────────────
        if st.button("🔄 Refresh Bracket"):
            st.cache_data.clear()
            for key in list(st.session_state.keys()):
                if "bracket" in key.lower() or "result" in key.lower() or "pipeline" in key.lower():
                    del st.session_state[key]
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
            picks_override = _parse_picks(manual_picks)   if manual_picks else {}

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

            # ── Recommended pick cards ────────────────────────────────────
            _all_types = _res.get("all_types")
            if _all_types:
                _card_cols = st.columns(3)
                for _ci, (_lbl, _internal) in enumerate([
                    ("Conservative", "safe"),
                    ("Value",        "value"),
                    ("Contrarian",   "contrarian"),
                ]):
                    _entry = _all_types.get(_internal)
                    _rec   = _entry.get("recommendation") if _entry else None
                    _prim  = _rec.primary if _rec else None
                    _m     = STYLE_META[_lbl]
                    _active = (_sty == _lbl)
                    _border = f"2px solid {_m['color']}" if _active else "1px solid #e0e4ef"
                    _bg     = f"{_m['color']}0f" if _active else "#fff"
                    with _card_cols[_ci]:
                        if _prim:
                            st.markdown(
                                f'<div class="pick-card" style="border:{_border}; background:{_bg};">'
                                f'<div style="font-size:0.65rem; font-weight:700; '
                                f'color:{_m["color"]}; text-transform:uppercase; letter-spacing:1px; '
                                f'margin-bottom:4px;">{_m["emoji"]} {_lbl} · {_m["best_for"]}</div>'
                                f'<div style="font-size:1rem; font-weight:800; color:#1a1a2e; '
                                f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
                                f'#{_prim.seed} {_prim.name}</div>'
                                f'<div style="font-size:0.7rem; color:#888; margin-top:3px;">'
                                f'{_prim.win_prob:.1%} title chance · {_prim.public_pct:.1%} public</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="pick-card" style="border:{_border}; background:{_bg};">'
                                f'<div style="font-size:0.65rem; font-weight:700; '
                                f'color:{_m["color"]}; text-transform:uppercase; letter-spacing:1px; '
                                f'margin-bottom:4px;">{_m["emoji"]} {_lbl}</div>'
                                f'<div style="font-size:0.82rem; color:#aaa;">—</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

            show_results(_res, _sty)
        else:
            show_welcome()

    # ══════════════════════════════════════════════════════════════════════
    # ODDS & ANALYSIS TAB
    # ══════════════════════════════════════════════════════════════════════
    with odds_tab:
        if st.session_state.get("run_ok") and "results" in st.session_state:
            show_odds_tab(st.session_state["results"])
        else:
            st.info("Generate a bracket in the **Bracket Predictions** tab first.")

    # ══════════════════════════════════════════════════════════════════════
    # VALUE PLAYS BY ROUND TAB
    # ══════════════════════════════════════════════════════════════════════
    with value_tab:
        st.caption(
            "Value edge compares this model's advancement probability to public bracket "
            "pick trends. Positive edge means the team is underpicked by the public."
        )
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
                    "Team":    _adv_df["team"].values,
                    "Round":   _adv_df["round"].values,
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
    # OPTIMAL SURVIVOR PATH TAB
    # ══════════════════════════════════════════════════════════════════════
    with survivor_tab:
        st.markdown(
            '<div class="coming-soon">🏗️ Coming Soon!</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # MOCK BRACKETS TAB
    # ══════════════════════════════════════════════════════════════════════
    with mock_tab:
        st.markdown(
            '<div class="coming-soon">🏗️ Coming Soon!</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # ABOUT TAB
    # ══════════════════════════════════════════════════════════════════════
    with about_tab:
        st.header("About This Model")
        st.write(
            "This bracket model combines team strength, historical tournament patterns, "
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
