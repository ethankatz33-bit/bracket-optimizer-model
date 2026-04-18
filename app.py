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

PROJECT_ROOT = Path(__file__).parent
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
    page_title="March Madness Pool Optimizer",
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
        "first_four":    first_four,
        "base_bracket":  base_bracket,
        "mc_results":    mc_results,
        "candidates":    candidates,
        "portfolio":     portfolio,
        "summary":       summary,
        "pool_rec":      pool_rec,
        "all_types":     all_types,
        "public_picks":  public_picks,
        "missing_picks": missing_picks,
        "orig_sum":      orig_sum,
        "norm_applied":  norm_applied,
        "picks_rows":    _build_picks_rows(mc_results, public_picks, df64),
        "pool_size":     pool_size,
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


def _bk_cell(w: dict, l: dict, champion_name: str, rc: str) -> str:
    """HTML for one game cell: winner (highlighted) + loser (struck through)."""
    wn, ws = w.get("name", "?"), w.get("seed", "?")
    ln, ls = l.get("name", "?"), l.get("seed", "?")
    hl = (wn == champion_name)
    H, G = _BH, _BG
    bd   = f"3px solid {rc}" if hl else f"2px solid #252535"
    bg   = "#16162e" if hl else "#0d0d1c"
    wc   = "#ffffff" if hl else "#cccccc"
    ww   = "700"     if hl else "400"
    return (
        f'<div style="height:{H}px; border-left:{bd}; background:{bg}; '
        f'padding:2px 5px 2px 6px; overflow:hidden; margin-bottom:{G}px;">'
        f'<div style="font-size:0.67rem; color:{wc}; font-weight:{ww}; '
        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
        f'<span style="font-size:0.58rem; color:{rc}; margin-right:3px;">#{ws}</span>'
        f'{wn}</div>'
        f'<div style="font-size:0.62rem; color:#333345; text-decoration:line-through; '
        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
        f'<span style="font-size:0.58rem; margin-right:3px;">#{ls}</span>{ln}</div>'
        f'</div>'
    )


def _bk_col(games: list, tops: list, champion_name: str, rc: str, total_h: int) -> str:
    """A round column: games absolutely positioned within a fixed-height container."""
    cells = "".join(
        f'<div style="position:absolute; top:{t}px; left:0; right:0;">'
        + _bk_cell(g["winner"], g["loser"], champion_name, rc)
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

    rc = REGION_COLORS.get(region, "#555")
    t64, t32, ts16, te8, total_h = _bk_tops()

    c64  = _bk_col(g64,  t64,  champion_name, rc, total_h)
    c32  = _bk_col(g32,  t32,  champion_name, rc, total_h)
    cs16 = _bk_col(gs16, ts16, champion_name, rc, total_h)
    ce8  = _bk_col(ge8,  te8,  champion_name, rc, total_h)

    def lbl(text: str, col: str) -> str:
        return (
            f'<div style="margin-right:{_BCG}px; flex-shrink:0;">'
            f'<div style="font-size:0.5rem; color:#555; text-align:center; '
            f'margin-bottom:3px; font-weight:600; letter-spacing:1px; '
            f'text-transform:uppercase;">{text}</div>'
            f'{col}</div>'
        )

    r_align = "left" if side == "left" else "right"
    header  = (
        f'<div style="font-size:0.57rem; color:{rc}; font-weight:700; '
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
        bg = "#16162e" if hl else "#0d0d1c"
        bd = f"2px solid {color}" if hl else "1px solid #252535"
        wc = "#ffffff" if hl else "#cccccc"
        ww = "700" if hl else "400"
        regions = f'{w.get("region","?")} · {l.get("region","?")}'
        return (
            f'<div style="background:{bg}; border:{bd}; border-radius:5px; '
            f'padding:4px 6px; width:106px; flex-shrink:0;">'
            f'<div style="font-size:0.48rem; color:#555; margin-bottom:2px; '
            f'white-space:nowrap;">{label} · {regions}</div>'
            f'<div style="font-size:0.67rem; color:{wc}; font-weight:{ww}; '
            f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
            f'<span style="font-size:0.58rem; color:{color}; margin-right:2px;">#{ws}</span>{wn}</div>'
            f'<div style="font-size:0.62rem; color:#333345; text-decoration:line-through; '
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
            f'<div style="background:linear-gradient(160deg,{color}25,#0d0d1c); '
            f'border:2px solid {color}; border-radius:7px; padding:7px 9px; '
            f'width:120px; flex-shrink:0; text-align:center;">'
            f'<div style="font-size:0.5rem; color:{color}; font-weight:700; '
            f'letter-spacing:1px; margin-bottom:3px;">CHAMPIONSHIP</div>'
            f'<div style="font-size:0.72rem; color:#fff; font-weight:700; '
            f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
            f'<span style="font-size:0.6rem; color:{color}; margin-right:3px;">#{ws}</span>{wn}</div>'
            f'<div style="font-size:0.62rem; color:#444; text-decoration:line-through; '
            f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
            f'<span style="font-size:0.6rem; margin-right:3px;">#{ls}</span>{ln}</div>'
            f'<div style="font-size:0.55rem; color:{color}; margin-top:4px; font-weight:700;">'
            f'🏆 CHAMPION</div>'
            f'</div>'
        )

    def arrow(ch: str) -> str:
        return (
            f'<div style="color:#333; font-size:0.9rem; padding:0 3px; '
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

    Left  half : East (top)  + West (bottom)   — rounds advancing →
    Center     : [FF SF1] → [CHAMP] ← [FF SF2] — all at bracket midpoint
    Right half : South (top) + Midwest (bottom) — rounds advancing ←
    """
    champion_name = bracket.get("champion", {}).get("name", "?")
    color = STYLE_META.get(style, {}).get("color", "#4A90D9")

    left   = _bk_region(bracket, "East",    champion_name, "left") \
           + _bk_region(bracket, "West",    champion_name, "left")
    center = _bk_center(bracket, champion_name, color)
    right  = _bk_region(bracket, "South",   champion_name, "right") \
           + _bk_region(bracket, "Midwest", champion_name, "right")

    html = (
        f'<div style="overflow-x:auto; background:#09091a; border-radius:10px; '
        f'padding:16px 12px; margin-bottom:8px;">'
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


# ── Welcome screen (no results yet) ──────────────────────────────────────────

def show_welcome() -> None:
    st.markdown("### How it works")
    st.markdown(
        "Upload your bracket CSV, tell the optimizer how big your pool is, "
        "pick a style, and get a complete bracket built to **win your pool** — "
        "not just predict the tournament."
    )
    st.markdown("---")
    cols = st.columns(3)
    for i, (style, meta) in enumerate(STYLE_META.items()):
        with cols[i]:
            st.markdown(
                f"""
                <div style="border-left:4px solid {meta['color']}; padding:12px 16px;
                            background:#12121f; border-radius:6px;">
                  <div style="font-size:1.3rem;">{meta['emoji']} <strong>{style}</strong></div>
                  <div style="font-size:0.85rem; color:#ccc; margin:6px 0;">
                    {meta['tagline']}
                  </div>
                  <div style="font-size:0.75rem; color:#777;">
                    Best for: {meta['best_for']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("")
    st.info(
        "👈 Upload your **bracket CSV** in the sidebar to get started. "
        "You can use `data/future/future_bracket_2026.csv` as a sample."
    )


# ── Odds & Analysis tab ───────────────────────────────────────────────────────

def show_odds_tab(res: dict) -> None:
    mc  = res.get("mc_results")
    candidates = res.get("candidates", [])

    if mc:
        st.subheader(f"Simulated probabilities &nbsp;·&nbsp; {mc.n_sims:,} runs")
        st.caption(
            "These probabilities come from running the full 64-team bracket thousands "
            "of times. Teams that win more simulations have stronger title cases."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Most likely to win the title**")
            rows = [
                {"Team": r.name, "Seed": r.seed, "Region": r.region,
                 "Title chance": f"{r.title_prob:.1%}",
                 "Final Four %": f"{r.ff_prob:.1%}"}
                for r in mc.top_by_title(10)
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**Most likely to reach Final Four**")
            rows = [
                {"Team": r.name, "Seed": r.seed, "Region": r.region,
                 "Final Four %": f"{r.ff_prob:.1%}",
                 "Title chance": f"{r.title_prob:.1%}"}
                for r in mc.top_by_ff(10)
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.markdown("---")

    if candidates:
        st.subheader("Value analysis — champion candidates")
        st.caption(
            "**Value score** = title probability ÷ public pick share. "
            "A score above 1.0 means the model thinks a team is more likely to win "
            "than the public does. The higher, the better the leverage play."
        )
        rows = [
            {
                "Team":         c.name,
                "Seed":         c.seed,
                "Region":       c.region,
                "Title chance": f"{c.win_prob:.1%}",
                "Public pick":  f"{c.public_pct:.2%}",
                "Value score":  f"{c.value_score:.2f}×",
                "Final Four %": f"{c.mc_ff_prob:.1%}" if c.mc_ff_prob else "—",
            }
            for c in candidates
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if not mc and not candidates:
        st.info(
            "Run with **Monte Carlo** enabled (sidebar → Advanced) "
            "to see simulated title probabilities."
        )

    # Pick share table
    picks_rows = res.get("picks_rows", [])
    if picks_rows:
        with st.expander("Full pick-share breakdown (all teams)", expanded=False):
            st.caption(
                "How the public is distributing their champion picks. "
                "Teams with low public % but real title probability are prime Value/Contrarian targets."
            )
            has_mc = any(r["title_prob"] > 0 for r in picks_rows)
            rows = sorted(picks_rows, key=lambda r: r["public_pct"], reverse=True)
            display = []
            for r in rows[:30]:
                d = {"Team": r["name"], "Seed": r["seed"], "Region": r["region"],
                     "Public pick %": f"{r['public_pct']:.2%}"}
                if has_mc:
                    d["Title chance"] = f"{r['title_prob']:.1%}" if r["title_prob"] > 0 else "—"
                    d["Value"]        = f"{r['value_score']:.2f}×" if r["value_score"] > 0 else "—"
                display.append(d)
            st.dataframe(pd.DataFrame(display), hide_index=True, use_container_width=True)

        # Missing picks note
        missing = res.get("missing_picks", [])
        if missing:
            with st.expander(f"{len(missing)} team(s) using seed-average pick %"):
                st.dataframe(pd.DataFrame(missing), hide_index=True, use_container_width=True)


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


# ── Main results view ─────────────────────────────────────────────────────────

def show_results(res: dict, selected_style: str) -> None:
    all_types = res.get("all_types")
    internal  = STYLE_MAP[selected_style]
    meta      = STYLE_META[selected_style]
    color     = meta["color"]

    # ── Resolve the style-specific champion and bracket ───────────────────
    candidate = None
    style_bracket = res["base_bracket"]   # fallback

    if all_types and internal in all_types:
        entry = all_types[internal]
        rec   = entry.get("recommendation")
        if rec and rec.primary:
            candidate = rec.primary
            style_bracket = build_champion_first_bracket(
                res["base_bracket"], candidate
            )

    # ── Style banner ──────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="padding:14px 18px; background:{color}1a; border-left:4px solid {color};
                    border-radius:6px; margin-bottom:20px;">
          <div style="font-size:0.8rem; color:{color}; font-weight:700;
                      text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">
            {meta['emoji']} {selected_style} Strategy &nbsp;·&nbsp; {meta['best_for']}
          </div>
          <div style="font-size:0.92rem; color:#ddd; line-height:1.5;">
            {meta['pitch']}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Coverage quick-stats ──────────────────────────────────────────────
    n_miss = len(res.get("missing_picks", []))
    total  = len(res["public_picks"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Pool size", f"{res['pool_size']:,} people")
    c2.metric("Public pick data", f"{total - n_miss}/{total} teams",
              help="Teams matched to real public pick data vs seed-average fallback")
    if res.get("mc_results"):
        top = res["mc_results"].top_by_title(1)
        if top:
            c3.metric("Model's top pick",
                      f"{top[0].name}",
                      f"{top[0].title_prob:.1%} title chance",
                      help="Team with highest simulated title probability")

    if res.get("norm_applied"):
        st.caption(f"Pick data normalized from {res['orig_sum']:.1%} → 100%")

    st.divider()

    # ── Main tabs ─────────────────────────────────────────────────────────
    tab_bracket, tab_odds, tab_portfolio, tab_download = st.tabs([
        "🏆  Your Bracket",
        "📊  Odds & Analysis",
        "📋  Portfolio",
        "📥  Download",
    ])

    with tab_bracket:
        # ── Champion summary strip ────────────────────────────────────────
        champ = style_bracket.get("champion", {})
        cname = champ.get("name", "?")
        cseed = champ.get("seed", "?")
        creg  = champ.get("region", "?")
        prob_txt = (
            f" &nbsp;·&nbsp; {candidate.win_prob:.1%} title probability"
            + (f" &nbsp;·&nbsp; {candidate.mc_ff_prob:.1%} FF prob" if candidate and candidate.mc_ff_prob > 0 else "")
            if candidate else ""
        )
        st.markdown(
            f'<div style="padding:10px 16px; background:{color}18; '
            f'border-left:4px solid {color}; border-radius:6px; margin-bottom:10px; '
            f'display:flex; align-items:center; gap:12px;">'
            f'<span style="font-size:1.4rem;">🏆</span>'
            f'<div>'
            f'<span style="font-size:1.05rem; font-weight:700; color:#fff;">{cname}</span>'
            f'<span style="font-size:0.8rem; color:#999; margin-left:8px;">'
            f'#{cseed} seed &nbsp;·&nbsp; {creg} region{prob_txt}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # ── Traditional bracket visual ────────────────────────────────────
        render_traditional_bracket(style_bracket, candidate, selected_style)

        # ── Upsets & detail ───────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.75rem; color:#666; margin:6px 0 2px;">'
            'Key upset picks</div>',
            unsafe_allow_html=True,
        )
        render_upset_picks(style_bracket)
        render_full_rounds_expander(style_bracket)

        # First Four (if any)
        ff = res.get("first_four", [])
        if ff:
            with st.expander(f"First Four play-in games ({len(ff)})", expanded=False):
                rows = [{"Region": g["region"], "Seed": g["seed"],
                         "Advances": g["winner"], "Eliminated": g["loser"]}
                        for g in ff]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with tab_odds:
        show_odds_tab(res)

    with tab_portfolio:
        show_portfolio_tab(res)

    with tab_download:
        show_download_tab(res, style_bracket)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🏀 March Madness Pool Optimizer")
    st.caption("Build a bracket designed to **win your pool** — not just predict the tournament.")

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Build Your Bracket")

        uploaded_csv = st.file_uploader(
            "Bracket CSV  *(required)*",
            type=["csv"],
            help=(
                "Required columns: canonical_team_name, seed, region, "
                "offensive_efficiency, defensive_efficiency, efficiency_margin."
            ),
        )

        st.divider()

        # Pool size — drives the auto-recommendation
        pool_size = st.number_input(
            "How many people are in your pool?",
            min_value=2, max_value=100_000, value=100, step=10,
        )

        # Auto-suggest style from pool size
        auto_tier = classify_pool(int(pool_size))
        auto_map  = {"small_pool": "Conservative", "medium_pool": "Value",
                     "large_pool": "Contrarian",   "mega_pool":   "Contrarian"}
        suggested = auto_map.get(auto_tier, "Value")

        st.markdown(
            f'<div style="font-size:0.75rem; color:#888; margin:-8px 0 8px;">'
            f'Suggested for your pool size: <b style="color:#ccc">{suggested}</b></div>',
            unsafe_allow_html=True,
        )

        selected_style = st.radio(
            "Pick a bracket style",
            options=list(STYLE_MAP.keys()),
            index=list(STYLE_MAP.keys()).index(suggested),
            format_func=lambda s: f"{STYLE_META[s]['emoji']} {s}  —  {STYLE_META[s]['tagline']}",
            help="This determines your champion pick strategy.",
        )

        n_brackets = st.number_input(
            "Number of brackets to generate",
            min_value=1, max_value=20, value=1, step=1,
        )

        st.divider()

        with st.expander("⚙️ Advanced settings"):
            sim_mode = st.selectbox(
                "Upset frequency",
                ["balanced", "conservative", "upset_heavy"],
                index=0,
                format_func=lambda m: {
                    "conservative":  "Low — stick to the chalk",
                    "balanced":      "Medium — realistic mix",
                    "upset_heavy":   "High — embrace chaos",
                }[m],
            )
            use_mc = st.checkbox(
                "Run Monte Carlo simulations",
                value=True,
                disabled=not _HAS_MC,
                help="Simulates the tournament thousands of times to estimate realistic title probabilities." if _HAS_MC
                     else "lib/monte_carlo.py not found.",
            )
            n_sims = st.number_input(
                "Simulations",
                min_value=500, max_value=50_000, value=5_000, step=500,
                disabled=not (use_mc and _HAS_MC),
            )
            picks_file = st.file_uploader(
                "Public picks CSV  *(optional)*",
                type=["csv"],
                help="CSV with canonical_team_name + public_pick_pct columns.",
            )
            manual_picks = st.text_input(
                "Manual pick % overrides  *(optional)*",
                placeholder='Duke=0.22, Kansas=0.15',
                help="Comma-separated Name=fraction pairs. Overrides everything else.",
            )

        st.divider()

        run_button = st.button(
            "▶  Build My Bracket",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_csv is None),
        )

        if uploaded_csv is None:
            st.caption("Upload a CSV above to get started.")

    # ── Run pipeline on button click ──────────────────────────────────────
    if run_button and uploaded_csv is not None:
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
            st.markdown(
                "**Required:** " +
                ", ".join(f"`{c}`" for c in sorted(REQUIRED_COLS))
            )
            st.stop()

        file_picks     = parse_picks_file(picks_file) if picks_file else {}
        picks_override = _parse_picks(manual_picks)   if manual_picks else {}

        label = f"Building your {selected_style} bracket" + (" + running Monte Carlo…" if use_mc else "…")
        with st.spinner(label):
            try:
                res = run_pipeline(
                    df             = df,
                    pool_size      = int(pool_size),
                    n_brackets     = int(n_brackets),
                    sim_mode       = sim_mode,
                    use_mc         = use_mc and _HAS_MC,
                    n_sims         = int(n_sims),
                    file_picks     = file_picks,
                    picks_override = picks_override,
                )
                st.session_state["results"]       = res
                st.session_state["selected_style"] = selected_style
                st.session_state["run_ok"]         = True
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())
                st.session_state["run_ok"] = False

    # ── Display ───────────────────────────────────────────────────────────
    if st.session_state.get("run_ok") and "results" in st.session_state:
        show_results(
            st.session_state["results"],
            st.session_state.get("selected_style", selected_style),
        )
    else:
        show_welcome()


if __name__ == "__main__":
    main()
