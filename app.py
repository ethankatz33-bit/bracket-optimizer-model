"""
app.py — March Madness Pool Optimizer
Streamlit MVP that wraps the existing prediction engine and strategy outputs.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import io
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Core library imports ──────────────────────────────────────────────────────
from lib.team_selector import simulate_bracket
from lib.bracket_strategy import (
    extract_candidates,
    generate_portfolio,
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

# ── Pipeline helpers (internal functions from predict_future_bracket) ─────────
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

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_REGIONS = {"East", "West", "South", "Midwest"}

_TYPE_COLOR = {
    "deterministic": "#4A90D9",
    "safe":          "#27AE60",
    "value":         "#E67E22",
    "contrarian":    "#8E44AD",
}
_TYPE_EMOJI = {
    "deterministic": "🎯",
    "safe":          "🛡️",
    "value":         "📈",
    "contrarian":    "🎲",
}

# ── Validation (Streamlit-safe — returns errors list, never sys.exit) ─────────

def validate_csv(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: **{', '.join(sorted(missing))}**")
    if "region" in df.columns:
        bad = set(df["region"].dropna().unique()) - VALID_REGIONS
        if bad:
            errors.append(f"Invalid region values: **{', '.join(sorted(bad))}**  "
                          f"(valid: East, West, South, Midwest)")
    n = len(df)
    if n not in (64, 68) and not errors:
        errors.append(f"Expected 64 or 68 teams, found **{n}**. "
                      "Check for duplicate or missing rows.")
    return errors


def parse_picks_file(uploaded) -> dict[str, float]:
    """Read an uploaded public picks CSV into a {name: float} dict."""
    try:
        picks_df = pd.read_csv(uploaded)
    except Exception as e:
        st.warning(f"Could not read public picks file: {e}")
        return {}
    name_col = next((c for c in _PICK_NAME_COLS if c in picks_df.columns), None)
    pct_col  = next((c for c in _PICK_PCT_COLS  if c in picks_df.columns), None)
    if not name_col or not pct_col:
        st.warning(
            f"Public picks file needs a name column "
            f"({' / '.join(_PICK_NAME_COLS)}) and a pick % column "
            f"({' / '.join(_PICK_PCT_COLS)})."
        )
        return {}
    out: dict[str, float] = {}
    for _, row in picks_df.iterrows():
        val = row.get(pct_col)
        if pd.notna(val):
            v = float(val)
            if v > 0:
                out[str(row[name_col]).strip()] = v
    return out


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    df:             pd.DataFrame,
    pool_size:      int,
    n_brackets:     int,
    mode:           str,
    use_mc:         bool,
    n_sims:         int,
    file_picks:     dict[str, float],
    picks_override: dict[str, float],
) -> dict:
    """
    Run the full prediction pipeline.  Returns a results dict — never calls
    sys.exit; raises RuntimeError on fatal problems.
    """
    # 1. First Four
    df64, first_four = _simulate_first_four(df)

    # 2. Build team overrides (ratings, CPS, etc.)
    teams_override = _build_teams_override(df64)

    # 3. Merge public picks
    public_picks, missing_picks = _build_public_picks(df64, file_picks, picks_override)

    # 4. Normalize
    public_picks, orig_sum, norm_applied = _normalize_public_picks(public_picks)

    # 5. Deterministic bracket simulation
    bracket = simulate_bracket(mode, _teams_override=teams_override)

    # 6. Monte Carlo (optional)
    mc_results = None
    if use_mc and _HAS_MC:
        mc_results = run_monte_carlo(
            teams_override=teams_override,
            n_sims=n_sims,
        )

    # 7. Champion candidates
    candidates = []
    if mc_results is not None or n_brackets > 0:
        candidates = extract_candidates(bracket, public_picks or None, mc_results)

    # 8. Portfolio
    portfolio: list = []
    if n_brackets > 0 and candidates:
        portfolio = generate_portfolio(
            base_bracket=bracket,
            n=n_brackets,
            pool_size=pool_size,
            public_picks=public_picks or None,
            mc_results=mc_results,
        )

    # 9. Strategy summary + bracket types
    summary   = _build_strategy_summary(bracket, mc_results, candidates)
    pool_rec  = build_recommendation(candidates, pool_size) if candidates else None
    det_champ = summary.get("mc_champion") or summary.get("deterministic_champion") or {}
    all_types = build_all_bracket_types(candidates, det_champ) if candidates else None

    # 10. Pick rows for analysis table
    picks_rows = _build_picks_rows(mc_results, public_picks, df64)

    return {
        "first_four":    first_four,
        "bracket":       bracket,
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
        "picks_rows":    picks_rows,
        "pool_size":     pool_size,
    }


# ── Display helpers ───────────────────────────────────────────────────────────

def _pct(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}%}"


def _val(v: float | None) -> str:
    if v is None or v == 0:
        return "—"
    return f"{v:.2f}×"


def show_type_card(btype: str, entry: dict) -> None:
    """Render one bracket-type recommendation card."""
    color = _TYPE_COLOR.get(btype, "#555")
    emoji = _TYPE_EMOJI.get(btype, "")
    label = entry["label"]
    arch  = entry["archetype"]
    desc  = entry["description"]

    if btype == "deterministic":
        champ  = entry.get("champion") or {}
        name   = champ.get("name",   "—")
        seed   = champ.get("seed")
        region = champ.get("region", "")
        title  = champ.get("title_prob")
        pub    = None
        vs     = None
    else:
        rec   = entry["recommendation"]
        c     = rec.primary
        name   = c.name   if c else "—"
        seed   = c.seed   if c else None
        region = c.region if c else ""
        title  = c.win_prob    if c else None
        pub    = c.public_pct  if c else None
        vs     = c.value_score if c else None

    seed_str = f"#{seed} " if seed is not None else ""

    st.markdown(
        f"""
        <div style="border-left: 4px solid {color}; padding: 10px 16px;
                    background: #1a1a2e; border-radius: 4px; margin-bottom: 8px;">
          <div style="font-size: 0.75rem; color: {color}; font-weight: 600;
                      text-transform: uppercase; letter-spacing: 0.05em;">
            {emoji} {label} — {arch}
          </div>
          <div style="font-size: 1.4rem; font-weight: 700; margin: 4px 0 2px;">
            {name}
            <span style="font-size: 0.9rem; color: #aaa;">({seed_str}{region})</span>
          </div>
          <div style="font-size: 0.85rem; color: #ccc; display: flex; gap: 16px; margin-bottom: 6px;">
            {"<span>Title: <b>" + _pct(title) + "</b></span>" if title else ""}
            {"<span>Public: <b>" + _pct(pub) + "</b></span>" if pub is not None else ""}
            {"<span>Value: <b>" + _val(vs) + "</b></span>" if vs is not None else ""}
          </div>
          <div style="font-size: 0.8rem; color: #999; font-style: italic;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_mc_tables(mc_results) -> None:
    top_title = mc_results.top_by_title(10)
    top_ff    = mc_results.top_by_ff(10)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 — Title Probability")
        rows = [
            {
                "Team":     r.name,
                "Seed":     r.seed,
                "Region":   r.region,
                "Title %":  f"{r.title_prob:.1%}",
                "FF %":     f"{r.ff_prob:.1%}",
                "E8 %":     f"{r.e8_prob:.1%}",
            }
            for r in top_title
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Top 10 — Final Four Probability")
        rows = [
            {
                "Team":     r.name,
                "Seed":     r.seed,
                "Region":   r.region,
                "FF %":     f"{r.ff_prob:.1%}",
                "Title %":  f"{r.title_prob:.1%}",
            }
            for r in top_ff
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def show_picks_table(picks_rows: list[dict]) -> None:
    has_mc = any(r["title_prob"] > 0 for r in picks_rows)
    rows = sorted(picks_rows, key=lambda r: r["title_prob"], reverse=True) if has_mc \
        else sorted(picks_rows, key=lambda r: r["public_pct"], reverse=True)

    display = []
    for r in rows[:20]:
        d = {
            "Team":    r["name"],
            "Seed":    r["seed"],
            "Region":  r["region"],
            "Public %": f"{r['public_pct']:.2%}",
        }
        if has_mc:
            d["Title %"] = f"{r['title_prob']:.1%}"
            d["FF %"]    = f"{r['ff_prob']:.1%}"
            d["Value"]   = f"{r['value_score']:.2f}×" if r["value_score"] > 0 else "—"
        display.append(d)

    st.dataframe(pd.DataFrame(display), hide_index=True, use_container_width=True)


def show_portfolio(portfolio: list, pool_size: int) -> None:
    if not portfolio:
        st.info("No portfolio brackets generated. Set **Number of brackets** > 0 and re-run.")
        return
    rows = []
    for e in portfolio:
        c = e.champion
        rows.append({
            "#":          e.index,
            "Champion":   c.name,
            "Seed":       c.seed,
            "Region":     c.region,
            "Title %":    f"{c.win_prob:.1%}",
            "FF %":       f"{c.mc_ff_prob:.1%}" if c.mc_ff_prob > 0 else "—",
            "Public %":   f"{c.public_pct:.2%}",
            "Value":      f"{c.value_score:.2f}×",
            "Style":      e.rationale.split(".")[0] if e.rationale else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with st.expander("Full rationale for each bracket"):
        for e in portfolio:
            st.markdown(f"**Bracket {e.index} — {e.champion.name}**")
            st.caption(e.rationale)
            if e.ev_note:
                st.caption(e.ev_note)
            st.divider()


def show_first_four(first_four: list) -> None:
    if not first_four:
        return
    with st.expander(f"First Four results ({len(first_four)} play-in games)", expanded=False):
        rows = [
            {
                "Region": g["region"],
                "Seed":   g["seed"],
                "Winner": f"{g['winner']} (EM {g['winner_em']:+.1f})",
                "Eliminated": f"{g['loser']} (EM {g['loser_em']:+.1f})",
            }
            for g in first_four
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def build_json_output(res: dict) -> dict:
    """Build the same JSON structure as predict_future_bracket.py."""
    bracket = res["bracket"]
    champ   = bracket.get("champion", {})
    ff      = bracket.get("final_four", [])
    cg      = bracket.get("championship") or {}

    out: dict = {
        "pool_size":  res["pool_size"],
        "first_four": res["first_four"],
        "champion": {
            "name":   champ.get("name"),
            "seed":   champ.get("seed"),
            "region": champ.get("region"),
        },
        "final_four": [
            {
                "winner":      g.get("winner", {}).get("name"),
                "winner_seed": g.get("winner", {}).get("seed"),
                "loser":       g.get("loser",  {}).get("name"),
                "loser_seed":  g.get("loser",  {}).get("seed"),
            }
            for g in ff
        ],
        "championship": {
            "winner":      cg.get("winner", {}).get("name"),
            "winner_seed": cg.get("winner", {}).get("seed"),
            "loser":       cg.get("loser",  {}).get("name"),
            "loser_seed":  cg.get("loser",  {}).get("seed"),
        },
    }

    if res.get("summary"):
        out["strategy_summary"] = res["summary"]

    if res.get("pool_rec"):
        out["pool_recommendation"] = res["pool_rec"].to_dict()

    if res.get("mc_results"):
        out["monte_carlo"] = res["mc_results"].to_dict()

    if res.get("portfolio"):
        out["portfolio"] = [
            {
                "index":       e.index,
                "champion":    e.champion.name,
                "seed":        e.champion.seed,
                "title_prob":  e.champion.win_prob,
                "public_pct":  e.champion.public_pct,
                "value_score": e.champion.value_score,
                "rationale":   e.rationale,
            }
            for e in res["portfolio"]
        ]

    for btype in ("deterministic", "safe", "value", "contrarian"):
        key = f"{btype}_recommendation"
        if res.get("all_types") and btype in res["all_types"]:
            entry = res["all_types"][btype]
            if btype == "deterministic":
                out[key] = {
                    "bracket_type": btype,
                    "champion":     entry.get("champion"),
                    "description":  entry.get("description"),
                }
            else:
                rec = entry["recommendation"]
                c   = rec.primary
                out[key] = {
                    "bracket_type":  btype,
                    "pool_category": rec.tier,
                    "description":   entry.get("description"),
                    "champion":      {
                        "name":        c.name,
                        "seed":        c.seed,
                        "title_prob":  round(c.win_prob,    4),
                        "public_pct":  round(c.public_pct,  4),
                        "value_score": round(c.value_score, 3),
                    } if c else None,
                    "n_brackets":    rec.n_brackets,
                }

    return out


# ════════════════════════════════════════════════════════════════════════════
# Main app
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Title ─────────────────────────────────────────────────────────────
    st.title("🏀 March Madness Pool Optimizer")
    st.caption(
        "Upload your bracket CSV, set your pool size, and let the model pick "
        "the champion strategy that maximizes your expected winnings."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        uploaded_csv = st.file_uploader(
            "Bracket / team stats CSV (required)",
            type=["csv"],
            help=(
                "Must include: season, canonical_team_name, seed, region, "
                "offensive_efficiency, defensive_efficiency, efficiency_margin. "
                "Optionally include public_pick_pct."
            ),
        )

        st.divider()

        pool_size = st.number_input(
            "Pool size (# of entrants)",
            min_value=2, max_value=100_000, value=100, step=10,
        )
        n_brackets = st.number_input(
            "Number of brackets to generate",
            min_value=0, max_value=20, value=1, step=1,
            help="0 = strategy output only, no portfolio. 1–20 = full portfolio.",
        )
        mode = st.selectbox(
            "Bracket mode",
            ["balanced", "conservative", "upset_heavy"],
            index=0,
        )

        st.divider()

        use_mc = st.checkbox(
            "Run Monte Carlo simulations",
            value=True,
            disabled=not _HAS_MC,
            help="Monte Carlo improves title probability estimates significantly." if _HAS_MC
                 else "lib/monte_carlo.py not found.",
        )
        n_sims = st.number_input(
            "Simulations",
            min_value=500, max_value=50_000, value=5_000, step=500,
            disabled=not use_mc,
        )

        st.divider()

        picks_file = st.file_uploader(
            "Public picks CSV (optional)",
            type=["csv"],
            help=(
                "CSV with columns: canonical_team_name (or team_name / name) + "
                "public_pick_pct (or pick_pct / champion_pick_pct / pct). "
                "Overrides any public_pick_pct column in the bracket CSV."
            ),
        )
        manual_picks = st.text_input(
            "Manual pick % overrides (optional)",
            placeholder='e.g.  Duke=0.22, Kansas=0.15',
            help="Comma-separated Name=fraction pairs. Highest priority.",
        )

        st.divider()

        run_button = st.button(
            "▶  Run Prediction",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_csv is None),
        )

        if uploaded_csv is None:
            st.info("Upload a bracket CSV to get started.")

    # ── Run pipeline ──────────────────────────────────────────────────────
    if run_button and uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Cannot read CSV: {e}")
            st.stop()

        # Validate
        errs = validate_csv(df)
        if errs:
            st.error("**CSV validation failed:**")
            for e in errs:
                st.markdown(f"- {e}")
            st.markdown(
                "**Required columns:** " +
                ", ".join(f"`{c}`" for c in sorted(REQUIRED_COLS))
            )
            st.stop()

        # Parse inputs
        file_picks     = parse_picks_file(picks_file) if picks_file else {}
        picks_override = _parse_picks(manual_picks)   if manual_picks else {}

        # Run
        with st.spinner(
            f"Running {'Monte Carlo + ' if use_mc else ''}prediction "
            f"({n_sims:,} sims)..." if use_mc else "Running prediction..."
        ):
            try:
                res = run_pipeline(
                    df             = df,
                    pool_size      = int(pool_size),
                    n_brackets     = int(n_brackets),
                    mode           = mode,
                    use_mc         = use_mc and _HAS_MC,
                    n_sims         = int(n_sims),
                    file_picks     = file_picks,
                    picks_override = picks_override,
                )
                st.session_state["results"] = res
                st.session_state["run_ok"]  = True
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                with st.expander("Full traceback"):
                    st.code(traceback.format_exc())
                st.session_state["run_ok"] = False

    # ── Display results ───────────────────────────────────────────────────
    if st.session_state.get("run_ok") and "results" in st.session_state:
        res = st.session_state["results"]
        _show_results(res)


def _show_results(res: dict) -> None:
    all_types  = res.get("all_types")
    mc_results = res.get("mc_results")
    pool_rec   = res.get("pool_rec")
    first_four = res.get("first_four", [])

    show_first_four(first_four)

    # ── Coverage banner ───────────────────────────────────────────────────
    n_missing = len(res.get("missing_picks", []))
    total     = len(res["public_picks"])
    norm      = res.get("norm_applied", False)
    orig_sum  = res.get("orig_sum", 0.0)

    cols = st.columns(3)
    cols[0].metric("Pick % coverage", f"{total - n_missing}/{total} teams",
                   help="Teams with real public pick data vs seed-default fallback")
    cols[1].metric("Pool size tier", classify_pool(res["pool_size"]).replace("_", " ").title())
    if mc_results:
        top = mc_results.top_by_title(1)
        if top:
            cols[2].metric("MC top champion",
                           f"{top[0].name} ({top[0].title_prob:.1%})",
                           help="Team with highest simulated title probability")
    if norm:
        st.caption(f"Pick % normalization applied: {orig_sum:.1%} → 100.0%")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_labels = ["🎯 Strategy", "📊 Monte Carlo", "📋 Portfolio", "📥 Download"]
    tabs = st.tabs(tab_labels)

    # ── Tab A: Strategy Summary ───────────────────────────────────────────
    with tabs[0]:
        st.header("Strategy Recommendations")
        st.caption(
            "Four bracket types, each optimized for a different pool structure. "
            "Pick the one that matches how you want to play."
        )

        if all_types:
            col1, col2 = st.columns(2)
            type_order = ["deterministic", "safe", "value", "contrarian"]
            for i, btype in enumerate(type_order):
                entry = all_types[btype]
                with (col1 if i % 2 == 0 else col2):
                    show_type_card(btype, entry)

        # Pool-specific recommendation
        if pool_rec and pool_rec.primary:
            st.divider()
            c      = pool_rec.primary
            btype  = pool_rec.bracket_type
            color  = _TYPE_COLOR.get(btype, "#4A90D9")
            emoji  = _TYPE_EMOJI.get(btype, "")
            label  = BRACKET_TYPES.get(btype, {}).get("label", btype.title())
            st.markdown(
                f"""
                <div style="border: 2px solid {color}; padding: 14px 18px;
                            border-radius: 6px; margin-top: 8px;">
                  <div style="font-size: 0.8rem; color: {color}; font-weight: 700;
                               text-transform: uppercase;">
                    {emoji} Your pool ({res['pool_size']:,} entrants) → {label} strategy
                  </div>
                  <div style="font-size: 1.5rem; font-weight: 700; margin: 6px 0 2px;">
                    {c.name}
                    <span style="font-size: 0.95rem; color: #aaa;">
                      (#{c.seed} {c.region})
                    </span>
                  </div>
                  <div style="font-size: 0.85rem; color: #ccc;">
                    Title: <b>{_pct(c.win_prob)}</b> &nbsp;|&nbsp;
                    Public: <b>{_pct(c.public_pct)}</b> &nbsp;|&nbsp;
                    Value: <b>{_val(c.value_score)}</b>
                    {"&nbsp;|&nbsp; FF: <b>" + _pct(c.mc_ff_prob) + "</b>" if c.mc_ff_prob > 0 else ""}
                  </div>
                  <div style="font-size: 0.8rem; color: #aaa; margin-top: 8px;">
                    {pool_rec.primary_reason}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Recommended portfolio size: **{pool_rec.n_brackets}** bracket(s)")

        # Picks analysis table
        if res.get("picks_rows"):
            with st.expander("Pick share & value analysis (top 20 teams)", expanded=False):
                show_picks_table(res["picks_rows"])

        # Missing picks
        if res.get("missing_picks"):
            with st.expander(f"⚠️ {n_missing} team(s) using seed-default pick %"):
                miss_df = pd.DataFrame(res["missing_picks"])
                st.dataframe(miss_df, hide_index=True, use_container_width=True)

    # ── Tab B: Monte Carlo ────────────────────────────────────────────────
    with tabs[1]:
        if mc_results:
            st.header(f"Monte Carlo Results — {mc_results.n_sims:,} simulations")
            show_mc_tables(mc_results)

            with st.expander("Full candidate list", expanded=False):
                cands = res.get("candidates", [])
                if cands:
                    rows = [
                        {
                            "Team":    c.name,
                            "Seed":    c.seed,
                            "Region":  c.region,
                            "Title %": f"{c.win_prob:.1%}",
                            "FF %":    f"{c.mc_ff_prob:.1%}" if c.mc_ff_prob else "—",
                            "Public %":f"{c.public_pct:.2%}",
                            "Value":   f"{c.value_score:.2f}×",
                            "In FF?":  "✓ FF" if c.in_base_ff else ("✓ E8" if c.in_base_e8 else "—"),
                        }
                        for c in cands
                    ]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info(
                "Monte Carlo was not run.  "
                "Enable **Run Monte Carlo simulations** in the sidebar and re-run "
                "for title probability estimates."
                if _HAS_MC else
                "Monte Carlo module (lib/monte_carlo.py) not found."
            )

    # ── Tab C: Portfolio ──────────────────────────────────────────────────
    with tabs[2]:
        st.header("Portfolio Brackets")
        show_portfolio(res.get("portfolio", []), res["pool_size"])

        # Base bracket champion path
        bracket = res["bracket"]
        champ   = bracket.get("champion", {})
        ff      = bracket.get("final_four", [])
        cg      = bracket.get("championship") or {}

        with st.expander("Base bracket (deterministic path)", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Champion:** {champ.get('name','?')} "
                            f"(#{champ.get('seed','?')} {champ.get('region','?')})")
                st.markdown("**Final Four:**")
                for g in ff:
                    w = g.get("winner", {})
                    l = g.get("loser",  {})
                    st.caption(f"#{w.get('seed','?')} {w.get('name','?')} "
                               f"def. #{l.get('seed','?')} {l.get('name','?')}")
            with c2:
                w = cg.get("winner", {})
                l = cg.get("loser",  {})
                st.markdown("**Championship:**")
                st.caption(f"#{w.get('seed','?')} {w.get('name','?')} "
                           f"def. #{l.get('seed','?')} {l.get('name','?')}")

    # ── Tab D: Download ───────────────────────────────────────────────────
    with tabs[3]:
        st.header("Download Results")

        # Build JSON
        json_output = build_json_output(res)
        json_bytes  = json.dumps(json_output, indent=2).encode("utf-8")

        st.download_button(
            label="⬇️  Download full results (JSON)",
            data=json_bytes,
            file_name="march_madness_results.json",
            mime="application/json",
        )

        # Champion candidates as CSV
        cands = res.get("candidates", [])
        if cands:
            cand_df = pd.DataFrame([
                {
                    "name":        c.name,
                    "seed":        c.seed,
                    "region":      c.region,
                    "title_prob":  round(c.win_prob,    4),
                    "ff_prob":     round(c.mc_ff_prob,  4),
                    "public_pct":  round(c.public_pct,  4),
                    "value_score": round(c.value_score, 3),
                }
                for c in cands
            ])
            st.download_button(
                label="⬇️  Download champion candidates (CSV)",
                data=cand_df.to_csv(index=False).encode("utf-8"),
                file_name="champion_candidates.csv",
                mime="text/csv",
            )

        if res.get("portfolio"):
            port_df = pd.DataFrame([
                {
                    "bracket":     e.index,
                    "champion":    e.champion.name,
                    "seed":        e.champion.seed,
                    "region":      e.champion.region,
                    "title_prob":  round(e.champion.win_prob, 4),
                    "public_pct":  round(e.champion.public_pct, 4),
                    "value_score": round(e.champion.value_score, 3),
                    "rationale":   e.rationale,
                }
                for e in res["portfolio"]
            ])
            st.download_button(
                label="⬇️  Download portfolio (CSV)",
                data=port_df.to_csv(index=False).encode("utf-8"),
                file_name="portfolio.csv",
                mime="text/csv",
            )

        st.divider()
        st.markdown("**JSON structure:**")
        st.json(
            {k: ("..." if isinstance(v, dict) else v)
             for k, v in json_output.items()
             if k not in ("monte_carlo", "bracket")},
            expanded=False,
        )


if __name__ == "__main__":
    main()
