"""
scripts/analyze_advancement_value.py
Advancement-level value edge: model probability vs ESPN public advancement %.

For each team × round, computes:
    edge        = model_pct - public_pct
    value_ratio = model_pct / public_pct

Usage
-----
  python3 scripts/analyze_advancement_value.py
  python3 scripts/analyze_advancement_value.py --sims 20000 --top 15

Output
------
  Console: per-round top-edge tables
  CSV:     data/processed/advancement_value_edges_2026.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_future_bracket import (
    _simulate_first_four,
    _build_teams_override,
    _load_public_picks_file,
    _build_public_picks,
    _normalize_public_picks,
)
from lib.monte_carlo import run_monte_carlo

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_BRACKET_CSV    = PROJECT_ROOT / "data" / "future" / "future_bracket_2026.csv"
DEFAULT_ESPN_CSV       = PROJECT_ROOT / "data" / "future" / "espn_advancement_2026.csv"
DEFAULT_PICKS_CSV      = PROJECT_ROOT / "data" / "future" / "public_picks_2026.csv"
OUTPUT_CSV             = PROJECT_ROOT / "data" / "processed" / "advancement_value_edges_2026.csv"

# Ordered round keys matching MCTeamResult fields and ESPN CSV columns
ROUNDS: list[tuple[str, str]] = [
    ("R32",        "round_of_32"),   # won R64
    ("Sweet 16",   "sweet_16"),      # won R32
    ("Elite 8",    "elite_8"),       # won S16
    ("Final Four", "final_four"),    # won E8
    ("Champ Game", "champ_game"),    # won FF
    ("Champion",   "title"),         # won championship
]

# Minimum edge magnitude to flag as a noteworthy value play
EDGE_THRESHOLDS: dict[str, float] = {
    "R32":        0.05,
    "Sweet 16":   0.05,
    "Elite 8":    0.04,
    "Final Four": 0.04,
    "Champ Game": 0.03,
    "Champion":   0.02,
}

W = 88


# ── ESPN advancement loader ───────────────────────────────────────────────────

def _load_espn_advancement(path: Path) -> dict[str, dict[str, float]]:
    """
    Returns {team_name: {round_label: pct}}.
    Skips comment lines and blank cells.
    """
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path, comment="#")
    except Exception as exc:
        print(f"  WARNING: could not read ESPN CSV: {exc}")
        return out

    round_cols = [r for r, _ in ROUNDS]
    for _, row in df.iterrows():
        team = str(row.get("team", "")).strip()
        if not team or team.startswith("#"):
            continue
        team_data: dict[str, float] = {}
        for col in round_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip() not in ("", "nan"):
                try:
                    team_data[col] = float(val)
                except (ValueError, TypeError):
                    pass
        if team_data:
            out[team] = team_data
    return out


# ── Core comparison ───────────────────────────────────────────────────────────

def _build_edge_rows(
    mc_results,
    espn_data: dict[str, dict[str, float]],
    df64: pd.DataFrame,
) -> list[dict]:
    """
    For each team × round, compute model_pct, public_pct, edge, value_ratio.
    public_pct = ESPN value if available, else None (omitted from output).
    """
    # Build MC lookup: {team_name: MCTeamResult}
    mc_lookup: dict[str, object] = {r.name: r for r in mc_results.results}

    # Map MCTeamResult field → round label
    mc_field_map: dict[str, str] = {
        "R32":        "round_of_32",
        "Sweet 16":   "sweet_16",
        "Elite 8":    "elite_8",
        "Final Four": "final_four",
        "Champ Game": "champ_game",
        "Champion":   "title",
    }

    rows: list[dict] = []
    teams_in_field = df64["canonical_team_name"].tolist()

    for team in teams_in_field:
        mc_r = mc_lookup.get(team)
        espn_team = espn_data.get(team, {})
        seed_val = int(df64.loc[df64["canonical_team_name"] == team, "seed"].iloc[0])

        for round_label, mc_attr in ROUNDS:
            # Model probability from MC
            if mc_r is not None:
                model_pct = getattr(mc_r, f"{mc_field_map[round_label].replace(' ', '_')}_prob", 0.0) \
                            if hasattr(mc_r, f"{mc_field_map[round_label].replace(' ', '_')}_prob") \
                            else _mc_round_prob(mc_r, round_label)
            else:
                model_pct = 0.0

            # ESPN public pct (may be absent)
            public_pct = espn_team.get(round_label)
            has_espn   = public_pct is not None

            edge        = (model_pct - public_pct) if has_espn else None
            value_ratio = (model_pct / public_pct) if (has_espn and public_pct > 0) else None
            threshold   = EDGE_THRESHOLDS.get(round_label, 0.03)
            is_value    = (edge is not None and edge >= threshold)

            rows.append({
                "team":        team,
                "seed":        seed_val,
                "round":       round_label,
                "model_pct":   round(model_pct, 4),
                "public_pct":  round(public_pct, 4) if has_espn else None,
                "edge":        round(edge, 4) if edge is not None else None,
                "value_ratio": round(value_ratio, 3) if value_ratio is not None else None,
                "is_value_play": is_value,
                "pub_source":  "espn" if has_espn else "missing",
            })

    return rows


def _mc_round_prob(mc_r, round_label: str) -> float:
    """Map round label to the correct MCTeamResult probability method."""
    mapping = {
        "R32":        mc_r.r32_prob,
        "Sweet 16":   mc_r.s16_prob,
        "Elite 8":    mc_r.e8_prob,
        "Final Four": mc_r.ff_prob,
        "Champ Game": mc_r.champ_game_prob,
        "Champion":   mc_r.title_prob,
    }
    return mapping.get(round_label, 0.0)


# ── Printing ──────────────────────────────────────────────────────────────────

def _print_tables(rows: list[dict], top_n: int, espn_loaded: bool) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print("No results to display.")
        return

    print("=" * W)
    print("  ADVANCEMENT VALUE EDGE ANALYSIS — 2026 NCAA Tournament")
    if espn_loaded:
        print("  model_advancement_pct (Monte Carlo) vs ESPN People's Bracket")
    else:
        print("  Monte Carlo advancement probabilities  [ESPN data not yet loaded]")
    print("=" * W)
    print()

    # If no ESPN data at all, just print the MC advancement table
    if not espn_loaded:
        mc_only = df[df["round"].isin(["R32", "Sweet 16", "Elite 8", "Final Four", "Champ Game", "Champion"])]
        pivot = mc_only.pivot(index="team", columns="round", values="model_pct").reset_index()
        # Order columns
        col_order = ["team"] + [r for r, _ in ROUNDS if r in pivot.columns]
        pivot = pivot[[c for c in col_order if c in pivot.columns]]
        pivot = pivot.sort_values("Champion" if "Champion" in pivot.columns else pivot.columns[-1], ascending=False)
        print("  MONTE CARLO ADVANCEMENT PROBABILITIES (all teams, sorted by title%)")
        print()
        header = f"  {'Team':<24}" + "".join(f"  {c:>11}" for c in col_order[1:])
        print(header)
        print("  " + "─" * (W - 2))
        for _, row in pivot.head(20).iterrows():
            line = f"  {row['team']:<24}"
            for c in col_order[1:]:
                v = row.get(c)
                line += f"  {v:>10.1%}" if pd.notna(v) else f"  {'—':>10}"
            print(line)
        print()
        print(f"  To add ESPN advancement data, fill in:")
        print(f"  {DEFAULT_ESPN_CSV}")
        print()
        return

    # Per-round top-edge tables
    for round_label, _ in ROUNDS:
        sub = df[(df["round"] == round_label) & df["edge"].notna()].copy()
        if sub.empty:
            continue
        threshold = EDGE_THRESHOLDS.get(round_label, 0.03)
        top = sub.nlargest(min(top_n, len(sub)), "edge")

        print(f"  ── {round_label.upper():<14} (value threshold: +{threshold:.0%})  {'─' * (W - 38)}")
        hdr = f"  {'Team':<24} {'Seed':>4}  {'Model':>7}  {'ESPN':>7}  {'Edge':>7}  {'Ratio':>6}  {'Tag'}"
        print(hdr)
        print("  " + "─" * (W - 2))
        for _, r in top.iterrows():
            tag = "★ VALUE" if r["is_value_play"] else ""
            ratio_str = f"{r['value_ratio']:.2f}x" if r["value_ratio"] is not None else "—"
            print(
                f"  {r['team']:<24} {r['seed']:>4}  "
                f"{r['model_pct']:>6.1%}  {r['public_pct']:>6.1%}  "
                f"{r['edge']:>+6.1%}  {ratio_str:>6}  {tag}"
            )
        print()

    # Summary: top value plays overall
    value_plays = df[df["is_value_play"] == True].sort_values("edge", ascending=False)
    if not value_plays.empty:
        print("=" * W)
        print(f"  TOP VALUE PLAYS — {len(value_plays)} total across all rounds")
        print("=" * W)
        hdr2 = f"  {'Team':<24} {'Round':<13} {'Seed':>4}  {'Model':>7}  {'ESPN':>7}  {'Edge':>7}  {'Ratio':>6}"
        print(hdr2)
        print("  " + "─" * (W - 2))
        for _, r in value_plays.head(25).iterrows():
            ratio_str = f"{r['value_ratio']:.2f}x" if r["value_ratio"] is not None else "—"
            print(
                f"  {r['team']:<24} {r['round']:<13} {r['seed']:>4}  "
                f"{r['model_pct']:>6.1%}  {r['public_pct']:>6.1%}  "
                f"{r['edge']:>+6.1%}  {ratio_str:>6}"
            )
        print()

    # Coverage
    covered = df[df["pub_source"] == "espn"]["team"].nunique()
    total   = df["team"].nunique()
    print(f"  ESPN coverage: {covered}/{total} teams have at least one round populated")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Advancement value analysis for 2026 bracket")
    parser.add_argument("--bracket", default=str(DEFAULT_BRACKET_CSV), metavar="FILE")
    parser.add_argument("--espn",    default=str(DEFAULT_ESPN_CSV),    metavar="FILE")
    parser.add_argument("--picks",   default=str(DEFAULT_PICKS_CSV),   metavar="FILE")
    parser.add_argument("--sims",    type=int, default=10000,
                        help="Monte Carlo simulations (default: 10000)")
    parser.add_argument("--top",     type=int, default=10,
                        help="Top N edges per round (default: 10)")
    args = parser.parse_args()

    # ── Load bracket ──────────────────────────────────────────────────────────
    print()
    df = pd.read_csv(args.bracket)
    df64, _ = _simulate_first_four(df)
    teams_override = _build_teams_override(df64)

    # ── Run Monte Carlo ───────────────────────────────────────────────────────
    print(f"  Running Monte Carlo ({args.sims:,} simulations)…", end=" ", flush=True)
    mc_results = run_monte_carlo(teams_override=teams_override, n_sims=args.sims)
    print("done")
    print()

    # ── Load ESPN advancement data ────────────────────────────────────────────
    espn_data  = _load_espn_advancement(Path(args.espn))
    espn_teams = len(espn_data)
    espn_cells = sum(len(v) for v in espn_data.values())
    print(f"  ESPN advancement entries: {espn_teams} teams, {espn_cells} round cells populated")
    if espn_teams == 0:
        print(f"  (no ESPN data — showing model-only advancement table)")
        print(f"  Fill in: {args.espn}")
    print()

    # ── Build edge rows ───────────────────────────────────────────────────────
    rows = _build_edge_rows(mc_results, espn_data, df64)

    # ── Print ─────────────────────────────────────────────────────────────────
    _print_tables(rows, args.top, espn_loaded=(espn_teams > 0))

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved → {OUTPUT_CSV}")
    print()


if __name__ == "__main__":
    main()
