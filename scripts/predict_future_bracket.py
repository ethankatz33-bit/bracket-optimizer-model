"""
scripts/predict_future_bracket.py
Predict a bracket for a future season from a team stats CSV.

Does NOT require Kaggle team_id. Uses canonical_team_name as identity.
Does NOT modify the stable prediction model. Feeds pre-computed stats
into simulate_bracket() via the _teams_override mechanism.

Usage
-----
  python3 scripts/predict_future_bracket.py --input data/future/future_bracket_template.csv

Options
-------
  --input FILE    Path to team stats CSV (required)
  --season YEAR   Tournament year for display/output naming (e.g. 2026)
  --pool N        Estimated pool size for portfolio strategy (default: 100)
  --n N           Portfolio brackets to generate; 0 = single bracket only (default: 0)
  --mode MODE     conservative|balanced|upset_heavy (default: balanced)
  --picks K=V,…   Override public pick %% as comma-separated name=fraction pairs
                  Example: --picks "Duke=0.22,Kansas=0.18"
  --output FILE   Output JSON path (default: data/processed/future_bracket_{season}.json)
  --monte-carlo   Run Monte Carlo simulations for advancement probability estimates
  --sims N        Number of Monte Carlo simulations (default: 10000)
  --public-picks-file FILE
                  CSV with public pick %% by team (canonical_team_name + public_pick_pct).
                  Overrides CSV column; lower priority than --picks.

Input CSV schema
----------------
  Required: season, team_name_raw, canonical_team_name, seed, region,
            offensive_efficiency, defensive_efficiency, efficiency_margin
  Optional: kenpom_rank, bart_torvik_rank, ap_top12_flag, public_pick_pct,
            conference, recent_form, strength_of_schedule,
            team_rating, champion_profile_score

Output
------
  Console: champion, Final Four, Championship game, upset summary
  JSON:    data/processed/future_bracket_{season}.json  (or --output path)

Examples
--------
  # Single best bracket for a 50-person office pool
  python3 scripts/predict_future_bracket.py \\
      --input data/future/future_bracket_2026.csv --season 2026 --pool 50

  # 5-bracket portfolio for a large ESPN pool
  python3 scripts/predict_future_bracket.py \\
      --input data/future/future_bracket_2026.csv --season 2026 --pool 10000 --n 5

  # With real public pick data overrides
  python3 scripts/predict_future_bracket.py \\
      --input data/future/future_bracket_2026.csv --season 2026 \\
      --pool 200 --n 6 --picks "Duke=0.22,Kansas=0.18"

  # Monte Carlo mode: 10,000 sims + 3-bracket portfolio
  python3 scripts/predict_future_bracket.py \\
      --input data/future/future_bracket_2026.csv --season 2026 \\
      --pool 200 --n 3 --monte-carlo --sims 10000
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.team_selector import simulate_bracket
from lib.bracket_strategy import (
    extract_candidates,
    generate_portfolio,
    format_portfolio,
    DEFAULT_PUBLIC_PCT,
)
try:
    from lib.monte_carlo import run_monte_carlo, format_mc_summary
    _HAS_MC = True
except ImportError:
    _HAS_MC = False

from lib.pool_strategy import (
    build_recommendation,
    format_recommendation,
    build_all_bracket_types,
    format_bracket_type_summary,
)

# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_COLS = {
    "canonical_team_name",
    "seed",
    "region",
    "offensive_efficiency",
    "defensive_efficiency",
    "efficiency_margin",
}

VALID_REGIONS  = {"East", "West", "South", "Midwest"}
OUTPUT_DIR     = PROJECT_ROOT / "data" / "processed"
W              = 72
SEP            = "=" * W


# ── CSV validation ────────────────────────────────────────────────────────────

def _validate_csv(df: pd.DataFrame) -> None:
    """Abort with a clear message if the CSV is malformed."""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"\n  ERROR: Missing required columns: {sorted(missing)}", file=sys.stderr)
        print(f"  Required: {sorted(REQUIRED_COLS)}", file=sys.stderr)
        print(f"  See data/future/schema.md for full column reference.", file=sys.stderr)
        sys.exit(1)

    bad_regions = set(df["region"].unique()) - VALID_REGIONS
    if bad_regions:
        print(f"\n  ERROR: Invalid region values: {sorted(bad_regions)}", file=sys.stderr)
        print(f"  Valid: {sorted(VALID_REGIONS)}", file=sys.stderr)
        sys.exit(1)

    n = len(df)
    if n not in (64, 68):
        print(f"  WARNING: Expected 64 or 68 teams, found {n}.", file=sys.stderr)

    # Duplicate (region, seed) pairs are allowed for First Four play-in slots
    # (up to 2 teams per slot).  More than 2 is always an error.
    dups = (
        df.groupby(["region", "seed"])
          .size()
          .reset_index(name="cnt")
    )
    bad_dups = dups[dups["cnt"] > 2]
    if not bad_dups.empty:
        print(f"\n  ERROR: More than 2 teams in a single (region, seed) slot:", file=sys.stderr)
        for _, r in bad_dups.iterrows():
            print(f"    {r['region']} seed {int(r['seed'])}: {int(r['cnt'])} rows",
                  file=sys.stderr)
        sys.exit(1)


# ── First Four handling ───────────────────────────────────────────────────────

def _simulate_first_four(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Detect First Four play-in pairs (duplicate region/seed rows), simulate each
    game using efficiency_margin as the strength measure, and return:
      - df_64: the 64-team DataFrame with only winners remaining
      - results: list of game result dicts for display
    """
    counts = (
        df.groupby(["region", "seed"])
          .size()
          .reset_index(name="cnt")
    )
    ff_slots = counts[counts["cnt"] == 2][["region", "seed"]]

    if ff_slots.empty:
        return df, []

    results: list[dict] = []
    loser_indices: list[int] = []

    for _, slot in ff_slots.iterrows():
        region = slot["region"]
        seed   = int(slot["seed"])
        pair   = df[(df["region"] == region) & (df["seed"] == seed)]

        # Higher efficiency_margin wins (deterministic, no outcome knowledge)
        sorted_pair = pair.sort_values("efficiency_margin", ascending=False)
        winner_row  = sorted_pair.iloc[0]
        loser_row   = sorted_pair.iloc[1]

        results.append({
            "region":     region,
            "seed":       seed,
            "winner":     winner_row["canonical_team_name"],
            "loser":      loser_row["canonical_team_name"],
            "winner_em":  round(float(winner_row["efficiency_margin"]), 1),
            "loser_em":   round(float(loser_row["efficiency_margin"]),  1),
        })
        loser_indices.append(loser_row.name)

    df_64 = df.drop(index=loser_indices).reset_index(drop=True)
    return df_64, results


def _print_first_four(results: list[dict]) -> None:
    if not results:
        return
    print()
    print("  FIRST FOUR (play-in simulation):")
    print(f"  {'Region':<8} {'Seed':>4}  {'Winner':<24} {'Loser':<24}")
    print("  " + "─" * 66)
    for r in results:
        print(f"  {r['region']:<8} #{r['seed']:>2}    "
              f"{r['winner']:<24} "
              f"{r['loser']:<24} (elim.)")


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_teams_override(df: pd.DataFrame) -> dict[str, dict[int, dict]]:
    """
    Convert the stats DataFrame to the _teams_override dict expected by
    simulate_bracket().

    team_rating (0–1)
        Normalized efficiency_margin within the full 64-team field.
        Scaled to [0.10, 0.95] so seed-1 teams sit near the top and
        seed-16 teams are well above zero.
        If the CSV already provides a 'team_rating' column, that value
        is used directly.

    champion_profile_score (0–1)
        Composite of seed quality (35%), EM rank (30%), offensive
        efficiency rank (20%), and defensive efficiency rank (15%).
        If the CSV provides 'champion_profile_score', it is used directly.

    offense_rating (0–1)
        Normalized offensive_efficiency within the field.
        Used by the model's seed-3 exception and early-round upset logic.
    """
    n   = len(df)
    df  = df.copy()

    # ── Min-max normalization helpers ─────────────────────────────────────
    def _minmax(series: pd.Series, lo: float = 0.10, hi: float = 0.95) -> pd.Series:
        mn, mx = series.min(), series.max()
        rng    = mx - mn if mx != mn else 1.0
        return ((series - mn) / rng * (hi - lo) + lo).clip(lo, hi)

    df["_team_rating"]    = _minmax(df["efficiency_margin"])
    df["_offense_rating"] = _minmax(df["offensive_efficiency"])

    # ── Rank-based scores for champion_profile_score ───────────────────
    df["_em_rank"]  = df["efficiency_margin"].rank(ascending=False, method="min")
    df["_oe_rank"]  = df["offensive_efficiency"].rank(ascending=False, method="min")
    df["_de_rank"]  = df["defensive_efficiency"].rank(ascending=True,  method="min")

    def _rank_norm(rank: float) -> float:
        return 1.0 - (rank - 1) / max(n - 1, 1)

    # ── Assemble one dict per team ─────────────────────────────────────
    teams_override: dict[str, dict[int, dict]] = {}

    for _, row in df.iterrows():
        region = str(row["region"])
        seed   = int(row["seed"])
        name   = str(row["canonical_team_name"])

        # team_rating
        if "team_rating" in df.columns and pd.notna(row.get("team_rating")):
            team_rating = float(row["team_rating"])
        else:
            team_rating = float(row["_team_rating"])

        # champion_profile_score
        if "champion_profile_score" in df.columns and pd.notna(
            row.get("champion_profile_score")
        ):
            cps = float(row["champion_profile_score"])
        else:
            seed_quality = (17 - seed) / 16.0
            em_norm      = _rank_norm(row["_em_rank"])
            oe_norm      = _rank_norm(row["_oe_rank"])
            de_norm      = _rank_norm(row["_de_rank"])
            cps          = round(
                0.35 * seed_quality
                + 0.30 * em_norm
                + 0.20 * oe_norm
                + 0.15 * de_norm,
                4,
            )

        team_dict: dict = {
            "name":                   name,
            "seed":                   seed,
            "region":                 region,
            "rating":                 round(20.0 + team_rating * 76.0, 1),
            "team_rating":            round(team_rating, 4),
            "champion_profile_score": round(cps, 4),
            "offense_rating":         round(float(row["_offense_rating"]), 4),
        }

        # ap_top12_flag — picked up by the FF/Championship AP tie-breaker
        if "ap_top12_flag" in df.columns and pd.notna(row.get("ap_top12_flag")):
            team_dict["ap_top12_flag"] = int(row["ap_top12_flag"])

        teams_override.setdefault(region, {})[seed] = team_dict

    return teams_override


# ── Public pick % ingestion ───────────────────────────────────────────────────

# Flexible column name aliases accepted in a --public-picks-file
_PICK_NAME_COLS = ("canonical_team_name", "team_name", "name")
_PICK_PCT_COLS  = ("public_pick_pct", "pick_pct", "champion_pick_pct", "pct")

# Minimum title probability to include a team in value_score tables
_PICKS_MIN_TITLE = 0.01


def _load_public_picks_file(path: Path) -> dict[str, float]:
    """
    Load public pick percentages from a standalone CSV.

    Accepts column names: canonical_team_name / team_name / name
                          public_pick_pct / pick_pct / champion_pick_pct / pct
    """
    try:
        picks_df = pd.read_csv(path)
    except Exception as e:
        print(f"\n  ERROR: Cannot read public picks file '{path}': {e}",
              file=sys.stderr)
        sys.exit(1)

    name_col = next((c for c in _PICK_NAME_COLS if c in picks_df.columns), None)
    pct_col  = next((c for c in _PICK_PCT_COLS  if c in picks_df.columns), None)

    if name_col is None or pct_col is None:
        print(
            f"\n  ERROR: public picks file must have a team name column "
            f"({'/'.join(_PICK_NAME_COLS)}) and a pick % column "
            f"({'/'.join(_PICK_PCT_COLS)}).",
            file=sys.stderr,
        )
        sys.exit(1)

    out: dict[str, float] = {}
    for _, row in picks_df.iterrows():
        val = row.get(pct_col)
        if pd.notna(val):
            v = float(val)
            if v > 0:
                out[str(row[name_col]).strip()] = v
    return out


def _build_public_picks(
    df:        pd.DataFrame,
    file_picks: dict[str, float],
    overrides:  dict[str, float],
) -> tuple[dict[str, float], list[dict]]:
    """
    Merge all pick % sources with priority: --picks > file > CSV column > seed fallback.

    Returns
    -------
    picks   : {team_name: fraction} for all teams in df
    missing : list of {name, seed, source} for teams that fell back to seed default
    """
    # Source 1: CSV column
    csv_picks: dict[str, float] = {}
    if "public_pick_pct" in df.columns:
        for _, row in df.iterrows():
            val = row.get("public_pick_pct")
            if pd.notna(val):
                v = float(val)
                if v > 0:
                    csv_picks[str(row["canonical_team_name"])] = v

    # Merge priority: overrides > file > csv
    merged: dict[str, float] = {}
    merged.update(csv_picks)
    merged.update(file_picks)
    merged.update(overrides)

    # Fill missing with seed-based fallback; build missing report
    missing: list[dict] = []
    for _, row in df.iterrows():
        name = str(row["canonical_team_name"])
        seed = int(row["seed"])
        if name not in merged:
            fallback = DEFAULT_PUBLIC_PCT.get(seed, 0.001)
            merged[name] = fallback
            source = (
                "seed default"
                if not (csv_picks.get(name) or file_picks.get(name) or overrides.get(name))
                else "partial"
            )
            missing.append({
                "name":         name,
                "seed":         seed,
                "fallback_pct": round(fallback, 5),
                "source":       source,
            })

    return merged, missing


# Normalization window: skip normalization when sum is already "close enough"
_NORM_LOW  = 0.95
_NORM_HIGH = 1.05


def _normalize_public_picks(
    picks: dict[str, float],
) -> tuple[dict[str, float], float, bool]:
    """
    Normalize pick percentages so they sum to 1.0.

    Normalization is applied only when the current sum is outside
    [_NORM_LOW, _NORM_HIGH] (i.e. outside [95%, 105%]).

    Returns
    -------
    normalized_picks : dict with rescaled values (copy, not in-place)
    original_sum     : sum before normalization
    applied          : True if normalization was performed
    """
    original_sum = sum(picks.values())
    if _NORM_LOW <= original_sum <= _NORM_HIGH:
        return dict(picks), original_sum, False

    factor = original_sum if original_sum > 0 else 1.0
    normalized = {name: pct / factor for name, pct in picks.items()}
    return normalized, original_sum, True


# ── Picks analysis ────────────────────────────────────────────────────────────

def _build_picks_rows(
    mc_results,
    public_picks: dict[str, float],
    df:           pd.DataFrame,
) -> list[dict]:
    """
    Build a joined table row per team with: seed, region, public_pct,
    MC title_prob, MC ff_prob, value_score.
    """
    rows = []
    for _, row in df.iterrows():
        name   = str(row["canonical_team_name"])
        seed   = int(row["seed"])
        region = str(row["region"])
        pub    = public_picks.get(name, DEFAULT_PUBLIC_PCT.get(seed, 0.001))

        if mc_results is not None:
            title = mc_results.title_prob(name)
            ff    = mc_results.ff_prob(name)
        else:
            title = 0.0
            ff    = 0.0

        value = round(title / max(pub, 0.0001), 3) if title >= _PICKS_MIN_TITLE else 0.0

        rows.append({
            "name": name, "seed": seed, "region": region,
            "public_pct": pub, "title_prob": title,
            "ff_prob": ff, "value_score": value,
        })
    return rows


def _print_picks_analysis(
    mc_results,
    public_picks:  dict[str, float],
    df:            pd.DataFrame,
    orig_sum:      float = 0.0,
    norm_applied:  bool  = False,
) -> None:
    rows    = _build_picks_rows(mc_results, public_picks, df)
    cur_sum = sum(public_picks.values())
    has_mc  = mc_results is not None

    print()
    print(SEP)
    print("  PICK SHARE ANALYSIS".center(W))
    print(SEP)
    if norm_applied:
        print(f"\n  Normalization  : applied")
        print(f"  Original sum   : {orig_sum:.3%}")
        print(f"  Normalized sum : {cur_sum:.3%}")
    else:
        print(f"\n  Normalization  : not needed  (sum={cur_sum:.3%}, within [95%, 105%])")

    # ── Table 1: Top 10 by MC title probability ───────────────────────
    if has_mc:
        by_title = sorted(rows, key=lambda r: r["title_prob"], reverse=True)[:10]
        print(f"\n  TOP 10 BY TITLE PROBABILITY")
        print(f"  {'#':<3} {'Name':<22} {'s':>2}  {'Region':<8}  "
              f"{'Title%':>6}  {'FF%':>5}  {'Public%':>7}  {'Value':>6}")
        print("  " + "─" * 64)
        for i, r in enumerate(by_title, 1):
            vs = f"{r['value_score']:.2f}x" if r["value_score"] > 0 else "   —"
            print(f"  {i:<3} {r['name']:<22} {r['seed']:>2}  {r['region']:<8}  "
                  f"{r['title_prob']:>6.1%}  {r['ff_prob']:>5.1%}  "
                  f"{r['public_pct']:>7.2%}  {vs:>6}")

    # ── Table 2: Top 10 by public pick % ─────────────────────────────
    by_pub = sorted(rows, key=lambda r: r["public_pct"], reverse=True)[:10]
    print(f"\n  TOP 10 BY PUBLIC PICK %")
    hdr2 = (f"  {'#':<3} {'Name':<22} {'s':>2}  {'Region':<8}  "
            f"{'Public%':>7}  {'Title%':>6}  {'Value':>6}"
            if has_mc else
            f"  {'#':<3} {'Name':<22} {'s':>2}  {'Region':<8}  {'Public%':>7}")
    print(hdr2)
    print("  " + "─" * (64 if has_mc else 44))
    for i, r in enumerate(by_pub, 1):
        if has_mc:
            vs = f"{r['value_score']:.2f}x" if r["value_score"] > 0 else "   —"
            print(f"  {i:<3} {r['name']:<22} {r['seed']:>2}  {r['region']:<8}  "
                  f"{r['public_pct']:>7.2%}  {r['title_prob']:>6.1%}  {vs:>6}")
        else:
            print(f"  {i:<3} {r['name']:<22} {r['seed']:>2}  {r['region']:<8}  "
                  f"{r['public_pct']:>7.2%}")

    # ── Table 3: Top 10 by value score (requires MC) ──────────────────
    if has_mc:
        viable = [r for r in rows if r["value_score"] > 0]
        by_val = sorted(viable, key=lambda r: r["value_score"], reverse=True)[:10]
        print(f"\n  TOP 10 BY VALUE SCORE  (title% ÷ public%,  min title {_PICKS_MIN_TITLE:.0%})")
        print(f"  {'#':<3} {'Name':<22} {'s':>2}  {'Region':<8}  "
              f"{'Value':>6}  {'Title%':>6}  {'Public%':>7}  {'FF%':>5}")
        print("  " + "─" * 64)
        for i, r in enumerate(by_val, 1):
            print(f"  {i:<3} {r['name']:<22} {r['seed']:>2}  {r['region']:<8}  "
                  f"{r['value_score']:>6.2f}x  {r['title_prob']:>6.1%}  "
                  f"{r['public_pct']:>7.2%}  {r['ff_prob']:>5.1%}")

    print()


def _picks_analysis_dict(
    mc_results,
    public_picks:  dict[str, float],
    df:            pd.DataFrame,
    orig_sum:      float = 0.0,
    norm_applied:  bool  = False,
) -> dict:
    rows = _build_picks_rows(mc_results, public_picks, df)

    def _top(sorted_rows: list[dict], n: int = 10) -> list[dict]:
        return [
            {k: round(v, 4) if isinstance(v, float) else v
             for k, v in r.items()}
            for r in sorted_rows[:n]
        ]

    cur_sum = sum(public_picks.values())
    out: dict = {
        "normalization": {
            "applied":        norm_applied,
            "original_sum":   round(orig_sum, 6),
            "normalized_sum": round(cur_sum,  6),
            "threshold_min":  _NORM_LOW,
            "threshold_max":  _NORM_HIGH,
        },
        "total_public_pct": round(cur_sum, 6),
    }
    if mc_results is not None:
        out["top_by_title_prob"] = _top(
            sorted(rows, key=lambda r: r["title_prob"], reverse=True))
        out["top_by_value_score"] = _top(
            sorted([r for r in rows if r["value_score"] > 0],
                   key=lambda r: r["value_score"], reverse=True))
    out["top_by_public_pct"] = _top(
        sorted(rows, key=lambda r: r["public_pct"], reverse=True))
    return out


def _parse_picks(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    out: dict[str, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        name, val = token.split("=", 1)
        try:
            out[name.strip()] = float(val.strip())
        except ValueError:
            print(f"  Warning: could not parse pick '{token}' — skipping",
                  file=sys.stderr)
    return out


# ── Strategy summary helpers ─────────────────────────────────────────────────

_MIN_VIABLE_PROB    = 0.025   # floor for "viable" champion pick
_CONTRARIAN_PUB_CAP = 0.05   # public pick % ceiling for "contrarian"


def _build_strategy_summary(
    bracket:    dict,
    mc_results,           # MCResults | None
    candidates: list,     # list[ChampionCandidate]
) -> dict:
    """
    Derive the five headline strategy picks.

    Returns a dict with keys:
      deterministic_champion, mc_champion, safest_pick,
      best_value_pick, most_contrarian_viable
    Each value is a sub-dict with name/seed/region + relevant prob fields.
    """
    det_champ_dict = bracket.get("champion", {})
    det = {
        "name":   det_champ_dict.get("name", "?"),
        "seed":   det_champ_dict.get("seed"),
        "region": det_champ_dict.get("region"),
    }

    mc_champ = None
    if mc_results is not None:
        top = mc_results.top_by_title(1)
        if top:
            r = top[0]
            mc_champ = {
                "name":       r.name,
                "seed":       r.seed,
                "region":     r.region,
                "title_prob": round(r.title_prob, 4),
                "ff_prob":    round(r.ff_prob,    4),
            }

    viable = [c for c in candidates if c.win_prob >= _MIN_VIABLE_PROB]

    safest = None
    if viable:
        c = max(viable, key=lambda x: x.win_prob)
        safest = {
            "name":       c.name, "seed": c.seed, "region": c.region,
            "title_prob": c.win_prob, "ff_prob": c.mc_ff_prob,
            "public_pct": c.public_pct,
        }

    best_value = None
    if viable:
        c = max(viable, key=lambda x: x.value_score)
        best_value = {
            "name":        c.name, "seed": c.seed, "region": c.region,
            "title_prob":  c.win_prob, "ff_prob": c.mc_ff_prob,
            "public_pct":  c.public_pct, "value_score": c.value_score,
        }

    contrarian_pool = [
        c for c in viable if c.public_pct < _CONTRARIAN_PUB_CAP
    ]
    most_contrarian = None
    if contrarian_pool:
        c = max(contrarian_pool, key=lambda x: x.value_score)
        most_contrarian = {
            "name":        c.name, "seed": c.seed, "region": c.region,
            "title_prob":  c.win_prob, "ff_prob": c.mc_ff_prob,
            "public_pct":  c.public_pct, "value_score": c.value_score,
        }

    return {
        "deterministic_champion": det,
        "mc_champion":            mc_champ,
        "safest_pick":            safest,
        "best_value_pick":        best_value,
        "most_contrarian_viable": most_contrarian,
    }


def _print_strategy_summary(summary: dict) -> None:
    """Print the clean strategy summary block."""
    W2  = 72
    print()
    print("=" * W2)
    print("  STRATEGY SUMMARY".center(W2))
    print("=" * W2)

    def _fmt(label: str, d: dict | None) -> None:
        if d is None:
            print(f"  {label:<30}  —")
            return
        name_str = f"{d['name']} (#{d.get('seed','?')} {d.get('region','')})"
        extras = []
        if d.get("title_prob"):
            extras.append(f"{d['title_prob']:.1%} title")
        if d.get("ff_prob"):
            extras.append(f"{d['ff_prob']:.1%} FF")
        if d.get("value_score") and d.get("value_score") != d.get("title_prob"):
            extras.append(f"{d['value_score']:.2f}× value")
        if d.get("public_pct"):
            extras.append(f"{d['public_pct']:.1%} public")
        suffix = "  — " + ",  ".join(extras) if extras else ""
        print(f"  {label:<30}  {name_str}{suffix}")

    _fmt("Deterministic champion",  summary["deterministic_champion"])
    _fmt("MC champion",             summary["mc_champion"])
    _fmt("Safest pick",             summary["safest_pick"])
    _fmt("Best value pick",         summary["best_value_pick"])
    _fmt("Most contrarian viable",  summary["most_contrarian_viable"])
    print("=" * W2)


# ── Display helpers ───────────────────────────────────────────────────────────

def _print_bracket_summary(bracket: dict, season) -> None:
    """Print champion, Final Four, Championship, and notable upsets."""
    label = f"BRACKET PREDICTION — {season}"
    print(SEP)
    print(label.center(W))
    print(SEP)

    champ = bracket.get("champion", {})
    print(f"\n  Champion: {champ.get('name', '?')} "
          f"(#{champ.get('seed', '?')} {champ.get('region', '?')})\n")

    print("  Final Four:")
    for game in bracket.get("final_four", []):
        w = game.get("winner", {})
        lo = game.get("loser", {})
        print(f"    #{w.get('seed','?')} {w.get('name','?'):22s}"
              f" def. #{lo.get('seed','?')} {lo.get('name','?')}")

    print()
    print("  Championship:")
    cg = bracket.get("championship") or {}
    w  = cg.get("winner", {})
    lo = cg.get("loser", {})
    print(f"    #{w.get('seed','?')} {w.get('name','?'):22s}"
          f" def. #{lo.get('seed','?')} {lo.get('name','?')}")

    # Upset summary — underdog (higher seed number) beat the favorite
    upsets: list[tuple[str, int, str, int, str]] = []
    for rnd_key, rnd_label in [
        ("round_of_64",  "R64"),
        ("round_of_32",  "R32"),
        ("sweet_16",     "S16"),
        ("elite_8",      "E8"),
        ("final_four",   "FF"),
    ]:
        games = bracket.get(rnd_key, [])
        if isinstance(games, dict):
            games = [games]
        for game in (games or []):
            w_s = int(game.get("winner", {}).get("seed", 0))
            l_s = int(game.get("loser",  {}).get("seed", 0))
            if w_s > l_s:
                upsets.append((
                    rnd_label,
                    w_s, game["winner"].get("name", "?"),
                    l_s, game["loser"].get("name", "?"),
                ))

    print()
    if upsets:
        print(f"  Notable upsets ({len(upsets)} total — showing up to 10):")
        for rnd, ws, wn, ls, ln in sorted(upsets, key=lambda x: x[1])[:10]:
            print(f"    [{rnd}] #{ws} {wn} over #{ls} {ln}")
    else:
        print("  Notable upsets: none")


def _clean_for_json(obj):
    """Strip internal '_'-prefixed and 'score' fields before saving."""
    if isinstance(obj, dict):
        return {
            k: _clean_for_json(v)
            for k, v in obj.items()
            if k != "score" and not k.startswith("_")
        }
    if isinstance(obj, list):
        return [_clean_for_json(i) for i in obj]
    return obj


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a bracket from future-season team stats CSV."
    )
    parser.add_argument("--input",  required=True,
                        help="Path to team stats CSV (required)")
    parser.add_argument("--season", type=int, default=None,
                        help="Tournament year (display/naming only, e.g. 2026)")
    parser.add_argument("--pool",   type=int, default=100,
                        help="Estimated pool size (default: 100)")
    parser.add_argument("--n",      type=int, default=0,
                        help="Portfolio brackets to generate; 0 = single only (default: 0)")
    parser.add_argument("--mode",   default="balanced",
                        choices=["conservative", "balanced", "upset_heavy"],
                        help="Bracket mode (default: balanced)")
    parser.add_argument("--picks",  default=None,
                        help='Override public pick %% as "Name=0.18,..." pairs')
    parser.add_argument("--output", default=None,
                        help="Output JSON path (overrides default naming)")
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Run Monte Carlo simulations to estimate advancement probabilities")
    parser.add_argument("--sims", type=int, default=10000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--public-picks-file", default=None, metavar="FILE",
                        help="CSV with columns canonical_team_name + public_pick_pct "
                             "(overrides CSV column, lower priority than --picks)")
    args = parser.parse_args()

    picks_override = _parse_picks(args.picks)
    season_label   = args.season or "future"

    # ── Load & validate CSV ────────────────────────────────────────────
    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"\n  ERROR: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    _validate_csv(df)

    print(SEP)
    print("FUTURE BRACKET PREDICTOR".center(W))
    print(f"Season: {season_label}   Mode: {args.mode.upper()}   "
          f"Pool: {args.pool}   Brackets: {max(args.n, 1)}".center(W))
    print(SEP)
    print(f"\n  Loaded {len(df)} teams from {csv_path.name}")

    # ── First Four ─────────────────────────────────────────────────────
    df, first_four_results = _simulate_first_four(df)
    if first_four_results:
        print(f"  First Four: {len(first_four_results)} play-in game(s) simulated")
        _print_first_four(first_four_results)
        print(f"\n  Proceeding with {len(df)} main-draw teams")

    # ── Load public picks file (if provided) ──────────────────────────
    file_picks: dict[str, float] = {}
    if args.public_picks_file:
        pp_path = Path(args.public_picks_file)
        if not pp_path.exists():
            print(f"\n  ERROR: Public picks file not found: {pp_path}", file=sys.stderr)
            sys.exit(1)
        file_picks = _load_public_picks_file(pp_path)
        print(f"  Loaded {len(file_picks)} pick % entries from {pp_path.name}")

    # ── Build team overrides + public picks ───────────────────────────
    print("  Computing team ratings from efficiency stats...", end="", flush=True)
    teams_override = _build_teams_override(df)
    public_picks, missing_picks = _build_public_picks(df, file_picks, picks_override)
    print("  done")

    # Normalize pick percentages
    public_picks, orig_sum, norm_applied = _normalize_public_picks(public_picks)

    # Pick coverage report
    n_real = len(public_picks) - len(missing_picks)
    print(f"\n  Public pick % coverage: {n_real}/{len(df)} teams with real data, "
          f"{len(missing_picks)} at seed defaults")
    if norm_applied:
        print(f"  Normalization applied:  {orig_sum:.3%} → {sum(public_picks.values()):.3%}")
    else:
        print(f"  Normalization skipped:  sum={sum(public_picks.values()):.3%} within [95%, 105%]")
    if missing_picks:
        print(f"  Seed-fallback teams ({len(missing_picks)}):")
        for m in sorted(missing_picks, key=lambda x: x["seed"])[:8]:
            print(f"    #{m['seed']:>2} {m['name']:<26}  → {m['fallback_pct']:.3%}")
        if len(missing_picks) > 8:
            print(f"    ... +{len(missing_picks) - 8} more")

    # Top-5 by public pick % (quick sanity check)
    top5_picks = sorted(public_picks.items(), key=lambda x: -x[1])[:5]
    print(f"\n  Top 5 by public pick %:  "
          + "  ".join(f"{nm}={pct:.2%}" for nm, pct in top5_picks))

    # Top-5 by efficiency margin (model sanity check)
    top5 = df.copy()
    top5["_tr"] = df["efficiency_margin"].rank(ascending=False, method="min")
    top5 = top5.sort_values("_tr").head(5)
    print(f"\n  Top 5 by efficiency margin:")
    print(f"  {'Name':<24} {'s':>2}  {'Region':<8}  {'EM':>6}  {'OE':>6}  {'DE':>6}")
    print("  " + "─" * 56)
    for _, r in top5.iterrows():
        print(f"  {r['canonical_team_name']:<24} {int(r['seed']):>2}  "
              f"{r['region']:<8}  {r['efficiency_margin']:>6.1f}  "
              f"{r['offensive_efficiency']:>6.1f}  {r['defensive_efficiency']:>6.1f}")

    # ── Simulate bracket ───────────────────────────────────────────────
    print(f"\n  Simulating {args.mode} bracket...", end="", flush=True)
    bracket = simulate_bracket(args.mode, _teams_override=teams_override)
    print("  done\n")

    _print_bracket_summary(bracket, season_label)

    # ── Monte Carlo (optional) ─────────────────────────────────────────
    mc_results = None
    if args.monte_carlo:
        if not _HAS_MC:
            print("\n  WARNING: lib/monte_carlo.py not found — skipping MC.",
                  file=sys.stderr)
        else:
            print(f"\n  Running Monte Carlo ({args.sims:,} simulations)...",
                  end="", flush=True)
            mc_results = run_monte_carlo(
                teams_override=teams_override,
                n_sims=args.sims,
            )
            print("  done")
            print(format_mc_summary(mc_results, top_n=10))

    # ── Picks analysis (title prob + public pick + value) ──────────────
    _print_picks_analysis(mc_results, public_picks, df,
                          orig_sum=orig_sum, norm_applied=norm_applied)

    # ── Champion candidates (always computed when MC run or portfolio requested)
    candidates = []
    if mc_results is not None or args.n > 0:
        source = "MC title probs" if mc_results is not None else "path-based estimate"
        print(f"\n  Extracting champion candidates ({source})...")
        candidates = extract_candidates(bracket, public_picks or None, mc_results)
        print(f"  Found {len(candidates)} candidates\n")
        has_mc_ff = mc_results is not None
        hdr = (f"  {'Name':<22} {'s':>2}  {'Region':<8}  "
               f"{'Title%':>6}  {'FF%':>5}  {'Pick%':>6}  {'Value':>6}  {'E8/FF?':>6}"
               if has_mc_ff else
               f"  {'Name':<22} {'s':>2}  {'Region':<8}  "
               f"{'WinP':>5}  {'PubP':>6}  {'Value':>6}  {'FF?':>4}")
        print(hdr)
        print("  " + "─" * (64 if has_mc_ff else 60))
        for c in candidates:
            ff_tag = "FF" if c.in_base_ff else ("E8" if c.in_base_e8 else "—")
            if has_mc_ff:
                print(f"  {c.name:<22} {c.seed:>2}  {c.region:<8}  "
                      f"{c.win_prob:>6.1%}  {c.mc_ff_prob:>5.1%}  {c.public_pct:>6.2%}  "
                      f"{c.value_score:>6.2f}x  {ff_tag:>6}")
            else:
                print(f"  {c.name:<22} {c.seed:>2}  {c.region:<8}  "
                      f"{c.win_prob:>5.1%}  {c.public_pct:>6.2%}  "
                      f"{c.value_score:>6.2f}x  {ff_tag:>4}")

    # ── Portfolio (optional) ───────────────────────────────────────────
    portfolio_entries = []
    if args.n > 0:
        print(f"\n  Building {args.n}-bracket portfolio (pool={args.pool})...")
        portfolio_entries = generate_portfolio(
            base_bracket=bracket,
            n=args.n,
            pool_size=args.pool,
            public_picks=public_picks or None,
            mc_results=mc_results,
        )
        print(f"  Generated {len(portfolio_entries)} bracket entries\n")
        print(format_portfolio(portfolio_entries, args.pool))

    # ── Strategy summary + pool recommendation ────────────────────────
    summary       = None
    pool_rec      = None
    all_types     = None
    if candidates or mc_results is not None:
        summary = _build_strategy_summary(bracket, mc_results, candidates)
        _print_strategy_summary(summary)

    if candidates:
        pool_rec = build_recommendation(candidates, args.pool)
        print(format_recommendation(pool_rec))

        det_champion = summary["deterministic_champion"] if summary else {}
        all_types = build_all_bracket_types(candidates, det_champion)
        print(format_bracket_type_summary(all_types, args.pool))

    # ── Save JSON ──────────────────────────────────────────────────────
    season_str = str(args.season) if args.season else "future"
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = OUTPUT_DIR / f"future_bracket_{season_str}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    champ = bracket.get("champion", {})
    ff    = bracket.get("final_four", [])
    cg    = bracket.get("championship") or {}

    output: dict = {
        "season":      season_label,
        "mode":        args.mode,
        "pool_size":   args.pool,
        "first_four":  first_four_results,
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
        "bracket": _clean_for_json(bracket),
    }

    output["picks_analysis"] = _picks_analysis_dict(mc_results, public_picks, df,
                                                     orig_sum=orig_sum,
                                                     norm_applied=norm_applied)
    output["picks_missing"]  = missing_picks

    if summary is not None:
        output["strategy_summary"] = summary

    if pool_rec is not None:
        output["pool_recommendation"] = pool_rec.to_dict()

    if all_types is not None:
        def _cand_dict(c) -> dict | None:
            if c is None:
                return None
            return {
                "name":        c.name,
                "seed":        c.seed,
                "region":      c.region,
                "title_prob":  round(c.win_prob,    4),
                "ff_prob":     round(c.mc_ff_prob,  4),
                "public_pct":  round(c.public_pct,  4),
                "value_score": round(c.value_score, 3),
            }

        det = all_types["deterministic"]
        output["deterministic_recommendation"] = {
            "bracket_type": "deterministic",
            "label":        det["label"],
            "archetype":    det["archetype"],
            "description":  det["description"],
            "champion":     det.get("champion"),
        }
        for btype in ("safe", "value", "contrarian"):
            entry = all_types[btype]
            rec   = entry["recommendation"]
            output[f"{btype}_recommendation"] = {
                "bracket_type": btype,
                "label":        entry["label"],
                "archetype":    entry["archetype"],
                "description":  entry["description"],
                "pool_category": rec.tier,
                "n_brackets":   rec.n_brackets,
                "primary":      _cand_dict(rec.primary),
                "primary_reason": rec.primary_reason,
                "safest_alt":   _cand_dict(rec.safest_alt),
                "value_alt":    _cand_dict(rec.value_alt),
            }

    if mc_results is not None:
        output["monte_carlo"] = mc_results.to_dict()

    if portfolio_entries:
        output["portfolio"] = [
            {
                "index":       e.index,
                "champion":    e.champion.name,
                "seed":        e.champion.seed,
                "region":      e.champion.region,
                "win_prob":    e.champion.win_prob,
                "public_pct":  e.champion.public_pct,
                "value_score": e.champion.value_score,
                "composite":   e.champion.composite,
                "rationale":   e.rationale,
                "ev_note":     e.ev_note,
                "bracket":     _clean_for_json(e.bracket),
            }
            for e in portfolio_entries
        ]

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved → {out_path}")
    print(SEP)


if __name__ == "__main__":
    main()
