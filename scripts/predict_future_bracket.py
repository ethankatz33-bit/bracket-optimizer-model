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


def _extract_public_picks(df: pd.DataFrame, overrides: dict[str, float]) -> dict[str, float]:
    """
    Build public_picks dict: CSV column first, then --picks overrides win.
    """
    picks: dict[str, float] = {}
    if "public_pick_pct" in df.columns:
        for _, row in df.iterrows():
            val = row.get("public_pick_pct")
            if pd.notna(val):
                v = float(val)
                if v > 0:
                    picks[str(row["canonical_team_name"])] = v
    picks.update(overrides)     # --picks flag always wins
    return picks


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

    # ── Build team overrides ───────────────────────────────────────────
    print("  Computing team ratings from efficiency stats...", end="", flush=True)
    teams_override = _build_teams_override(df)
    public_picks   = _extract_public_picks(df, picks_override)
    print("  done")

    # Show top-5 teams by computed team_rating for a quick sanity check
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

    if public_picks:
        print(f"\n  Public pick data: {len(public_picks)} team(s)")
        for nm, pct in sorted(public_picks.items(), key=lambda x: -x[1])[:5]:
            print(f"    {nm}: {pct:.1%}")
        if len(public_picks) > 5:
            print(f"    ... +{len(public_picks) - 5} more")

    # ── Simulate bracket ───────────────────────────────────────────────
    print(f"\n  Simulating {args.mode} bracket...", end="", flush=True)
    bracket = simulate_bracket(args.mode, _teams_override=teams_override)
    print("  done\n")

    _print_bracket_summary(bracket, season_label)

    # ── Portfolio (optional) ───────────────────────────────────────────
    portfolio_entries = []
    if args.n > 0:
        print(f"\n  Extracting champion candidates from E8...")
        candidates = extract_candidates(bracket, public_picks or None)
        print(f"  Found {len(candidates)} candidates\n")
        print(f"  {'Name':<22} {'s':>2}  {'Region':<8}  "
              f"{'WinP':>5}  {'PubP':>6}  {'Value':>6}  {'FF?':>4}")
        print("  " + "─" * 60)
        for c in candidates:
            ff_tag = "YES" if c.in_base_ff else ("E8" if c.in_base_e8 else "no")
            print(f"  {c.name:<22} {c.seed:>2}  {c.region:<8}  "
                  f"{c.win_prob:>5.1%}  {c.public_pct:>6.2%}  "
                  f"{c.value_score:>6.2f}x  {ff_tag:>4}")

        print(f"\n  Building {args.n}-bracket portfolio (pool={args.pool})...")
        portfolio_entries = generate_portfolio(
            base_bracket=bracket,
            n=args.n,
            pool_size=args.pool,
            public_picks=public_picks or None,
        )
        print(f"  Generated {len(portfolio_entries)} bracket entries\n")
        print(format_portfolio(portfolio_entries, args.pool))

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
        "champion":  {
            "name":   champ.get("name"),
            "seed":   champ.get("seed"),
            "region": champ.get("region"),
        },
        "final_four": [
            {
                "winner": g.get("winner", {}).get("name"),
                "winner_seed": g.get("winner", {}).get("seed"),
                "loser":  g.get("loser",  {}).get("name"),
                "loser_seed":  g.get("loser",  {}).get("seed"),
            }
            for g in ff
        ],
        "championship": {
            "winner": cg.get("winner", {}).get("name"),
            "winner_seed": cg.get("winner", {}).get("seed"),
            "loser":  cg.get("loser",  {}).get("name"),
            "loser_seed":  cg.get("loser",  {}).get("seed"),
        },
        "bracket": _clean_for_json(bracket),
    }

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
