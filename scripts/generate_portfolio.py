"""
scripts/generate_portfolio.py
Champion-value bracket portfolio generator.

Generates N diverse champion-first brackets ranked by expected pool value.

Usage
-----
  python3 scripts/generate_portfolio.py [options]

Options
-------
  --n N            Number of brackets in portfolio (default: 5)
  --pool POOL_SIZE  Estimated number of entrants in your pool (default: 100)
  --mode MODE       Bracket mode: conservative|balanced|upset_heavy (default: balanced)
  --picks K=V,...   Override public pick percentages as comma-separated key=value pairs.
                    Example:  --picks "Duke=0.18,Kansas=0.22"

Output
------
  Portfolio summary table (all brackets)
  Per-bracket detail: champion, path, rationale, expected-value note
  JSON file: data/processed/portfolio.json

Examples
--------
  # 5-bracket portfolio for a 50-person office pool
  python3 scripts/generate_portfolio.py --n 5 --pool 50

  # 8-bracket portfolio for a large ESPN-style pool
  python3 scripts/generate_portfolio.py --n 8 --pool 10000

  # With real public pick data
  python3 scripts/generate_portfolio.py --n 6 --pool 200 --picks "T1181=0.22,T1314=0.15"
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.team_selector import simulate_bracket
from lib.bracket_strategy import (
    extract_candidates,
    generate_portfolio,
    format_portfolio,
)

PORTFOLIO_FILE = PROJECT_ROOT / "data" / "processed" / "portfolio.json"

W   = 72
SEP = "=" * W


def _parse_picks(raw: str | None) -> dict[str, float]:
    """Parse '--picks "Name=0.18,Other=0.12"' into {name: float}."""
    if not raw:
        return {}
    out = {}
    for token in raw.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        name, val = token.split("=", 1)
        try:
            out[name.strip()] = float(val.strip())
        except ValueError:
            print(f"  Warning: could not parse pick '{token}' — skipping", file=sys.stderr)
    return out


def _clean_for_json(obj):
    """Strip internal 'score' fields before saving."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items() if k != "score"}
    if isinstance(obj, list):
        return [_clean_for_json(i) for i in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a champion-value bracket portfolio."
    )
    parser.add_argument("--n",    type=int, default=5,
                        help="Number of brackets (default: 5)")
    parser.add_argument("--pool", type=int, default=100,
                        help="Estimated pool size (default: 100)")
    parser.add_argument("--mode", default="balanced",
                        choices=["conservative", "balanced", "upset_heavy"],
                        help="Bracket simulation mode (default: balanced)")
    parser.add_argument("--picks", default=None,
                        help='Override public pick %% as "Name=0.18,..." pairs')
    args = parser.parse_args()

    public_picks = _parse_picks(args.picks)

    # ── Header ────────────────────────────────────────────────────────────
    print(SEP)
    print("BRACKET PORTFOLIO GENERATOR".center(W))
    print(f"Mode: {args.mode.upper()}   Pool: {args.pool} people   "
          f"Brackets: {args.n}".center(W))
    print(SEP)

    # ── Generate base bracket ─────────────────────────────────────────────
    print(f"\n  Generating base {args.mode} bracket...", end="", flush=True)
    base_bracket = simulate_bracket(args.mode)
    print("  done")

    base_champ = base_bracket.get("champion", {})
    print(f"  Base model champion: {base_champ.get('name', 'unknown')}"
          f" (seed {base_champ.get('seed', '?')})")

    if public_picks:
        print(f"\n  Public pick overrides: {len(public_picks)} team(s)")
        for name, pct in public_picks.items():
            print(f"    {name}: {pct:.1%}")

    # ── Extract candidates ────────────────────────────────────────────────
    print(f"\n  Extracting champion candidates from E8...")
    candidates = extract_candidates(base_bracket, public_picks or None)
    print(f"  Found {len(candidates)} candidates (all E8 participants)")
    print()
    print(f"  {'Name':<22} {'s':>2}  {'Region':<8}  "
          f"{'WinP':>5}  {'PubP':>6}  {'Value':>6}  {'FF?':>4}")
    print("  " + "─" * 60)
    for c in candidates:
        ff_tag = "YES" if c.in_base_ff else ("E8" if c.in_base_e8 else "no")
        print(f"  {c.name:<22} {c.seed:>2}  {c.region:<8}  "
              f"{c.win_prob:>5.1%}  {c.public_pct:>6.2%}  {c.value_score:>6.2f}x  {ff_tag:>4}")

    # ── Generate portfolio ────────────────────────────────────────────────
    print(f"\n  Building portfolio ({args.n} brackets, pool={args.pool})...")
    entries = generate_portfolio(
        base_bracket=base_bracket,
        n=args.n,
        pool_size=args.pool,
        public_picks=public_picks or None,
    )
    print(f"  Generated {len(entries)} bracket entries\n")

    # ── Print formatted portfolio ─────────────────────────────────────────
    print(format_portfolio(entries, args.pool))

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "pool_size":   args.pool,
        "mode":        args.mode,
        "n_brackets":  len(entries),
        "base_champion": {
            "name": base_champ.get("name"),
            "seed": base_champ.get("seed"),
        },
        "brackets": [
            {
                "index":       e.index,
                "champion":    e.champion.name,
                "seed":        e.champion.seed,
                "region":      e.champion.region,
                "win_prob":    e.champion.win_prob,
                "public_pct":  e.champion.public_pct,
                "value_score": e.champion.value_score,
                "composite":   e.champion.composite,
                "in_base_ff":  e.champion.in_base_ff,
                "rationale":   e.rationale,
                "ev_note":     e.ev_note,
                "bracket":     _clean_for_json(e.bracket),
            }
            for e in entries
        ],
    }

    PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved → {PORTFOLIO_FILE}")
    print(SEP)


if __name__ == "__main__":
    main()
