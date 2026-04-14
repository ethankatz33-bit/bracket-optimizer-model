"""
build_champion_profile.py
Compute champion profile scores and build historical analysis.

Usage
-----
  python3 scripts/build_champion_profile.py

What it does
------------
  1. Loads historical_team_ratings.csv (built by build_historical_ratings.py)
  2. Computes within-season profile_score for every team-season row
  3. Saves updated CSV back to historical_team_ratings.csv (adds profile_score col)
  4. Identifies which teams reached each round from TourneyCompactResults.csv
  5. Aggregates stats for: champions, title game, Final Four, Elite 8, all teams
  6. Prints a detailed analysis report
  7. Saves champion_profile_stats.json

Outputs
-------
  data/processed/historical_team_ratings.csv  (updated with profile_score)
  data/processed/champion_profile_stats.json
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from lib.champion_profile import (
    compute_profile_scores_df,
    identify_round_teams,
    build_champion_stats,
    save_profile_stats,
    PROFILE_FEATURES,
)

HIST_CSV      = PROJECT_ROOT / "data" / "processed" / "historical_team_ratings.csv"
RESULTS_CSV   = PROJECT_ROOT / "data" / "raw"       / "TourneyCompactResults.csv"
STATS_JSON    = PROJECT_ROOT / "data" / "processed" / "champion_profile_stats.json"

W    = 72
SEP  = "=" * W
THIN = "─" * W

START_YEAR = 1990
END_YEAR   = 2023


# ── Formatting ────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v:.1%}"


def _fmt(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "  —  "
    return f"{v:.{decimals}f}"


# ── Print sections ────────────────────────────────────────────────────────────

GROUP_LABELS = {
    "champion":   "Champions      (n={n})",
    "title_game": "Title Game     (n={n})",
    "final_four": "Final Four     (n={n})",
    "elite_8":    "Elite 8        (n={n})",
    "all_teams":  "All Teams      (n={n})",
}

FEAT_DISPLAY = {
    "offense_rating":       "Offense Rtg",
    "efficiency_margin":    "Eff. Margin",
    "defense_rating":       "Defense Rtg",
    "recent_form":          "Recent Form",
    "strength_of_schedule": "SOS",
}

FLAG_DISPLAY = {
    "is_ap_top12_week6":      "AP Top-12 (Wk6)",
    "is_top10_kenpom_torvik": "Top-10 KenPom/Torvik",
    "is_top10_offense":       "Top-10 Offense",
}


def print_feature_table(stats: dict) -> None:
    groups = ["champion", "title_game", "final_four", "elite_8", "all_teams"]
    present = [g for g in groups if g in stats and stats[g]]

    print(f"\n{THIN}")
    print(f"  FEATURE AVERAGES  (raw values)")
    print(THIN)

    col_w = 16
    header = f"  {'Feature':<22}" + "".join(
        f"{'Champ' if g == 'champion' else 'TitleGm' if g == 'title_game' else 'FF' if g == 'final_four' else 'E8' if g == 'elite_8' else 'All':>{col_w}}"
        for g in present
    )
    print(header)
    print(f"  {'─'*22}" + "─" * (col_w * len(present)))

    for feat, label in FEAT_DISPLAY.items():
        row = f"  {label:<22}"
        for g in present:
            fstats = stats[g].get("features", {}).get(feat)
            val = _fmt(fstats["mean"]) if fstats else "  —  "
            row += f"{val:>{col_w}}"
        print(row)


def print_percentile_table(stats: dict) -> None:
    groups = ["champion", "title_game", "final_four", "elite_8", "all_teams"]
    present = [g for g in groups if g in stats and stats[g]]

    print(f"\n{THIN}")
    print(f"  WITHIN-SEASON PERCENTILE RANKS  (higher = better relative to season)")
    print(THIN)

    col_w = 16
    header = f"  {'Feature':<22}" + "".join(
        f"{'Champ' if g == 'champion' else 'TitleGm' if g == 'title_game' else 'FF' if g == 'final_four' else 'E8' if g == 'elite_8' else 'All':>{col_w}}"
        for g in present
    )
    print(header)
    print(f"  {'─'*22}" + "─" * (col_w * len(present)))

    for feat, label in FEAT_DISPLAY.items():
        row = f"  {label:<22}"
        for g in present:
            pct = stats[g].get("percentile_means", {}).get(feat)
            val = _fmt(pct, 3) if pct is not None else "  —  "
            row += f"{val:>{col_w}}"
        print(row)


def print_profile_score_table(stats: dict) -> None:
    groups = ["champion", "title_game", "final_four", "elite_8", "all_teams"]
    present = [g for g in groups if g in stats and stats[g]]

    print(f"\n{THIN}")
    print(f"  PROFILE SCORE  [0,1]  (weighted within-season percentile)")
    print(THIN)

    for g in present:
        ps = stats[g].get("profile_score")
        n  = stats[g].get("n", 0)
        label = GROUP_LABELS.get(g, g).format(n=n)
        if ps:
            print(
                f"  {label:<28}  "
                f"mean={_fmt(ps['mean'], 4)}  "
                f"std={_fmt(ps['std'], 4)}  "
                f"p25={_fmt(ps['p25'], 4)}  "
                f"p75={_fmt(ps['p75'], 4)}"
            )
        else:
            print(f"  {label:<28}  (no data)")


def print_flag_table(stats: dict) -> None:
    groups = ["champion", "title_game", "final_four", "elite_8", "all_teams"]
    present = [g for g in groups if g in stats and stats[g]]

    # Check if any flags exist
    any_flags = any(
        stats[g].get("flags") for g in present
    )
    if not any_flags:
        return

    print(f"\n{THIN}")
    print(f"  FLAG PREVALENCE  (fraction of teams with each flag)")
    print(THIN)

    col_w = 16
    header = f"  {'Flag':<28}" + "".join(
        f"{'Champ' if g == 'champion' else 'TitleGm' if g == 'title_game' else 'FF' if g == 'final_four' else 'E8' if g == 'elite_8' else 'All':>{col_w}}"
        for g in present
    )
    print(header)
    print(f"  {'─'*28}" + "─" * (col_w * len(present)))

    for flag, label in FLAG_DISPLAY.items():
        row = f"  {label:<28}"
        for g in present:
            fv = stats[g].get("flags", {}).get(flag)
            val = _pct(fv) if fv is not None else "  —  "
            row += f"{val:>{col_w}}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load and validate ─────────────────────────────────────────────────
    if not HIST_CSV.exists():
        sys.exit(
            f"Error: {HIST_CSV} not found.\n"
            "Run scripts/build_historical_ratings.py first."
        )
    if not RESULTS_CSV.exists():
        sys.exit(f"Error: {RESULTS_CSV} not found.")

    print(SEP)
    print(f"{'CHAMPION PROFILE ANALYSIS  —  {}-{}'.format(START_YEAR, END_YEAR):^{W}}")
    print(SEP)

    # ── Step 1: Load historical ratings ──────────────────────────────────
    print(f"\n  Loading {HIST_CSV.name} …", end=" ")
    hist_df = pd.read_csv(HIST_CSV)
    hist_df = hist_df[
        (hist_df["season"] >= START_YEAR) & (hist_df["season"] <= END_YEAR)
    ].copy()
    print(f"{len(hist_df)} rows, {hist_df['season'].nunique()} seasons")

    # ── Step 2: Compute profile_score ────────────────────────────────────
    print(f"  Computing profile_score …", end=" ")
    missing_feats = [f for f in PROFILE_FEATURES if f not in hist_df.columns]
    if missing_feats:
        print(f"\n  WARNING: missing features: {missing_feats}")

    hist_df = compute_profile_scores_df(hist_df)
    print(f"done  (mean={hist_df['profile_score'].mean():.4f}, "
          f"std={hist_df['profile_score'].std():.4f})")

    # ── Step 3: Save updated CSV ──────────────────────────────────────────
    # Load full CSV (may include seasons outside our window), update profile_score
    full_df = pd.read_csv(HIST_CSV)
    if "profile_score" in full_df.columns:
        full_df = full_df.drop(columns=["profile_score"])
    full_df = full_df.merge(
        hist_df[["season", "team_id", "profile_score"]],
        on=["season", "team_id"],
        how="left",
    )
    full_df.to_csv(HIST_CSV, index=False)
    print(f"  Saved updated ratings → {HIST_CSV.name}")

    # ── Step 4: Identify round teams ──────────────────────────────────────
    print(f"  Parsing round teams from {RESULTS_CSV.name} …", end=" ")
    round_teams = identify_round_teams(RESULTS_CSV, START_YEAR, END_YEAR)
    n_years = len(round_teams)
    n_champs = sum(len(v.get("champion", [])) for v in round_teams.values())
    print(f"{n_years} seasons, {n_champs} champions identified")

    # ── Step 5: Build stats ───────────────────────────────────────────────
    print(f"  Building champion stats …", end=" ")
    stats = build_champion_stats(hist_df, round_teams)
    print("done")

    # ── Step 6: Print report ──────────────────────────────────────────────
    print_feature_table(stats)
    print_percentile_table(stats)
    print_profile_score_table(stats)
    print_flag_table(stats)

    # Champion vs. all_teams gap summary
    print(f"\n{THIN}")
    print(f"  KEY DISCRIMINATORS  (champion vs. all teams)")
    print(THIN)

    champ_feats = stats.get("champion", {}).get("features", {})
    all_feats   = stats.get("all_teams", {}).get("features", {})
    champ_pcts  = stats.get("champion", {}).get("percentile_means", {})

    for feat, label in FEAT_DISPLAY.items():
        cm = champ_feats.get(feat, {}).get("mean")
        am = all_feats.get(feat, {}).get("mean")
        cp = champ_pcts.get(feat)
        if cm is None or am is None:
            continue
        gap = cm - am
        direction = "↑" if gap > 0 else "↓"
        print(
            f"  {label:<22}  champ={_fmt(cm):>7}  all={_fmt(am):>7}  "
            f"gap={direction}{abs(gap):.2f}  "
            f"pct={_fmt(cp, 3) if cp else '—':>6}"
        )

    # Profile score summary line
    champ_ps = stats.get("champion", {}).get("profile_score", {})
    all_ps   = stats.get("all_teams", {}).get("profile_score", {})
    if champ_ps and all_ps:
        print(f"\n  profile_score  →  champions: {champ_ps['mean']:.4f}  "
              f"all teams: {all_ps['mean']:.4f}  "
              f"gap: +{champ_ps['mean'] - all_ps['mean']:.4f}")

    # ── Step 7: Save stats JSON ───────────────────────────────────────────
    save_profile_stats(stats, STATS_JSON)
    print(f"\n{SEP}")
    print(f"  Saved → {STATS_JSON.name}")
    print(SEP)


if __name__ == "__main__":
    main()
