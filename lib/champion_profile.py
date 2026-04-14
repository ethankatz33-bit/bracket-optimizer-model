"""
lib/champion_profile.py
Champion profile feature layer.

Provides:
  - compute_profile_scores_df(df)          → adds profile_score column [0,1]
  - identify_round_teams(results, start, end) → {year: {round_name: [team_id]}}
  - build_champion_stats(hist_df, round_teams) → stats dict for analysis
  - save_profile_stats / load_profile_stats

profile_score
-------------
  Measures how closely a team's pre-tournament tournament history matches
  historical champion traits.

  For each feature, the team's within-season percentile is computed:
    - Higher is better features (offense, efficiency, form, sos):
        team at 90th percentile → score contribution 0.90
    - Lower is better features (defense_rating):
        team at lowest allowed in season → 1.00; at highest allowed → 0.00

  Score = weighted sum of within-season percentiles.
  Teams with no prior history receive 0.5 (population mean).

  Champions historically average ~0.86–0.90 profile_score.
  Average tournament teams average ~0.50.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT        = Path(__file__).parent.parent
PROFILE_STATS_PATH  = PROJECT_ROOT / "data" / "processed" / "champion_profile_stats.json"

# ── Recency weighting ─────────────────────────────────────────────────────────
# Mirrors the weights used in lib/backtest.py.
# Applied when computing champion profile baselines so recent seasons have
# stronger influence on what "champion-quality" looks like.
#
#   2015–2025  → 1.00  (modern era — full weight)
#   2005–2014  → 0.70  (mid-era)
#   1990–2004  → 0.40  (early era)
def _season_weight(year: int) -> float:
    if year >= 2015:
        return 1.0
    if year >= 2005:
        return 0.7
    return 0.4


def _wavg(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean."""
    return float(np.average(values, weights=weights))


def _wstd(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted standard deviation (population)."""
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def _wquantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile via cumulative-weight interpolation."""
    idx    = np.argsort(values)
    sv, sw = values[idx], weights[idx]
    cdf    = np.cumsum(sw) / sw.sum()
    return float(np.interp(q, cdf, sv))

_DAYNUM_TO_ROUND: dict[int, int] = {
    134: 0, 135: 0,    # Play-in
    136: 1, 137: 1,    # Round of 64
    138: 2, 139: 2,    # Round of 32
    143: 3, 144: 3,    # Sweet 16
    145: 4, 146: 4,    # Elite 8 (winners enter Final Four)
    152: 5,            # Final Four (winners enter Championship)
    154: 6,            # Championship
}

# Features used in profile_score.
# Tuple: (direction, weight)
#   direction="higher" → ascending rank → highest value = percentile 1.0
#   direction="lower"  → descending rank → lowest value = percentile 1.0
PROFILE_FEATURES: dict[str, tuple[str, float]] = {
    "offense_rating":       ("higher", 0.30),
    "efficiency_margin":    ("higher", 0.30),
    "defense_rating":       ("lower",  0.25),   # fewer points allowed = better
    "recent_form":          ("higher", 0.10),
    "strength_of_schedule": ("higher", 0.05),
    # ap_top12_flag / ap_rank_week6 excluded: AP is restricted to Final Four /
    # Championship upset scoring in team_selector.py; not used in early rounds.
}

_TOTAL_PROFILE_WEIGHT: float = sum(w for _, w in PROFILE_FEATURES.values())


# ════════════════════════════════════════════════════════════════════════════
# Profile score computation
# ════════════════════════════════════════════════════════════════════════════

def compute_profile_scores_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a profile_score column [0,1] to the DataFrame.

    Score = weighted sum of within-season percentile ranks for key features.
    Missing features reduce the available weight; score is renormalized.
    Cold-start teams (all features NaN) receive 0.5.

    Parameters
    ----------
    df : DataFrame with 'season' column and at least some PROFILE_FEATURES columns.

    Returns
    -------
    Copy of df with 'profile_score' column added.
    """
    df = df.copy()

    score_sum  = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)

    for feat, (direction, weight) in PROFILE_FEATURES.items():
        if feat not in df.columns:
            continue

        # ascending=True  → highest value gets rank n → pct 1.0 ("higher is better")
        # ascending=False → lowest  value gets rank n → pct 1.0 ("lower is better")
        ascending = (direction == "higher")

        pct = (
            df.groupby("season")[feat]
            .rank(ascending=ascending, pct=True, na_option="keep")
            .fillna(float("nan"))
        )

        has_val = df[feat].notna()
        score_sum  += (pct * weight).fillna(0.0)
        weight_sum += has_val.astype(float) * weight

    # Renormalize: scale partial scores to the full [0,1] range.
    # Teams with weight_sum == 0 (all features NaN) default to 0.5.
    profile = (score_sum / weight_sum * _TOTAL_PROFILE_WEIGHT).where(
        weight_sum > 0, 0.5
    )
    df["profile_score"] = profile.clip(0.0, 1.0).round(4)
    return df


# ════════════════════════════════════════════════════════════════════════════
# Historical round-team identification
# ════════════════════════════════════════════════════════════════════════════

def identify_round_teams(
    results_path: str | Path,
    start_year:   int = 1990,
    end_year:     int = 2016,
) -> dict[int, dict[str, list[int]]]:
    """
    Identify which team_ids reached each round in each season.

    Returns
    -------
    {
      year: {
        "champion":    [team_id],
        "title_game":  [team_id, team_id],  # 2 teams (winner + runner-up)
        "final_four":  [team_id, ...],       # 4 teams
        "elite_8":     [team_id, ...],       # 8 teams
      }
    }

    Definitions
    -----------
    champion   = winner of round-6 game (Daynum 154)
    title_game = both participants in round-6 game (Daynum 154) = 2 teams
    final_four = all participants in round-5 games (Daynum 152) = 4 teams
    elite_8    = all participants in round-4 games (Daynum 145/146) = 8 teams
    """
    results = pd.read_csv(results_path)
    results["round"] = results["Daynum"].map(_DAYNUM_TO_ROUND)
    results = results[
        (results["Season"] >= start_year) & (results["Season"] <= end_year)
    ]

    round_teams: dict[int, dict[str, list[int]]] = {}

    for year in sorted(results["Season"].unique()):
        yr = results[results["Season"] == year]

        champ_game = yr[yr["round"] == 6]
        ff_games   = yr[yr["round"] == 5]
        e8_games   = yr[yr["round"] == 4]

        champion   = list(champ_game["Wteam"].astype(int))
        title_game = sorted(set(
            list(champ_game["Wteam"].astype(int)) +
            list(champ_game["Lteam"].astype(int))
        ))
        final_four = sorted(set(
            list(ff_games["Wteam"].astype(int)) +
            list(ff_games["Lteam"].astype(int))
        ))
        elite_8    = sorted(set(
            list(e8_games["Wteam"].astype(int)) +
            list(e8_games["Lteam"].astype(int))
        ))

        round_teams[int(year)] = {
            "champion":   champion,
            "title_game": title_game,
            "final_four": final_four,
            "elite_8":    elite_8,
        }

    return round_teams


# ════════════════════════════════════════════════════════════════════════════
# Champion stats aggregation
# ════════════════════════════════════════════════════════════════════════════

def _collect_group(
    hist_df:     pd.DataFrame,
    round_teams: dict[int, dict[str, list[int]]],
    group_key:   str,   # "champion" | "final_four" | "elite_8"
) -> pd.DataFrame:
    """Extract rows from hist_df for all teams in the given group across years."""
    rows = []
    for year, groups in round_teams.items():
        for tid in groups.get(group_key, []):
            row = hist_df[(hist_df["season"] == year) & (hist_df["team_id"] == tid)]
            if not row.empty:
                rows.append(row.iloc[0])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_champion_stats(
    hist_df:     pd.DataFrame,
    round_teams: dict[int, dict[str, list[int]]],
) -> dict:
    """
    Aggregate statistics for champions, Final Four, Elite 8, and all teams.

    Returns a nested dict suitable for JSON serialization and display.
    """
    groups = {
        "champion":   _collect_group(hist_df, round_teams, "champion"),
        "title_game": _collect_group(hist_df, round_teams, "title_game"),
        "final_four": _collect_group(hist_df, round_teams, "final_four"),
        "elite_8":    _collect_group(hist_df, round_teams, "elite_8"),
        "all_teams":  hist_df,
    }

    numeric_feats  = list(PROFILE_FEATURES.keys())
    flag_feats     = ["is_ap_top12_week6", "is_top10_kenpom_torvik", "is_top10_offense"]
    profile_cols   = [f + "_pct" for f in numeric_feats]

    # Precompute per-row season weights on the full hist_df so group subsets can
    # slice the same array.  Shape: (len(hist_df),) aligned to hist_df.index.
    all_weights_s = hist_df["season"].map(_season_weight)

    stats: dict[str, dict] = {}

    for label, gdf in groups.items():
        if gdf.empty:
            stats[label] = {}
            continue

        # Recency weights for this group's rows
        gw = all_weights_s.loc[gdf.index].values

        entry: dict = {"n": len(gdf)}

        # Raw feature stats (recency-weighted)
        feat_stats: dict[str, dict] = {}
        for feat in numeric_feats:
            if feat not in gdf.columns:
                continue
            mask = gdf[feat].notna()
            vals = gdf.loc[mask, feat].values.astype(float)
            wts  = gw[mask.values]
            if len(vals) == 0:
                continue
            feat_stats[feat] = {
                "mean": round(_wavg(vals, wts), 2),
                "std":  round(_wstd(vals, wts), 2),
                "p25":  round(_wquantile(vals, wts, 0.25), 2),
                "p75":  round(_wquantile(vals, wts, 0.75), 2),
            }
        entry["features"] = feat_stats

        # Within-season percentile means (recency-weighted)
        pct_means: dict[str, float] = {}
        for col in profile_cols:
            feat = col.replace("_pct", "")
            if feat not in gdf.columns:
                continue
            direction = PROFILE_FEATURES[feat][0]
            ascending = (direction == "higher")
            pct = hist_df.groupby("season")[feat].rank(
                ascending=ascending, pct=True, na_option="keep"
            )
            group_mask = hist_df.index.isin(gdf.index)
            pct_vals   = pct[group_mask]
            valid      = pct_vals.notna()
            if valid.sum() > 0:
                pct_means[feat] = round(
                    float(np.average(
                        pct_vals[valid].values,
                        weights=all_weights_s.loc[pct_vals[valid].index].values,
                    )),
                    3,
                )
        entry["percentile_means"] = pct_means

        # Flag prevalence (recency-weighted)
        flag_stats: dict[str, float] = {}
        for flag in flag_feats:
            if flag not in gdf.columns:
                continue
            flag_vals = gdf[flag].values.astype(float)
            flag_stats[flag] = round(float(np.average(flag_vals, weights=gw)), 3)
        entry["flags"] = flag_stats

        # Profile score (recency-weighted)
        if "profile_score" in gdf.columns:
            mask = gdf["profile_score"].notna()
            ps   = gdf.loc[mask, "profile_score"].values.astype(float)
            pw   = gw[mask.values]
            entry["profile_score"] = {
                "mean": round(_wavg(ps, pw), 4),
                "std":  round(_wstd(ps, pw), 4),
                "p25":  round(_wquantile(ps, pw, 0.25), 4),
                "p75":  round(_wquantile(ps, pw, 0.75), 4),
            }

        stats[label] = entry

    return stats


# ════════════════════════════════════════════════════════════════════════════
# Persistence
# ════════════════════════════════════════════════════════════════════════════

def save_profile_stats(
    stats: dict,
    path:  str | Path = PROFILE_STATS_PATH,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_profile_stats(
    path: str | Path = PROFILE_STATS_PATH,
) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
