"""
lib/team_ratings.py
Team rating and win probability model — Version 1.

Pipeline
--------
  1. Load team stats CSV (handles missing columns gracefully)
  2. Normalize each feature to z-scores within the season
  3. Compute weighted composite  →  team_rating      (game picks, upset picks)
  4. Compute champion profile    →  champion_profile_score  (deep run / champion)
  5. Win probability via logistic transform on z-score difference
  6. Save to CSV + JSON

Public API
----------
  load_team_data(path)                         → raw DataFrame
  rate_teams(df)                               → df + rating columns
  rate_teams_by_season(df)                     → rate each season independently
  predict_win_probability(team_a, team_b)      → {team_a: float, team_b: float}
  load_team_ratings(path)                      → pre-saved ratings DataFrame
  get_team_rating(df, team_name, season)       → single team dict or None
  save_team_ratings(df)                        → writes CSV + JSON
  build_and_save(input_path)                   → full pipeline convenience wrapper

Integration hooks (for team_selector.py / backtest)
------------------------------------------------------
  - Replace MOCK_TEAMS ratings with values from load_team_ratings()
  - Pass team_rating as the "rating" field in team dicts
  - Pass champion_profile_score to score_team() champ_profile term
  - Use predict_win_probability() in _upset_desirability() for real EV
"""

import json
import math
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).parent.parent
DEFAULT_INPUT  = PROJECT_ROOT / "data" / "processed" / "team_stats.csv"
OUTPUT_CSV     = PROJECT_ROOT / "data" / "processed" / "team_ratings.csv"
OUTPUT_JSON    = PROJECT_ROOT / "data" / "processed" / "team_ratings.json"
HIST_CSV       = PROJECT_ROOT / "data" / "processed" / "historical_team_ratings.csv"

# ── Feature weights ───────────────────────────────────────────────────────────

# Composite team quality — used for round-by-round game picks and upset EV.
# All weights must sum to 1.0.  Missing columns are dropped and the remaining
# weights are renormalized automatically.
TEAM_RATING_WEIGHTS: dict[str, float] = {
    "efficiency_margin":    0.30,   # net points per 100 poss; higher = better
    "offense_rating":       0.20,   # off. efficiency; higher = better
    "defense_rating":       0.20,   # def. efficiency; LOWER = better (inverted)
    "recent_form":          0.15,   # recent win rate or momentum; higher = better
    "strength_of_schedule": 0.10,   # quality of opponents; higher = better
    "seed_score":           0.05,   # derived: 17 - seed; higher = better
}

# Champion profile — used for Final Four / championship picks.
# Balanced modern champion profile: offense and defense weighted equally to
# reflect the modern game, while maintaining strong emphasis on efficiency margin.
CHAMPION_PROFILE_WEIGHTS: dict[str, float] = {
    "seed_score":        0.25,   # lower seed → deeper historical run rate
    "defense_rating":    0.21,   # inverted; equal weight with offense
    "offense_rating":    0.21,
    "efficiency_margin": 0.20,   # raised: net efficiency is highly predictive
    "recent_form":       0.09,   # lowered: peak performance matters more than recency
    "ap_top12_flag":     0.04,   # minor signal: 1 if team was AP top-12 at week 6
    # ap_rank_week6 removed — early-January ranking too noisy as a predictive signal
}

# Features where the raw value is inverted before z-scoring:
#   defense_rating : fewer points allowed = better defense
INVERT_FEATURES: frozenset[str] = frozenset({"defense_rating"})

# Logistic steepness constant — calibrated for z-score scale (roughly −3 to +3).
# k=1.2 gives:
#   z-diff 4.0  → ~99%  (seed 1 vs seed 16)
#   z-diff 2.0  → ~92%  (strong favorite)
#   z-diff 1.0  → ~77%  (moderate favorite)
#   z-diff 0.5  → ~65%  (slight edge)
WIN_PROB_K: float = 1.2


# ════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════

def _zscore(series: pd.Series, fill_na: float = 0.0) -> pd.Series:
    """
    Z-score normalize a Series.

    - Mean and std are computed from non-null values only.
    - NaN entries are filled with `fill_na` after scaling (default 0.0 = mean).
    - Returns all zeros when std == 0 (constant or single-value column).
    """
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(0.0, index=series.index, dtype=float)
    mean = valid.mean()
    std  = valid.std()
    if std == 0.0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return ((series - mean) / std).fillna(fill_na)


def _minmax(series: pd.Series) -> pd.Series:
    """
    Min-max scale a Series to [0, 1].
    All-equal values → 0.5 (neutral, not zero).
    """
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (series - lo) / (hi - lo)


def _build_feature_matrix(
    df: pd.DataFrame,
    weights: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Build a z-scored feature matrix for the given weight dict.

    Handles:
      - Derived column "seed_score" = 17 - seed
      - Missing columns are silently skipped
      - Inverted features (INVERT_FEATURES) are negated before z-scoring
      - ap_rank_week6 NaN entries treated as rank 26 (just outside top 25)
      - Remaining weights renormalized to sum to 1.0

    Returns
    -------
    feat_df : DataFrame, one column per available feature (z-scored)
    used_w  : weight dict for available features, normalized to sum 1.0
    """
    feat_cols: dict[str, pd.Series] = {}
    used_w:    dict[str, float]     = {}

    for feature, weight in weights.items():
        # ── Derive or fetch the raw series ────────────────────────────────
        if feature == "seed_score":
            if "seed" not in df.columns:
                continue
            raw = (17.0 - df["seed"].astype(float))

        elif feature == "ap_rank_week6":
            if "ap_rank_week6" not in df.columns:
                continue
            raw = df["ap_rank_week6"].astype(float)
            # Treat unranked teams as rank 26 before inversion
            raw = raw.fillna(26.0)

        else:
            if feature not in df.columns:
                continue
            raw = df[feature].astype(float)

        # ── Invert features where lower raw value = better ─────────────────
        if feature in INVERT_FEATURES:
            raw = -raw

        feat_cols[feature] = _zscore(raw)
        used_w[feature]    = weight

    # Renormalize weights so they sum to 1.0 over available features
    total_w = sum(used_w.values())
    if total_w > 0:
        used_w = {k: v / total_w for k, v in used_w.items()}

    return pd.DataFrame(feat_cols, index=df.index), used_w


# ════════════════════════════════════════════════════════════════════════════
# Core rating functions
# ════════════════════════════════════════════════════════════════════════════

def rate_teams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team_rating and champion_profile_score for all teams in df.

    Both scores are computed as weighted sums of per-feature z-scores,
    then min-max scaled to [0, 1] for interpretability.

    The raw z-score composites are stored in private columns:
      _rating_zscore    — used by predict_win_probability (larger scale → better separation)
      _champion_zscore  — same, for champion profile

    Parameters
    ----------
    df : DataFrame with columns including season, team_name, seed.
         Additional stat columns are used when present; missing ones are
         dropped and their weights redistributed.

    Returns
    -------
    Copy of df with four added columns:
      team_rating             float [0, 1]
      champion_profile_score  float [0, 1]
      _rating_zscore          float (unbounded)
      _champion_zscore        float (unbounded)
    """
    df = df.copy()

    # ── Team rating ───────────────────────────────────────────────────────
    feat_df, used_w = _build_feature_matrix(df, TEAM_RATING_WEIGHTS)

    zscore_sum = pd.Series(0.0, index=df.index)
    for feature, w in used_w.items():
        zscore_sum += feat_df[feature] * w

    df["_rating_zscore"] = zscore_sum
    df["team_rating"]    = _minmax(zscore_sum)

    # ── Champion profile ──────────────────────────────────────────────────
    champ_feat_df, champ_w = _build_feature_matrix(df, CHAMPION_PROFILE_WEIGHTS)

    champ_sum = pd.Series(0.0, index=df.index)
    for feature, w in champ_w.items():
        champ_sum += champ_feat_df[feature] * w

    df["_champion_zscore"]      = champ_sum
    df["champion_profile_score"] = _minmax(champ_sum)

    return df


def rate_teams_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run rate_teams independently for each season in df.

    Z-scores are computed within each season so that a seed-1 team in 2010
    is rated relative to its own tournament field, not cross-year.

    Parameters
    ----------
    df : DataFrame with a 'season' column

    Returns
    -------
    Concatenated DataFrame with rating columns added, one block per season.
    """
    parts = []
    for season in sorted(df["season"].unique()):
        subset = df[df["season"] == season].copy()
        parts.append(rate_teams(subset))
    return pd.concat(parts, ignore_index=True)


# ════════════════════════════════════════════════════════════════════════════
# Win probability
# ════════════════════════════════════════════════════════════════════════════

def predict_win_probability(
    team_a: dict | pd.Series,
    team_b: dict | pd.Series,
    k: float = WIN_PROB_K,
) -> dict[str, float]:
    """
    Compute head-to-head win probability using a logistic transform.

    Formula:
      prob_a = 1 / (1 + exp(-k * (score_a - score_b)))

    Uses _rating_zscore when available (better discrimination at extreme
    matchups).  Falls back to team_rating [0, 1] with k rescaled to k*5
    to produce similar probability ranges.

    Parameters
    ----------
    team_a, team_b : dicts or pd.Series with at least 'team_rating'.
                     Including '_rating_zscore' improves accuracy.
    k              : logistic steepness (default 1.2 for z-score inputs)

    Returns
    -------
    {"team_a": float, "team_b": float}  — probabilities that sum to 1.0

    Examples
    --------
    >>> pa = {"team_name": "Duke", "_rating_zscore": 1.8, "team_rating": 0.9}
    >>> pb = {"team_name": "Norfolk St", "_rating_zscore": -2.1, "team_rating": 0.1}
    >>> predict_win_probability(pa, pb)
    {'team_a': 0.9896, 'team_b': 0.0104}
    """
    # Prefer z-score for larger, more discriminating differences
    if "_rating_zscore" in team_a and "_rating_zscore" in team_b:
        score_a = float(team_a["_rating_zscore"])
        score_b = float(team_b["_rating_zscore"])
        effective_k = k
    else:
        # [0, 1] scale: differences are ~5-6× smaller; rescale k accordingly
        score_a     = float(team_a.get("team_rating", 0.5))
        score_b     = float(team_b.get("team_rating", 0.5))
        effective_k = k * 5.0

    diff   = score_a - score_b
    prob_a = 1.0 / (1.0 + math.exp(-effective_k * diff))

    return {
        "team_a": round(prob_a, 4),
        "team_b": round(1.0 - prob_a, 4),
    }


# ════════════════════════════════════════════════════════════════════════════
# I/O
# ════════════════════════════════════════════════════════════════════════════

def load_team_data(path: str | Path) -> pd.DataFrame:
    """
    Load raw team stats CSV.

    Required columns : season, team_name, seed
    Soft required    : efficiency_margin, offense_rating, defense_rating,
                       recent_form, strength_of_schedule
    Optional         : ap_rank_week6

    Missing soft-required columns are silently handled; the rating model
    uses available features with renormalized weights.

    Raises
    ------
    ValueError  if required columns are absent
    FileNotFoundError  if path does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Team stats file not found: {path}")

    df = pd.read_csv(path)

    # Accept either canonical_team_name or team_name as the identity column.
    # If only canonical_team_name is present, alias it to team_name so all
    # downstream rating and simulation code continues to work unchanged.
    if "canonical_team_name" in df.columns and "team_name" not in df.columns:
        df = df.rename(columns={"canonical_team_name": "team_name"})
    elif "canonical_team_name" in df.columns:
        # Both present — keep team_name; canonical_team_name is extra context
        pass

    required = {"season", "team_name", "seed"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}\n"
            f"Found columns: {sorted(df.columns.tolist())}"
        )

    # Coerce types
    df["season"] = df["season"].astype(int)
    df["seed"]   = df["seed"].astype(int)

    return df


def save_team_ratings(
    df:        pd.DataFrame,
    csv_path:  Path = OUTPUT_CSV,
    json_path: Path = OUTPUT_JSON,
) -> None:
    """
    Write team ratings to CSV and JSON.

    Output columns: season, team_name, seed, team_rating, champion_profile_score
    Sorted by season ascending, team_rating descending within each season.
    """
    keep = ["season", "team_name", "seed", "team_rating", "champion_profile_score"]
    keep = [c for c in keep if c in df.columns]

    out = (
        df[keep]
        .copy()
        .sort_values(["season", "team_rating"], ascending=[True, False])
        .round({"team_rating": 4, "champion_profile_score": 4})
        .reset_index(drop=True)
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)

    records = out.to_dict(orient="records")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)


def load_team_ratings(path: str | Path = OUTPUT_CSV) -> pd.DataFrame:
    """
    Load pre-computed team ratings from CSV.

    This is the primary integration hook for team_selector.py and backtest.py.
    After calling this, use get_team_rating() to look up individual teams.
    """
    return pd.read_csv(path)


def get_team_rating(
    df:        pd.DataFrame,
    team_name: str,
    season:    int,
) -> dict | None:
    """
    Look up a single team's ratings by name and season.

    Returns a dict with all columns (including team_rating and
    champion_profile_score) or None if the team is not found.
    """
    mask = (df["team_name"] == team_name) & (df["season"] == season)
    rows = df[mask]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


# ════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ════════════════════════════════════════════════════════════════════════════

def build_and_save(input_path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """
    Full pipeline: load → rate by season → save → return rated DataFrame.

    This is the main entry point for scripts/build_team_ratings.py.
    """
    df   = load_team_data(input_path)
    rated = rate_teams_by_season(df)
    save_team_ratings(rated)
    return rated


# ════════════════════════════════════════════════════════════════════════════
# Historical ratings — backtest bridge
# ════════════════════════════════════════════════════════════════════════════

def load_historical_ratings(path: str | Path = HIST_CSV) -> pd.DataFrame | None:
    """
    Load pre-computed historical team ratings from CSV.

    Returns None if the file does not exist (build it first with
    scripts/build_historical_ratings.py).

    Columns: season, team_id, team_name, seed, offense_rating,
             defense_rating, efficiency_margin, recent_form,
             strength_of_schedule, team_rating, champion_profile_score
    """
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "team_id" in df.columns:
        df["team_id"] = df["team_id"].astype(int)
    return df


def get_team_rating_by_id(
    df:      pd.DataFrame,
    team_id: int,
    season:  int,
) -> dict | None:
    """
    Look up a team's ratings by numeric team_id and season.

    Used for backtest teams whose names follow the "T{id}" pattern.
    Returns None if team_id is not found or 'team_id' column is absent.
    """
    if "team_id" not in df.columns:
        return None
    mask = (df["team_id"] == team_id) & (df["season"] == season)
    rows = df[mask]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()
