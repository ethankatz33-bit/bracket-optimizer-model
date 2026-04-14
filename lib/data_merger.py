"""
lib/data_merger.py
External dataset loading, normalization, and team-ID mapping.

Responsibilities
----------------
  - normalize_name(s)          → canonical lowercase string
  - load_cbb(path)             → cleaned DataFrame from cbb.csv
  - load_kenpom_torvik(path)   → cleaned DataFrame (optional)
  - load_ap_week6(path)        → cleaned DataFrame (optional)
  - build_team_id_map(...)     → {(year, cbb_team_name): kaggle_team_id}
  - build_merged_stats(...)    → merged DataFrame ready for output

Team-ID mapping strategy
------------------------
Stage 1 — Tournament path (definitive):
    Key = (season, seed, round_reached, is_champion)
    Unique when no other same-seed team in that season reached the same round.

Stage 2 — Rank-based within ambiguous groups:
    For groups of N cbb teams and N Kaggle teams sharing the same
    (season, seed, round_reached), match by sorted offensive efficiency rank.
    cbb: rank by ADJOE descending  ↔  Kaggle: rank by proxy offense_rating descending.
    This assumes within-group rank ordering is consistent across the two datasets.
    Matches are flagged RANK_MATCH to distinguish from definitive ones.

Stage 3 — Unmatched:
    Any remaining teams where the group sizes differ (data inconsistency).
    These keep all cbb features but have no team_id.
"""

import re
from pathlib import Path

import pandas as pd

# ── Name normalization ────────────────────────────────────────────────────────

# Explicit alias overrides applied AFTER basic normalization.
# Keys and values are already lowercased / stripped.
_ALIASES: dict[str, str] = {
    # "St." suffix → "state"
    "michigan st":          "michigan state",
    "ohio st":              "ohio state",
    "penn st":              "penn state",
    "florida st":           "florida state",
    "kansas st":            "kansas state",
    "iowa st":              "iowa state",
    "mississippi st":       "mississippi state",
    "oklahoma st":          "oklahoma state",
    "arizona st":           "arizona state",
    "washington st":        "washington state",
    "colorado st":          "colorado state",
    "utah st":              "utah state",
    "oregon st":            "oregon state",
    "nc st":                "north carolina state",
    "nc state":             "north carolina state",
    # Common initialisms
    "vcu":                  "virginia commonwealth",
    "smu":                  "southern methodist",
    "tcu":                  "texas christian",
    "byu":                  "brigham young",
    "lsu":                  "louisiana state",
    "usc":                  "southern california",
    "ucla":                 "california los angeles",
    "uconn":                "connecticut",
    "unlv":                 "nevada las vegas",
    "uncw":                 "north carolina wilmington",
    "unc":                  "north carolina",
    "utep":                 "texas el paso",
    "utsa":                 "texas san antonio",
    "uab":                  "alabama birmingham",
    "ucf":                  "central florida",
    "fiu":                  "florida international",
    "fau":                  "florida atlantic",
    "ole miss":             "mississippi",
    "miami fl":             "miami florida",
    "miami oh":             "miami ohio",
    "uc irvine":            "california irvine",
    "uc santa barbara":     "california santa barbara",
    "uc davis":             "california davis",
    "cal poly":             "california poly",
    "saint marys":          "saint marys",
    "st marys":             "saint marys",
    "st johns":             "saint johns",
    "saint josephs":        "saint josephs",
    "st josephs":           "saint josephs",
    "saint peters":         "saint peters",
    "st peters":            "saint peters",
    "saint louis":          "saint louis",
    "st louis":             "saint louis",
    "saint francis":        "saint francis",
    "st francis":           "saint francis",
    "stephen f austin":     "stephen f austin",
    "sfa":                  "stephen f austin",
    "texas am":             "texas am",
    "texas a&m":            "texas am",
    "middle tennessee":     "middle tennessee state",
    "middle tenn":          "middle tennessee state",
    "western ky":           "western kentucky",
    "eastern ky":           "eastern kentucky",
    "northern ky":          "northern kentucky",
    "southern miss":        "southern mississippi",
    "southern illinois":    "southern illinois",
    "loyola md":            "loyola maryland",
    "loyola il":            "loyola chicago",
    "loyola chicago":       "loyola chicago",
}


def normalize_name(raw: str) -> str:
    """
    Return a canonical lowercase string for team name matching.

    Steps:
      1. Lowercase + strip
      2. Remove trailing period-abbreviations (e.g., "St." → "St")
      3. Remove punctuation except hyphens (keep "mid-american" intact)
      4. Collapse whitespace
      5. Apply explicit alias overrides
    """
    if not isinstance(raw, str):
        return ""
    s = raw.lower().strip()
    # Remove trailing dot from abbreviations like "St." "Jr."
    s = re.sub(r"\.(?=\s|$)", "", s)
    # Remove possessives and other punctuation except hyphen and space
    s = re.sub(r"[''&,()]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _ALIASES.get(s, s)


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_cbb(path: str | Path) -> pd.DataFrame:
    """
    Load cbb.csv and return only tournament teams with standardized columns.

    Output columns:
        season, team_name_raw, team_name, seed, postseason,
        adjoe, adjde, efficiency_margin, barthag, wins, games
    """
    df = pd.read_csv(path)
    # Keep only tournament teams
    df = df[df["SEED"].notna()].copy()
    df["SEED"] = df["SEED"].astype(int)

    # Postseason R68 = play-in round (round 0); keep but mark
    df = df.rename(columns={
        "YEAR":       "season",
        "TEAM":       "team_name_raw",
        "SEED":       "seed",
        "POSTSEASON": "postseason",
        "ADJOE":      "adjoe",
        "ADJDE":      "adjde",
        "BARTHAG":    "barthag",
        "W":          "wins",
        "G":          "games",
    })

    df["team_name"]        = df["team_name_raw"].apply(normalize_name)
    df["efficiency_margin"] = (df["adjoe"] - df["adjde"]).round(2)

    keep = [
        "season", "team_name_raw", "team_name", "seed", "postseason",
        "adjoe", "adjde", "efficiency_margin", "barthag", "wins", "games",
    ]
    return df[keep].reset_index(drop=True)


def load_kenpom_torvik(path: str | Path) -> pd.DataFrame | None:
    """
    Load KenPom/Torvik CSV if present.

    Expected columns (flexible): season/year, team, kenpom_rank or torvik_rank,
    adjoe, adjde.  Returns None if file not found.
    """
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    col_map: dict[str, str] = {}
    for col in df.columns:
        if col in ("year", "season"):
            col_map[col] = "season"
        elif col in ("team", "team_name", "school"):
            col_map[col] = "team_name_raw"
        elif "adjoe" in col or "off_eff" in col:
            col_map[col] = "adjoe_kt"
        elif "adjde" in col or "def_eff" in col:
            col_map[col] = "adjde_kt"
        elif "rank" in col or "rating" in col or "barthag" in col:
            col_map[col] = "kt_rating"
    df = df.rename(columns=col_map)

    if "team_name_raw" in df.columns:
        df["team_name"] = df["team_name_raw"].apply(normalize_name)
    return df


def load_ap_week6(path: str | Path) -> pd.DataFrame | None:
    """
    Load AP Week 6 top-12 list CSV if present.

    The file is a sparse list: only teams that were in the AP top 12 at
    Week 6 are listed.  Presence of a (season, team_name) row is sufficient
    to set ap_top12_flag = 1 — no rank column is required.

    Required columns : season (or year), team_name (or team / school)
    Optional columns : ap_rank_week6 (or any column containing "rank")

    Returns None if file not found or if the file has no data rows.
    """
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df.columns = [c.lower().strip() for c in df.columns]

    col_map: dict[str, str] = {}
    for col in df.columns:
        if col in ("year", "season"):
            col_map[col] = "season"
        elif col in ("team", "team_name", "school"):
            col_map[col] = "team_name_raw"
        elif "rank" in col:
            col_map[col] = "ap_rank"
    df = df.rename(columns=col_map)

    required = {"season", "team_name_raw"}
    if not required.issubset(df.columns):
        return None

    df["season"]    = df["season"].astype(int)
    df["team_name"] = df["team_name_raw"].apply(normalize_name)
    return df


# ── Tournament path helpers ───────────────────────────────────────────────────

_POSTSEASON_TO_ROUND: dict[str, int] = {
    "Champions": 6,
    "2ND":       6,
    "F4":        5,
    "E8":        4,
    "S16":       3,
    "R32":       2,
    "R64":       1,
    "R68":       0,
}


def _build_kaggle_paths(
    results_path: str | Path,
    seeds_path:   str | Path,
) -> pd.DataFrame:
    """
    Build a DataFrame of (season, team_id, seed, max_round, is_champion)
    for all tournament teams in the Kaggle dataset.
    """
    results = pd.read_csv(results_path)
    seeds   = pd.read_csv(seeds_path)
    seeds["seed_num"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)
    seeds = seeds.rename(columns={"Season": "season", "Team": "team_id"})

    # Map Daynum → round
    daynum_round = {
        134: 0, 135: 0,
        136: 1, 137: 1,
        138: 2, 139: 2,
        143: 3, 144: 3,
        145: 4, 146: 4,
        152: 5,
        154: 6,
    }
    results["round"] = results["Daynum"].map(daynum_round)
    results = results[results["round"].notna()]

    rows = []
    for _, g in results.iterrows():
        r = int(g["round"])
        rows.append({"season": int(g["Season"]), "team_id": int(g["Wteam"]),
                     "round": r, "won": True})
        rows.append({"season": int(g["Season"]), "team_id": int(g["Lteam"]),
                     "round": r, "won": False})

    game_df = pd.DataFrame(rows)

    def _summarize(grp: pd.DataFrame) -> pd.Series:
        mr = grp["round"].max()
        champ = bool(((grp["round"] == 6) & grp["won"]).any())
        return pd.Series({"max_round": mr, "is_champion": champ})

    paths = game_df.groupby(["season", "team_id"]).apply(
        _summarize, include_groups=False
    ).reset_index()
    paths = paths.merge(seeds[["season", "team_id", "seed_num"]], on=["season", "team_id"])
    return paths


# ── Main mapping builder ──────────────────────────────────────────────────────

def build_team_id_map(
    cbb_df:       pd.DataFrame,
    results_path: str | Path,
    seeds_path:   str | Path,
    hist_path:    str | Path | None = None,
) -> tuple[dict[tuple, int], dict[tuple, str], list[str]]:
    """
    Map (season, cbb_team_name) → kaggle_team_id.

    Returns
    -------
    confirmed : {(season, team_name): team_id}   — definitive path matches
    estimated : {(season, team_name): team_id}   — rank-based matches (uncertain)
    unmatched : list of "season|team_name_raw"   — could not resolve
    """
    kaggle_paths = _build_kaggle_paths(results_path, seeds_path)

    # Optionally load proxy features for rank-based disambiguation
    hist_df: pd.DataFrame | None = None
    if hist_path and Path(hist_path).exists():
        hist_df = pd.read_csv(hist_path)

    confirmed: dict[tuple, int] = {}
    estimated: dict[tuple, int] = {}
    unmatched: list[str]        = []

    # Process one season at a time
    for season, season_cbb in cbb_df.groupby("season"):
        season_kp = kaggle_paths[kaggle_paths["season"] == season]
        if season_kp.empty:
            for _, row in season_cbb.iterrows():
                unmatched.append(f"{season}|{row['team_name_raw']}")
            continue

        for postseason, ps_group_cbb in season_cbb.groupby("postseason"):
            round_reached = _POSTSEASON_TO_ROUND.get(postseason)
            if round_reached is None:
                for _, row in ps_group_cbb.iterrows():
                    unmatched.append(f"{season}|{row['team_name_raw']}")
                continue

            # All cbb teams in this (season, postseason) bucket
            cbb_rows = ps_group_cbb.copy()

            for seed_val, seed_group_cbb in cbb_rows.groupby("seed"):
                # Matching Kaggle teams: same season, seed, round
                kp_cands = season_kp[
                    (season_kp["seed_num"] == seed_val) &
                    (season_kp["max_round"] == round_reached)
                ].copy()

                # For round 6: distinguish champion (won) from runner-up (lost)
                if round_reached == 6:
                    is_champ = (postseason == "Champions")
                    kp_cands = kp_cands[kp_cands["is_champion"] == is_champ]

                n_cbb = len(seed_group_cbb)
                n_kp  = len(kp_cands)

                if n_cbb == 0 or n_kp == 0:
                    for _, row in seed_group_cbb.iterrows():
                        unmatched.append(f"{season}|{row['team_name_raw']}")
                    continue

                if n_cbb != n_kp:
                    # Size mismatch — can't resolve
                    for _, row in seed_group_cbb.iterrows():
                        unmatched.append(
                            f"{season}|{row['team_name_raw']} "
                            f"(group size: cbb={n_cbb}, kaggle={n_kp})"
                        )
                    continue

                if n_cbb == 1:
                    # ── Stage 1: definitive path match ──────────────────
                    tid = int(kp_cands.iloc[0]["team_id"])
                    key = (int(season), seed_group_cbb.iloc[0]["team_name"])
                    confirmed[key] = tid
                else:
                    # ── Stage 2: rank-based disambiguation ───────────────
                    # Sort cbb group by adjoe descending (best offense first)
                    cbb_sorted = seed_group_cbb.sort_values(
                        "adjoe", ascending=False
                    ).reset_index(drop=True)

                    # Sort Kaggle group by proxy offense descending (if available)
                    if hist_df is not None:
                        kp_with_feat = kp_cands.merge(
                            hist_df[["season", "team_id", "offense_rating"]],
                            on=["season", "team_id"],
                            how="left",
                        )
                        kp_sorted = kp_with_feat.sort_values(
                            "offense_rating", ascending=False
                        ).reset_index(drop=True)
                    else:
                        # No proxy data: sort by team_id as stable tiebreak
                        kp_sorted = kp_cands.sort_values(
                            "team_id"
                        ).reset_index(drop=True)

                    for idx in range(n_cbb):
                        tid = int(kp_sorted.iloc[idx]["team_id"])
                        key = (
                            int(season),
                            cbb_sorted.iloc[idx]["team_name"],
                        )
                        estimated[key] = tid

    return confirmed, estimated, unmatched


# ── Merged stats builder ──────────────────────────────────────────────────────

def build_merged_stats(
    cbb_df:       pd.DataFrame,
    confirmed:    dict[tuple, int],
    estimated:    dict[tuple, int],
    ap_df:        pd.DataFrame | None = None,
    kt_df:        pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the merged team stats DataFrame.

    Columns
    -------
    season, team_name_raw, team_name, seed, postseason,
    team_id, match_type,
    offensive_efficiency, defensive_efficiency, efficiency_margin,
    kenpom_torvik_rating,
    ap_rank_week6, ap_top12_flag
    """
    rows = []
    for _, r in cbb_df.iterrows():
        key = (int(r["season"]), r["team_name"])

        if key in confirmed:
            tid        = confirmed[key]
            match_type = "CONFIRMED"
        elif key in estimated:
            tid        = estimated[key]
            match_type = "RANK_MATCH"
        else:
            tid        = None
            match_type = "UNMATCHED"

        row: dict = {
            "season":               int(r["season"]),
            "team_name_raw":        r["team_name_raw"],
            "team_name":            r["team_name"],
            "seed":                 int(r["seed"]),
            "postseason":           r["postseason"],
            "team_id":              tid,
            "match_type":           match_type,
            "offensive_efficiency": r["adjoe"],
            "defensive_efficiency": r["adjde"],
            "efficiency_margin":    r["efficiency_margin"],
            "kenpom_torvik_rating": r.get("barthag"),
            "ap_rank_week6":        None,
            "ap_top12_flag":        0,
        }
        rows.append(row)

    merged = pd.DataFrame(rows)

    # Merge AP week-6 data if available.
    # ap_week6.csv is a sparse top-12 list: presence of a (season, team_name) row
    # is sufficient to set ap_top12_flag = 1.  ap_rank is optional.
    if ap_df is not None and not ap_df.empty:
        ap_top12_set: set[tuple] = set(
            zip(ap_df["season"].astype(int), ap_df["team_name"])
        )
        # Optional rank lookup (may not exist)
        has_rank = "ap_rank" in ap_df.columns
        if has_rank:
            ap_rank_lookup = (
                ap_df.dropna(subset=["ap_rank"])
                     .set_index(["season", "team_name"])["ap_rank"]
                     .to_dict()
            )
        for idx, row in merged.iterrows():
            k = (int(row["season"]), row["team_name"])
            if k in ap_top12_set:
                merged.at[idx, "ap_top12_flag"] = 1
                if has_rank:
                    rank = ap_rank_lookup.get(k)
                    if rank is not None and not pd.isna(rank):
                        merged.at[idx, "ap_rank_week6"] = int(rank)

    # Merge KenPom/Torvik if it adds data not in cbb.csv
    if kt_df is not None and not kt_df.empty:
        if "kt_rating" in kt_df.columns:
            kt_lookup = kt_df.set_index(["season", "team_name"])["kt_rating"].to_dict()
            for idx, row in merged.iterrows():
                k = (row["season"], row["team_name"])
                if pd.isna(row.get("kenpom_torvik_rating")) and k in kt_lookup:
                    merged.at[idx, "kenpom_torvik_rating"] = kt_lookup[k]

    return merged
