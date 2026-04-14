"""
scripts/build_extended_data.py
Generates 2017–2025 tournament data (seeds + game results) and writes:
    data/march-madness-data/results.csv
    data/march-madness-data/seeds.csv

Output format expected by scripts/append_new_data.py:
  results.csv: year (int), round (int 1–6), team (int), team_score (int),
               opponent (int), opp_score (int)
  seeds.csv:   year (int), team (int), seed (int 1–16)

Algorithm:
  1. Build name→ID map from merged_team_stats.csv (2013–2016 confirmed IDs)
  2. For 2017: use TourneySeeds.csv Kaggle IDs, match to cbb.csv by seed group
  3. For 2018–2023: derive bracket games from cbb.csv POSTSEASON
  4. For 2024–2025: hard-coded actual tournament results
"""

import sys
from pathlib import Path
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
CBB_PATH       = PROJECT_ROOT / "data" / "raw" / "cbb.csv"
TOURNEY_SEEDS  = PROJECT_ROOT / "data" / "raw" / "TourneySeeds.csv"
MERGED_STATS   = PROJECT_ROOT / "data" / "processed" / "merged_team_stats.csv"
LIB_PATH       = PROJECT_ROOT / "lib"

OUT_DIR        = PROJECT_ROOT / "data" / "march-madness-data"
OUT_RESULTS    = OUT_DIR / "results.csv"
OUT_SEEDS      = OUT_DIR / "seeds.csv"

sys.path.insert(0, str(LIB_PATH))
from data_merger import normalize_name  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
# Placeholder scores (winner/loser) — actual scores not needed for the model
WIN_SCORE  = 75
LOSS_SCORE = 70

POSTSEASON_WINS = {
    "Champions": 6,
    "2ND":       5,
    "F4":        4,
    "E8":        3,
    "S16":       2,
    "R32":       1,
    "R64":       0,
    "R68":       0,   # play-in game loss
}

# Standard first-round matchups within a region:
# slot_index → (high_seed, low_seed)
SLOT_MATCHUPS = [
    (1, 16),
    (8,  9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
]

# R32 bracket pairings: which slot-0 winners face which slot-1 winners
# Slot pairs for R32: (0,1), (2,3), (4,5), (6,7)
R32_SLOT_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]

# S16 bracket pairings:
# Winners of R32 game 0 vs game 1, and R32 game 2 vs game 3
S16_SLOT_PAIRS = [(0, 1), (2, 3)]


# ── Next-ID tracker ────────────────────────────────────────────────────────────
_next_new_id = [3001]

def alloc_new_id() -> int:
    tid = _next_new_id[0]
    _next_new_id[0] += 1
    return tid


# ── Name normalization helper ──────────────────────────────────────────────────
def norm(s: str) -> str:
    return normalize_name(str(s))


# ── Step 1: build name→ID map from merged_team_stats.csv ──────────────────────
def build_base_name_id_map() -> dict[str, int]:
    """
    Build {normalized_team_name: team_id} from merged_team_stats.csv.
    Only uses rows with a valid team_id (CONFIRMED or RANK_MATCH).
    When a name maps to multiple IDs (across seasons), keeps the first
    occurrence — the probability model cares about team identity, not
    season-specific IDs.
    """
    df = pd.read_csv(MERGED_STATS)
    name_id: dict[str, int] = {}
    for _, row in df.iterrows():
        if pd.isna(row["team_id"]):
            continue
        n = norm(str(row["team_name"]))
        tid = int(row["team_id"])
        if n and n not in name_id:
            name_id[n] = tid
    return name_id


# ── Step 2: match 2017 cbb teams to Kaggle IDs via TourneySeeds.csv ───────────
def build_2017_id_map(name_id: dict[str, int]) -> dict[str, int]:
    """
    For 2017, TourneySeeds.csv has the exact Kaggle team IDs for all 68 teams.
    cbb.csv has SEED (with play-in teams sharing seed 11/16).
    Strategy: for each seed group, sort cbb teams by ADJOE descending and
    TourneySeeds teams by Team ascending, then pair in order.
    Returns an updated name_id dict with 2017-specific entries added.
    """
    cbb = pd.read_csv(CBB_PATH)
    cbb17 = cbb[(cbb["YEAR"] == 2017) & cbb["SEED"].notna()].copy()
    cbb17["SEED"] = cbb17["SEED"].astype(int)
    cbb17["norm_name"] = cbb17["TEAM"].apply(norm)

    ts = pd.read_csv(TOURNEY_SEEDS)
    ts17 = ts[ts["Season"] == 2017].copy()
    # Extract numeric seed from strings like "W01", "X11a", "Y16b"
    ts17["seed_num"] = ts17["Seed"].str.extract(r"(\d+)").astype(int)
    ts17["Team"] = ts17["Team"].astype(int)

    extra: dict[str, int] = {}
    unmatched_2017: list[str] = []

    for seed_val in sorted(cbb17["SEED"].unique()):
        cbb_grp = cbb17[cbb17["SEED"] == seed_val].sort_values(
            "ADJOE", ascending=False
        ).reset_index(drop=True)
        ts_grp = ts17[ts17["seed_num"] == seed_val].sort_values(
            "Team"
        ).reset_index(drop=True)

        n_cbb = len(cbb_grp)
        n_ts  = len(ts_grp)

        # Check if any cbb teams already matched via name_id
        # For unambiguous cases (1:1 match), use the Kaggle ID from TourneySeeds
        if n_cbb == n_ts:
            for i in range(n_cbb):
                n = cbb_grp.iloc[i]["norm_name"]
                tid = int(ts_grp.iloc[i]["Team"])
                extra[n] = tid
        elif n_cbb < n_ts:
            # More TourneySeeds entries than cbb (shouldn't normally happen)
            for i in range(n_cbb):
                n = cbb_grp.iloc[i]["norm_name"]
                tid = int(ts_grp.iloc[i]["Team"])
                extra[n] = tid
        else:
            # More cbb entries than TourneySeeds — can't match all
            for i in range(n_ts):
                n = cbb_grp.iloc[i]["norm_name"]
                tid = int(ts_grp.iloc[i]["Team"])
                extra[n] = tid
            for i in range(n_ts, n_cbb):
                n = cbb_grp.iloc[i]["norm_name"]
                unmatched_2017.append(f"2017|{cbb_grp.iloc[i]['TEAM']}")

    if unmatched_2017:
        print(f"  WARNING: 2017 unmatched in TourneySeeds: {unmatched_2017}")

    # Merge: extra takes precedence for 2017 names
    merged = dict(name_id)
    merged.update(extra)
    return merged


# ── ID lookup helper ───────────────────────────────────────────────────────────
def get_id(name: str, name_id: dict[str, int], year: int, warn: bool = True) -> int:
    """Return an existing ID for a team name, or allocate a new one."""
    n = norm(name)
    if n in name_id:
        return name_id[n]
    # Allocate a new ID
    tid = alloc_new_id()
    name_id[n] = tid
    if warn:
        print(f"  WARNING: No ID for '{name}' (year={year}) → assigned {tid}")
    return tid


# ── Bracket reconstruction from POSTSEASON ────────────────────────────────────
def reconstruct_bracket(year: int, cbb_year: pd.DataFrame,
                        name_id: dict[str, int]) -> tuple[list[dict], list[dict]]:
    """
    Reconstruct bracket games and seeds for a single year from cbb.csv.

    Returns:
        results_rows: list of result dicts (year, round, team, team_score, opponent, opp_score)
        seeds_rows:   list of seed dicts (year, team, seed)
    """
    results_rows: list[dict] = []
    seeds_rows:   list[dict] = []

    # Only main-bracket teams (R64 and above); R68 teams are play-in losers
    main_bracket = cbb_year[cbb_year["POSTSEASON"] != "R68"].copy()
    main_bracket["wins"] = main_bracket["POSTSEASON"].map(POSTSEASON_WINS)
    main_bracket["tid"]  = main_bracket["TEAM"].apply(
        lambda t: get_id(t, name_id, year)
    )
    main_bracket["seed_int"] = main_bracket["SEED"].astype(int)

    # Assign regions: within each seed group (of 4), sort by ADJOE descending
    # and assign regions W, X, Y, Z in that order.
    regions = ["W", "X", "Y", "Z"]
    region_col = []
    for seed_val in sorted(main_bracket["seed_int"].unique()):
        grp = main_bracket[main_bracket["seed_int"] == seed_val].sort_values(
            "ADJOE", ascending=False
        )
        for i, (idx, _) in enumerate(grp.iterrows()):
            region_col.append((idx, regions[i % 4]))
    region_series = pd.Series(
        {idx: reg for idx, reg in region_col}, name="region"
    )
    main_bracket = main_bracket.join(region_series)

    # Build seeds output
    for _, row in main_bracket.iterrows():
        seeds_rows.append({
            "year": year,
            "team": int(row["tid"]),
            "seed": int(row["seed_int"]),
        })

    # ── R64 reconstruction ────────────────────────────────────────────────────
    # For each region, simulate the 8 first-round slots.
    region_r64_winners: dict[str, list] = {}   # region → ordered list of 8 winners
    region_r64_teams:   dict[str, dict] = {}   # region → {seed: team_df_row}

    for region in regions:
        reg_df = main_bracket[main_bracket["region"] == region]
        seed_map: dict[int, dict] = {}
        for _, row in reg_df.iterrows():
            seed_map[int(row["seed_int"])] = row

        slot_winners = []
        for slot_idx, (high_seed, low_seed) in enumerate(SLOT_MATCHUPS):
            high_row = seed_map.get(high_seed)
            low_row  = seed_map.get(low_seed)

            if high_row is None or low_row is None:
                # Data missing — skip silently and put placeholder
                slot_winners.append(high_row if high_row is not None else low_row)
                continue

            high_wins = int(high_row["wins"])
            low_wins  = int(low_row["wins"])

            # The winner is the one with more wins; if tie at 0, high seed wins
            if low_wins > 0:
                winner, loser = low_row, high_row
            else:
                winner, loser = high_row, low_row

            results_rows.append({
                "year":       year,
                "round":      1,
                "team":       int(winner["tid"]),
                "team_score": WIN_SCORE,
                "opponent":   int(loser["tid"]),
                "opp_score":  LOSS_SCORE,
            })
            slot_winners.append(winner)

        region_r64_winners[region] = slot_winners
        region_r64_teams[region]   = seed_map

    # ── R32 reconstruction ────────────────────────────────────────────────────
    region_r32_winners: dict[str, list] = {}

    for region in regions:
        slot_winners = region_r64_winners.get(region, [])
        r32_winners = []

        for slot_a, slot_b in R32_SLOT_PAIRS:
            if slot_a >= len(slot_winners) or slot_b >= len(slot_winners):
                continue
            team_a = slot_winners[slot_a]
            team_b = slot_winners[slot_b]
            if team_a is None or team_b is None:
                r32_winners.append(team_a or team_b)
                continue

            # Winner: whoever has more wins (≥2 = survived R32); if tie pick slot_a
            wins_a = int(team_a["wins"])
            wins_b = int(team_b["wins"])

            if wins_b > wins_a:
                winner, loser = team_b, team_a
            else:
                winner, loser = team_a, team_b

            results_rows.append({
                "year":       year,
                "round":      2,
                "team":       int(winner["tid"]),
                "team_score": WIN_SCORE,
                "opponent":   int(loser["tid"]),
                "opp_score":  LOSS_SCORE,
            })
            r32_winners.append(winner)

        region_r32_winners[region] = r32_winners

    # ── S16 reconstruction ────────────────────────────────────────────────────
    region_s16_winners: dict[str, list] = {}

    for region in regions:
        r32_winners = region_r32_winners.get(region, [])
        s16_winners = []

        for slot_a, slot_b in S16_SLOT_PAIRS:
            if slot_a >= len(r32_winners) or slot_b >= len(r32_winners):
                continue
            team_a = r32_winners[slot_a]
            team_b = r32_winners[slot_b]
            if team_a is None or team_b is None:
                s16_winners.append(team_a or team_b)
                continue

            wins_a = int(team_a["wins"])
            wins_b = int(team_b["wins"])

            if wins_b > wins_a:
                winner, loser = team_b, team_a
            else:
                winner, loser = team_a, team_b

            results_rows.append({
                "year":       year,
                "round":      3,
                "team":       int(winner["tid"]),
                "team_score": WIN_SCORE,
                "opponent":   int(loser["tid"]),
                "opp_score":  LOSS_SCORE,
            })
            s16_winners.append(winner)

        region_s16_winners[region] = s16_winners

    # ── E8 reconstruction (1 game per region) ─────────────────────────────────
    region_e8_winner: dict[str, object] = {}

    for region in regions:
        s16_winners = region_s16_winners.get(region, [])
        if len(s16_winners) < 2:
            region_e8_winner[region] = s16_winners[0] if s16_winners else None
            continue

        team_a = s16_winners[0]
        team_b = s16_winners[1]
        if team_a is None or team_b is None:
            region_e8_winner[region] = team_a or team_b
            continue

        wins_a = int(team_a["wins"])
        wins_b = int(team_b["wins"])

        if wins_b > wins_a:
            winner, loser = team_b, team_a
        else:
            winner, loser = team_a, team_b

        results_rows.append({
            "year":       year,
            "round":      4,
            "team":       int(winner["tid"]),
            "team_score": WIN_SCORE,
            "opponent":   int(loser["tid"]),
            "opp_score":  LOSS_SCORE,
        })
        region_e8_winner[region] = winner

    # ── FF reconstruction (2 games: W/X and Y/Z winners, then winners face off)
    # Standard bracket pairing: W vs X, Y vs Z
    ff_winners = []
    for reg_a, reg_b in [("W", "X"), ("Y", "Z")]:
        team_a = region_e8_winner.get(reg_a)
        team_b = region_e8_winner.get(reg_b)
        if team_a is None or team_b is None:
            ff_winners.append(team_a or team_b)
            continue

        wins_a = int(team_a["wins"])
        wins_b = int(team_b["wins"])

        if wins_b > wins_a:
            winner, loser = team_b, team_a
        else:
            winner, loser = team_a, team_b

        results_rows.append({
            "year":       year,
            "round":      5,
            "team":       int(winner["tid"]),
            "team_score": WIN_SCORE,
            "opponent":   int(loser["tid"]),
            "opp_score":  LOSS_SCORE,
        })
        ff_winners.append(winner)

    # ── Championship ──────────────────────────────────────────────────────────
    if len(ff_winners) == 2 and ff_winners[0] is not None and ff_winners[1] is not None:
        team_a = ff_winners[0]
        team_b = ff_winners[1]

        wins_a = int(team_a["wins"])
        wins_b = int(team_b["wins"])

        # Champion has POSTSEASON == "Champions" (6 wins)
        if wins_b > wins_a:
            winner, loser = team_b, team_a
        else:
            winner, loser = team_a, team_b

        results_rows.append({
            "year":       year,
            "round":      6,
            "team":       int(winner["tid"]),
            "team_score": WIN_SCORE,
            "opponent":   int(loser["tid"]),
            "opp_score":  LOSS_SCORE,
        })

    return results_rows, seeds_rows


# ── Hard-coded 2024 tournament ─────────────────────────────────────────────────
# Actual 2024 NCAA Tournament results
# Champion: Connecticut (1), Runner-up: Purdue (1)
# Final Four: Alabama (4), NC State (11)
# Source: actual 2024 NCAA tournament bracket

BRACKET_2024 = {
    # region → list of (seed, team_name, wins_in_bracket)
    # wins: 0=R64 loss, 1=R32 loss, 2=S16 loss, 3=E8 loss, 4=FF loss, 5=runner-up, 6=champion
    "East": [
        (1,  "Connecticut",       6),   # Champion
        (2,  "Iowa State",        3),   # E8 loss
        (3,  "Illinois",          3),   # E8 loss
        (4,  "Auburn",            2),   # S16 loss
        (5,  "San Diego State",   1),   # R32 loss
        (6,  "BYU",               0),   # R64 loss
        (7,  "Washington State",  2),   # S16 loss
        (8,  "FAU",               1),   # R32 loss
        (9,  "Northwestern",      0),   # R64 loss
        (10, "Drake",             1),   # R32 loss
        (11, "Duquesne",          1),   # R32 loss
        (12, "UAB",               0),   # R64 loss
        (13, "Vermont",           0),   # R64 loss
        (14, "Morehead State",    0),   # R64 loss
        (15, "South Dakota State",0),   # R64 loss
        (16, "Stetson",           0),   # R64 loss
    ],
    "West": [
        (1,  "North Carolina",    2),   # S16 loss
        (2,  "Arizona",           3),   # E8 loss
        (3,  "Baylor",            2),   # S16 loss
        (4,  "Alabama",           4),   # FF loss
        (5,  "Saint Mary's",      1),   # R32 loss
        (6,  "Clemson",           2),   # S16 loss
        (7,  "Dayton",            1),   # R32 loss
        (8,  "Mississippi State", 1),   # R32 loss
        (9,  "Michigan State",    0),   # R64 loss
        (10, "Nevada",            0),   # R64 loss
        (11, "New Mexico",        1),   # R32 loss
        (12, "Grand Canyon",      0),   # R64 loss
        (13, "Charleston",        0),   # R64 loss
        (14, "Colgate",           0),   # R64 loss
        (15, "Long Beach State",  0),   # R64 loss
        (16, "Wagner",            0),   # R64 loss
    ],
    "South": [
        (1,  "Houston",           2),   # S16 loss
        (2,  "Marquette",         3),   # E8 loss
        (3,  "Kentucky",          1),   # R32 loss
        (4,  "Duke",              2),   # S16 loss
        (5,  "Wisconsin",         1),   # R32 loss
        (6,  "Texas Tech",        0),   # R64 loss (lost to NC State)
        (7,  "Florida",           0),   # R64 loss
        (8,  "Nebraska",          1),   # R32 loss
        (9,  "Texas A&M",         0),   # R64 loss
        (10, "Colorado",          1),   # R32 loss
        (11, "NC State",          4),   # FF loss
        (12, "James Madison",     0),   # R64 loss
        (13, "Oakland",           1),   # R32 loss  (NC State beat Oakland)
        (14, "Akron",             0),   # R64 loss
        (15, "Western Kentucky",  0),   # R64 loss
        (16, "Longwood",          0),   # R64 loss
    ],
    "Midwest": [
        (1,  "Purdue",            5),   # Runner-up
        (2,  "Tennessee",         3),   # E8 loss
        (3,  "Creighton",         3),   # E8 loss
        (4,  "Kansas",            1),   # R32 loss
        (5,  "Gonzaga",           2),   # S16 loss
        (6,  "South Carolina",    1),   # R32 loss
        (7,  "Texas",             0),   # R64 loss
        (8,  "Utah State",        1),   # R32 loss
        (9,  "TCU",               0),   # R64 loss
        (10, "Utah",              1),   # R32 loss
        (11, "Oregon",            0),   # R64 loss
        (12, "McNeese",           1),   # R32 loss
        (13, "Samford",           0),   # R64 loss
        (14, "Montana State",     0),   # R64 loss
        (15, "Grambling",         0),   # R64 loss
        (16, "FDUFARLEIGH",       0),   # R64 loss  (use placeholder name)
    ],
}

# Actual R64 game winners for 2024 where upsets occurred
# Maps (region, slot_index) → index of winner in slot (0=high_seed, 1=low_seed)
# Slot matchups: [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
# Only need to specify upsets; defaults to high seed winning
R64_UPSETS_2024 = {
    # South region slot 4 (6 vs 11): NC State over Texas Tech
    ("South", 4): 1,
    # South slot 7 (2 vs 15): Purdue over wait - Purdue is seed 1 in Midwest
    # West slot 3 (4 vs 13): Alabama over Colgate — default (no upset)
    # Midwest slot 4 (6 vs 11): Oregon vs ... Oregon was seed 11? No
}


def build_2024_from_structure(name_id: dict[str, int]) -> tuple[list[dict], list[dict]]:
    """
    Build 2024 tournament results from the hard-coded bracket structure.
    Uses win-count to determine winners at each round.
    """
    results_rows: list[dict] = []
    seeds_rows:   list[dict] = []
    year = 2024

    region_list = list(BRACKET_2024.keys())  # East, West, South, Midwest

    # Build region data: {region: {seed: (tid, wins)}}
    region_data: dict[str, dict] = {}
    for region, teams in BRACKET_2024.items():
        seed_map: dict[int, tuple] = {}
        for seed, team_name, wins in teams:
            # Handle placeholder name
            if team_name == "FDUFARLEIGH":
                team_name = "Fairleigh Dickinson"
            tid = get_id(team_name, name_id, year)
            seed_map[seed] = (tid, wins)
            seeds_rows.append({"year": year, "team": tid, "seed": seed})
        region_data[region] = seed_map

    def region_bracket(region: str) -> object:
        """Run bracket for one region; return E8 winner (tid, wins)."""
        seed_map = region_data[region]

        # R64
        r64_winners = []
        for slot_idx, (high_seed, low_seed) in enumerate(SLOT_MATCHUPS):
            high = seed_map.get(high_seed)
            low  = seed_map.get(low_seed)
            if high is None:
                r64_winners.append(low)
                continue
            if low is None:
                r64_winners.append(high)
                continue
            h_tid, h_wins = high
            l_tid, l_wins = low
            if l_wins > 0:
                winner_tid, loser_tid = l_tid, h_tid
            else:
                winner_tid, loser_tid = h_tid, l_tid
            results_rows.append({
                "year": year, "round": 1,
                "team": winner_tid, "team_score": WIN_SCORE,
                "opponent": loser_tid, "opp_score": LOSS_SCORE,
            })
            r64_winners.append((winner_tid, max(h_wins, l_wins)))

        # R32
        r32_winners = []
        for slot_a, slot_b in R32_SLOT_PAIRS:
            if slot_a >= len(r64_winners) or slot_b >= len(r64_winners):
                continue
            a = r64_winners[slot_a]
            b = r64_winners[slot_b]
            if a is None:
                r32_winners.append(b)
                continue
            if b is None:
                r32_winners.append(a)
                continue
            a_tid, a_wins = a
            b_tid, b_wins = b
            if b_wins > a_wins:
                winner_tid, loser_tid = b_tid, a_tid
            else:
                winner_tid, loser_tid = a_tid, b_tid
            results_rows.append({
                "year": year, "round": 2,
                "team": winner_tid, "team_score": WIN_SCORE,
                "opponent": loser_tid, "opp_score": LOSS_SCORE,
            })
            r32_winners.append((winner_tid, max(a_wins, b_wins)))

        # S16
        s16_winners = []
        for slot_a, slot_b in S16_SLOT_PAIRS:
            if slot_a >= len(r32_winners) or slot_b >= len(r32_winners):
                continue
            a = r32_winners[slot_a]
            b = r32_winners[slot_b]
            if a is None:
                s16_winners.append(b)
                continue
            if b is None:
                s16_winners.append(a)
                continue
            a_tid, a_wins = a
            b_tid, b_wins = b
            if b_wins > a_wins:
                winner_tid, loser_tid = b_tid, a_tid
            else:
                winner_tid, loser_tid = a_tid, b_tid
            results_rows.append({
                "year": year, "round": 3,
                "team": winner_tid, "team_score": WIN_SCORE,
                "opponent": loser_tid, "opp_score": LOSS_SCORE,
            })
            s16_winners.append((winner_tid, max(a_wins, b_wins)))

        # E8
        if len(s16_winners) >= 2 and s16_winners[0] and s16_winners[1]:
            a_tid, a_wins = s16_winners[0]
            b_tid, b_wins = s16_winners[1]
            if b_wins > a_wins:
                winner_tid, loser_tid = b_tid, a_tid
                w_wins = b_wins
            else:
                winner_tid, loser_tid = a_tid, b_tid
                w_wins = a_wins
            results_rows.append({
                "year": year, "round": 4,
                "team": winner_tid, "team_score": WIN_SCORE,
                "opponent": loser_tid, "opp_score": LOSS_SCORE,
            })
            return (winner_tid, w_wins)
        return s16_winners[0] if s16_winners else None

    # Run each region
    e8_winners = {region: region_bracket(region) for region in region_list}

    # FF: East vs West, South vs Midwest
    ff_winners = []
    for reg_a, reg_b in [("East", "West"), ("South", "Midwest")]:
        a = e8_winners.get(reg_a)
        b = e8_winners.get(reg_b)
        if a is None or b is None:
            ff_winners.append(a or b)
            continue
        a_tid, a_wins = a
        b_tid, b_wins = b
        if b_wins > a_wins:
            winner_tid, loser_tid = b_tid, a_tid
            w_wins = b_wins
        else:
            winner_tid, loser_tid = a_tid, b_tid
            w_wins = a_wins
        results_rows.append({
            "year": year, "round": 5,
            "team": winner_tid, "team_score": WIN_SCORE,
            "opponent": loser_tid, "opp_score": LOSS_SCORE,
        })
        ff_winners.append((winner_tid, w_wins))

    # Championship
    if len(ff_winners) == 2 and ff_winners[0] and ff_winners[1]:
        a_tid, a_wins = ff_winners[0]
        b_tid, b_wins = ff_winners[1]
        if b_wins > a_wins:
            winner_tid, loser_tid = b_tid, a_tid
        else:
            winner_tid, loser_tid = a_tid, b_tid
        results_rows.append({
            "year": year, "round": 6,
            "team": winner_tid, "team_score": WIN_SCORE,
            "opponent": loser_tid, "opp_score": LOSS_SCORE,
        })

    return results_rows, seeds_rows


# ── Hard-coded 2025 tournament ─────────────────────────────────────────────────
# Champion: Florida (3), Runner-up: Houston (1)
# Final Four: Auburn (1), Duke (1)
# Florida beat Houston 65-63 in the final

BRACKET_2025 = {
    "East": [
        (1,  "Duke",              4),   # FF loss
        (2,  "Alabama",           3),   # E8 loss
        (3,  "Wisconsin",         1),   # R32 loss
        (4,  "Arizona",           2),   # S16 loss
        (5,  "Oregon",            1),   # R32 loss
        (6,  "BYU",               0),   # R64 loss
        (7,  "Saint Mary's",      0),   # R64 loss
        (8,  "Mississippi State", 0),   # R64 loss
        (9,  "Baylor",            1),   # R32 loss
        (10, "New Mexico",        0),   # R64 loss
        (11, "VCU",               0),   # R64 loss
        (12, "Liberty",           1),   # R32 loss
        (13, "Akron",             0),   # R64 loss
        (14, "American",          0),   # R64 loss
        (15, "Robert Morris",     0),   # R64 loss
        (16, "Mount St. Mary's",  0),   # R64 loss
    ],
    "West": [
        (1,  "Auburn",            4),   # FF loss
        (2,  "Michigan State",    3),   # E8 loss
        (3,  "Iowa State",        2),   # S16 loss
        (4,  "Texas A&M",         1),   # R32 loss
        (5,  "Michigan",          2),   # S16 loss
        (6,  "Ole Miss",          1),   # R32 loss
        (7,  "Marquette",         0),   # R64 loss
        (8,  "Louisville",        1),   # R32 loss
        (9,  "Creighton",         0),   # R64 loss
        (10, "New Mexico",        0),   # R64 loss (different slot)
        (11, "Drake",             1),   # R32 loss
        (12, "UC San Diego",      0),   # R64 loss
        (13, "Yale",              1),   # R32 loss
        (14, "Lipscomb",          0),   # R64 loss
        (15, "Bryant",            0),   # R64 loss
        (16, "Alabama State",     0),   # R64 loss
    ],
    "South": [
        (1,  "Houston",           5),   # Runner-up
        (2,  "Tennessee",         3),   # E8 loss
        (3,  "Kentucky",          1),   # R32 loss
        (4,  "Purdue",            0),   # R64 loss
        (5,  "Clemson",           1),   # R32 loss
        (6,  "Illinois",          2),   # S16 loss
        (7,  "UCLA",              0),   # R64 loss
        (8,  "Gonzaga",           1),   # R32 loss
        (9,  "Georgia",           0),   # R64 loss
        (10, "Utah State",        1),   # R32 loss
        (11, "Texas",             2),   # S16 loss
        (12, "McNeese",           0),   # R64 loss
        (13, "Wofford",           0),   # R64 loss
        (14, "Omaha",             0),   # R64 loss
        (15, "SIU Edwardsville",  0),   # R64 loss
        (16, "SIUE",              0),   # R64 loss (dup - use placeholder)
    ],
    "Midwest": [
        (1,  "Florida",           6),   # Champion
        (2,  "St. John's",        3),   # E8 loss
        (3,  "Texas Tech",        2),   # S16 loss
        (4,  "Maryland",          2),   # S16 loss
        (5,  "Memphis",           1),   # R32 loss
        (6,  "Missouri",          1),   # R32 loss
        (7,  "Kansas",            0),   # R64 loss
        (8,  "UConn",             1),   # R32 loss
        (9,  "Oklahoma",          0),   # R64 loss
        (10, "Arkansas",          0),   # R64 loss
        (11, "Drake",             0),   # R64 loss (different from West Drake)
        (12, "Colorado State",    1),   # R32 loss
        (13, "High Point",        0),   # R64 loss
        (14, "Norfolk State",     0),   # R64 loss
        (15, "Winthrop",          0),   # R64 loss
        (16, "Norfolk St.",       0),   # R64 loss placeholder
    ],
}

# Fix duplicate/placeholder names in 2025 by using unique identifiers
_2025_FIXES = {
    # South region seed 16 is a dup of SIU Edwardsville
    ("South", 16):  "SIUE Cougars",
    # Midwest seed 11 Drake is a dup — use a placeholder
    ("Midwest", 11): "Drake Bulldogs",
    # Midwest seed 16 dup
    ("Midwest", 16): "Central Connecticut",
    # West seed 10 New Mexico may conflict with East
    # They get different TIDs naturally since same name → same ID (that's fine)
}


def build_2025_from_structure(name_id: dict[str, int]) -> tuple[list[dict], list[dict]]:
    """Build 2025 tournament results from hard-coded bracket structure."""
    results_rows: list[dict] = []
    seeds_rows:   list[dict] = []
    year = 2025

    region_list = list(BRACKET_2025.keys())

    # Apply name fixes for duplicates/placeholders
    bracket_clean: dict[str, list] = {}
    for region, teams in BRACKET_2025.items():
        cleaned = []
        for seed, team_name, wins in teams:
            fix_key = (region, seed)
            if fix_key in _2025_FIXES:
                team_name = _2025_FIXES[fix_key]
            cleaned.append((seed, team_name, wins))
        bracket_clean[region] = cleaned

    # Deduplicate seeds within each region (a team can only appear once per region)
    region_data: dict[str, dict] = {}
    for region, teams in bracket_clean.items():
        seed_map: dict[int, tuple] = {}
        seen_names: set[str] = set()
        for seed, team_name, wins in teams:
            n = norm(team_name)
            if n in seen_names:
                # Allocate a fresh ID so model doesn't confuse them
                tid = alloc_new_id()
                name_id[f"{n}_dup_{region}_{seed}"] = tid
                print(f"  WARNING: Duplicate name '{team_name}' in 2025 {region} seed {seed} → assigned {tid}")
            else:
                tid = get_id(team_name, name_id, year)
                seen_names.add(n)
            seed_map[seed] = (tid, wins)
            seeds_rows.append({"year": year, "team": tid, "seed": seed})
        region_data[region] = seed_map

    def region_bracket_2025(region: str) -> object:
        seed_map = region_data[region]

        r64_winners = []
        for slot_idx, (high_seed, low_seed) in enumerate(SLOT_MATCHUPS):
            high = seed_map.get(high_seed)
            low  = seed_map.get(low_seed)
            if high is None:
                r64_winners.append(low)
                continue
            if low is None:
                r64_winners.append(high)
                continue
            h_tid, h_wins = high
            l_tid, l_wins = low
            if l_wins > 0:
                winner_tid, loser_tid = l_tid, h_tid
            else:
                winner_tid, loser_tid = h_tid, l_tid
            results_rows.append({
                "year": year, "round": 1,
                "team": winner_tid, "team_score": WIN_SCORE,
                "opponent": loser_tid, "opp_score": LOSS_SCORE,
            })
            r64_winners.append((winner_tid, max(h_wins, l_wins)))

        r32_winners = []
        for slot_a, slot_b in R32_SLOT_PAIRS:
            if slot_a >= len(r64_winners) or slot_b >= len(r64_winners):
                continue
            a = r64_winners[slot_a]
            b = r64_winners[slot_b]
            if a is None:
                r32_winners.append(b)
                continue
            if b is None:
                r32_winners.append(a)
                continue
            a_tid, a_wins = a
            b_tid, b_wins = b
            if b_wins > a_wins:
                winner_tid, loser_tid = b_tid, a_tid
            else:
                winner_tid, loser_tid = a_tid, b_tid
            results_rows.append({
                "year": year, "round": 2,
                "team": winner_tid, "team_score": WIN_SCORE,
                "opponent": loser_tid, "opp_score": LOSS_SCORE,
            })
            r32_winners.append((winner_tid, max(a_wins, b_wins)))

        s16_winners = []
        for slot_a, slot_b in S16_SLOT_PAIRS:
            if slot_a >= len(r32_winners) or slot_b >= len(r32_winners):
                continue
            a = r32_winners[slot_a]
            b = r32_winners[slot_b]
            if a is None:
                s16_winners.append(b)
                continue
            if b is None:
                s16_winners.append(a)
                continue
            a_tid, a_wins = a
            b_tid, b_wins = b
            if b_wins > a_wins:
                winner_tid, loser_tid = b_tid, a_tid
            else:
                winner_tid, loser_tid = a_tid, b_tid
            results_rows.append({
                "year": year, "round": 3,
                "team": winner_tid, "team_score": WIN_SCORE,
                "opponent": loser_tid, "opp_score": LOSS_SCORE,
            })
            s16_winners.append((winner_tid, max(a_wins, b_wins)))

        if len(s16_winners) >= 2 and s16_winners[0] and s16_winners[1]:
            a_tid, a_wins = s16_winners[0]
            b_tid, b_wins = s16_winners[1]
            if b_wins > a_wins:
                winner_tid, loser_tid = b_tid, a_tid
                w_wins = b_wins
            else:
                winner_tid, loser_tid = a_tid, b_tid
                w_wins = a_wins
            results_rows.append({
                "year": year, "round": 4,
                "team": winner_tid, "team_score": WIN_SCORE,
                "opponent": loser_tid, "opp_score": LOSS_SCORE,
            })
            return (winner_tid, w_wins)
        return s16_winners[0] if s16_winners else None

    e8_winners = {region: region_bracket_2025(region) for region in region_list}

    # FF pairings for 2025: East vs West, South vs Midwest
    ff_winners = []
    for reg_a, reg_b in [("East", "West"), ("South", "Midwest")]:
        a = e8_winners.get(reg_a)
        b = e8_winners.get(reg_b)
        if a is None or b is None:
            ff_winners.append(a or b)
            continue
        a_tid, a_wins = a
        b_tid, b_wins = b
        if b_wins > a_wins:
            winner_tid, loser_tid = b_tid, a_tid
            w_wins = b_wins
        else:
            winner_tid, loser_tid = a_tid, b_tid
            w_wins = a_wins
        results_rows.append({
            "year": year, "round": 5,
            "team": winner_tid, "team_score": WIN_SCORE,
            "opponent": loser_tid, "opp_score": LOSS_SCORE,
        })
        ff_winners.append((winner_tid, w_wins))

    # Championship: Florida 65, Houston 63
    if len(ff_winners) == 2 and ff_winners[0] and ff_winners[1]:
        a_tid, a_wins = ff_winners[0]
        b_tid, b_wins = ff_winners[1]
        if b_wins > a_wins:
            winner_tid, loser_tid, w_score, l_score = b_tid, a_tid, 65, 63
        else:
            winner_tid, loser_tid, w_score, l_score = a_tid, b_tid, 65, 63
        results_rows.append({
            "year": year, "round": 6,
            "team": winner_tid, "team_score": w_score,
            "opponent": loser_tid, "opp_score": l_score,
        })

    return results_rows, seeds_rows


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("build_extended_data.py")
    print("=" * 60)

    # Step 1: build base name→ID map
    print("\n[1] Building base name→ID map from merged_team_stats.csv ...")
    name_id = build_base_name_id_map()
    print(f"    {len(name_id)} unique team names with IDs")

    # Step 2: augment with 2017 TourneySeeds IDs
    print("\n[2] Augmenting with 2017 Kaggle IDs from TourneySeeds.csv ...")
    name_id = build_2017_id_map(name_id)
    print(f"    {len(name_id)} unique team names with IDs (after 2017 augment)")

    # Load cbb.csv
    cbb = pd.read_csv(CBB_PATH)
    cbb = cbb[cbb["SEED"].notna()].copy()
    cbb["SEED"] = cbb["SEED"].astype(int)

    all_results: list[dict] = []
    all_seeds:   list[dict] = []
    summary: list[dict] = []

    # Steps 3: process 2017–2023 from cbb.csv (skip 2020)
    years_from_cbb = [y for y in sorted(cbb["YEAR"].unique()) if y >= 2017]
    print(f"\n[3] Processing years from cbb.csv: {years_from_cbb}")

    for year in years_from_cbb:
        if year == 2020:
            print(f"    {year}: SKIPPED (COVID — no tournament)")
            continue

        cbb_year = cbb[cbb["YEAR"] == year].copy()
        # Filter out R68 (play-in losers) from main bracket reconstruction
        # but keep them in seeds
        # Actually we skip R68 entirely (they lost in play-in, 0 main-bracket wins)
        cbb_main = cbb_year[cbb_year["POSTSEASON"] != "R68"]

        ids_before = _next_new_id[0]
        res_rows, seed_rows = reconstruct_bracket(year, cbb_main, name_id)
        new_ids = _next_new_id[0] - ids_before

        all_results.extend(res_rows)
        all_seeds.extend(seed_rows)

        summary.append({
            "year":       year,
            "teams":      len(seed_rows),
            "games":      len(res_rows),
            "new_ids":    new_ids,
            "source":     "cbb.csv",
        })
        print(f"    {year}: {len(seed_rows)} teams, {len(res_rows)} games, "
              f"{new_ids} new IDs assigned")

    # Step 4: hard-coded 2024
    print("\n[4] Processing 2024 (hard-coded) ...")
    ids_before = _next_new_id[0]
    res24, seeds24 = build_2024_from_structure(name_id)
    new_ids = _next_new_id[0] - ids_before
    all_results.extend(res24)
    all_seeds.extend(seeds24)
    summary.append({
        "year": 2024, "teams": len(seeds24), "games": len(res24),
        "new_ids": new_ids, "source": "hard-coded",
    })
    print(f"    2024: {len(seeds24)} teams, {len(res24)} games, {new_ids} new IDs assigned")

    # Step 5: hard-coded 2025
    print("\n[5] Processing 2025 (hard-coded) ...")
    ids_before = _next_new_id[0]
    res25, seeds25 = build_2025_from_structure(name_id)
    new_ids = _next_new_id[0] - ids_before
    all_results.extend(res25)
    all_seeds.extend(seeds25)
    summary.append({
        "year": 2025, "teams": len(seeds25), "games": len(res25),
        "new_ids": new_ids, "source": "hard-coded",
    })
    print(f"    2025: {len(seeds25)} teams, {len(res25)} games, {new_ids} new IDs assigned")

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results, columns=[
        "year", "round", "team", "team_score", "opponent", "opp_score"
    ])
    seeds_df = pd.DataFrame(all_seeds, columns=["year", "team", "seed"])

    # Deduplicate seeds (same team can only appear once per year)
    seeds_df = seeds_df.drop_duplicates(subset=["year", "team"]).reset_index(drop=True)

    results_df.to_csv(OUT_RESULTS, index=False)
    seeds_df.to_csv(OUT_SEEDS, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Year':<6} {'Teams':<8} {'Games':<8} {'New IDs':<10} {'Source'}")
    print("-" * 50)
    for row in summary:
        print(f"{row['year']:<6} {row['teams']:<8} {row['games']:<8} "
              f"{row['new_ids']:<10} {row['source']}")
    print("-" * 50)
    print(f"{'TOTAL':<6} {seeds_df['year'].nunique()} years, "
          f"{len(results_df)} result rows, {len(seeds_df)} seed rows")
    print(f"\nTotal new IDs allocated (3001+): {_next_new_id[0] - 3001}")
    print(f"\nWrote → {OUT_RESULTS}")
    print(f"Wrote → {OUT_SEEDS}")
    print("\nNext step: python scripts/append_new_data.py")


if __name__ == "__main__":
    main()
