"""
integrate_real_features.py
Enrich historical_team_ratings.csv with real features from merged_team_stats.csv
for matched teams in 2013–2016, then re-run backtests to compare.

Feature mapping (matched teams only)
-------------------------------------
  offense_rating        ← ADJOE  (real adjusted offensive efficiency)
  defense_rating        ← ADJDE  (real adjusted defensive efficiency)
  efficiency_margin     ← real efficiency margin
  strength_of_schedule  ← kenpom_torvik_rating / BARTHAG  (0–1 power rating)
  ap_rank_week6         ← real ap_rank_week6  (if ap_week6.csv was loaded; else null)
  recent_form           ← keep existing proxy  (no real data source yet)

Unmatched teams in 2013–2016 (9 teams across 2014/2016)
---------------------------------------------------------
  Impute offense/defense/efficiency from seed-tier median of matched season peers
  so all teams in a season share the same feature scale for within-season z-scoring.

Outputs
-------
  data/processed/historical_team_ratings.csv  (updated in-place)
  data/processed/historical_team_ratings_pre_enrichment.csv  (backup of original)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json

import numpy as np
import pandas as pd

from lib.team_ratings import rate_teams_by_season
from lib.champion_profile import compute_profile_scores_df

HIST_CSV    = PROJECT_ROOT / "data" / "processed" / "historical_team_ratings.csv"
MERGED_CSV  = PROJECT_ROOT / "data" / "processed" / "merged_team_stats.csv"
BACKUP_CSV  = PROJECT_ROOT / "data" / "processed" / "historical_team_ratings_pre_enrichment.csv"
BACKTEST_DIR = PROJECT_ROOT / "data" / "processed" / "backtests"

ENRICH_SEASONS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]

W   = 72
SEP = "=" * W
THN = "─" * W


# ════════════════════════════════════════════════════════════════════════════
# Backtest helpers
# ════════════════════════════════════════════════════════════════════════════

def _run_backtests(label: str) -> dict[int, dict]:
    """Run balanced backtests for 2013–2016 and return results keyed by year."""
    from lib.backtest import run_backtest
    results: dict[int, dict] = {}
    for year in ENRICH_SEASONS:
        try:
            r = run_backtest(year, mode="balanced")
            results[year] = r
        except Exception as e:
            print(f"  [ERROR] {year} backtest failed: {e}")
    return results


def _extract_metrics(bt: dict) -> dict:
    """Pull the key numbers we want to compare."""
    pc   = bt["predicted_champion"]
    ac   = bt["actual_champion"]
    pff  = bt["predicted_final_four"]
    aff  = bt["actual_final_four"]
    aff_names = {t["name"] for t in aff}
    ff_correct = sum(1 for t in pff if t["name"] in aff_names)
    tc   = bt["total_correct"]
    tp   = bt["total_possible"]
    by_r = bt.get("by_round_detail", {})
    return {
        "total_correct":    tc,
        "total_possible":   tp,
        "accuracy":         tc / tp if tp else 0,
        "champ_correct":    pc["name"] == ac["name"],
        "pred_champ":       f"{pc['name']} (seed {pc['seed']})",
        "actual_champ":     f"{ac['name']} (seed {ac['seed']})",
        "ff_correct":       ff_correct,
        "pred_ff_seeds":    sorted([t["seed"] for t in pff]),
        "actual_ff_seeds":  sorted([t["seed"] for t in aff]),
        "by_round":         {r: v["correct"] for r, v in by_r.items()},
    }


def _print_comparison(before: dict[int, dict], after: dict[int, dict]) -> None:
    print(f"\n{SEP}")
    print(f"{'BEFORE vs AFTER  —  2013–2016 balanced':^{W}}")
    print(SEP)

    rounds = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    round_max = {"Round of 64": 32, "Round of 32": 16, "Sweet 16": 8,
                 "Elite 8": 4, "Final Four": 2, "Championship": 1}

    for year in ENRICH_SEASONS:
        b = _extract_metrics(before[year])
        a = _extract_metrics(after[year])
        delta = a["total_correct"] - b["total_correct"]
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        champ_b = "✓" if b["champ_correct"] else "✗"
        champ_a = "✓" if a["champ_correct"] else "✗"

        print(f"\n  {year}  ({'BEFORE':^6})  →  ({'AFTER':^6})")
        print(f"  {'─'*60}")
        print(f"  {'Total:':<22} {b['total_correct']:>2}/{b['total_possible']}  ({b['accuracy']:.1%})"
              f"  →  {a['total_correct']:>2}/{a['total_possible']}  ({a['accuracy']:.1%})"
              f"  Δ={delta_str}")
        print(f"  {'Champion:':<22} {champ_b} {b['pred_champ']:<28}"
              f"  →  {champ_a} {a['pred_champ']}")
        print(f"  {'Actual champion:':<22}   {b['actual_champ']}")
        print(f"  {'FF seeds (pred):':<22} {b['pred_ff_seeds']}"
              f"  →  {a['pred_ff_seeds']}")
        print(f"  {'FF seeds (actual):':<22} {a['actual_ff_seeds']}")
        print(f"  {'FF correct:':<22} {b['ff_correct']}/4  →  {a['ff_correct']}/4")
        print(f"  {'By round (correct picks):':}")
        for rnd in rounds:
            max_p = round_max[rnd]
            bv    = b["by_round"].get(rnd, 0)
            av    = a["by_round"].get(rnd, 0)
            d     = av - bv
            d_str = f"+{d}" if d > 0 else str(d) if d != 0 else " 0"
            print(f"    {rnd:<18} {bv:>2}/{max_p}  →  {av:>2}/{max_p}  Δ={d_str}")

    # Aggregate
    print(f"\n{THN}")
    print(f"  AGGREGATE  2013–2016")
    print(THN)

    def _agg(results):
        tc = sum(_extract_metrics(v)["total_correct"] for v in results.values())
        tp = sum(_extract_metrics(v)["total_possible"] for v in results.values())
        nc = sum(1 for v in results.values() if _extract_metrics(v)["champ_correct"])
        ff = sum(_extract_metrics(v)["ff_correct"] for v in results.values())
        return tc, tp, nc, ff

    btc, btp, bnc, bff = _agg(before)
    atc, atp, anc, aff = _agg(after)
    print(f"  {'Metric':<28}  {'Before':>8}  {'After':>8}  {'Delta':>8}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'Total correct:':<28}  {btc:>5}/{btp}  {atc:>5}/{atp}  {atc-btc:>+8}")
    print(f"  {'Overall accuracy:':<28}  {btc/btp:>8.1%}  {atc/atp:>8.1%}  {(atc-btc)/btp:>+8.1%}")
    print(f"  {'Champion correct (of 4):':<28}  {bnc:>8}  {anc:>8}  {anc-bnc:>+8}")
    print(f"  {'FF correct (of 16):':<28}  {bff:>8}  {aff:>8}  {aff-bff:>+8}")


# ════════════════════════════════════════════════════════════════════════════
# Enrichment logic
# ════════════════════════════════════════════════════════════════════════════

def _enrich_historical(hist_df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace proxy features with real features for matched teams (2013–2016).

    For the handful of unmatched teams within 2013–2016 (where all other teams
    have real ADJOE-scale values), impute from the seed-tier median of matched
    peers in the same season to maintain consistent z-score scaling.

    Returns a copy of hist_df with updated columns and a new 'data_source' column.
    """
    df = hist_df.copy()

    # Add new columns if not already present
    if "kenpom_torvik_rating" not in df.columns:
        df["kenpom_torvik_rating"] = np.nan
    if "ap_rank_week6" not in df.columns:
        df["ap_rank_week6"] = np.nan
    if "ap_top12_flag" not in df.columns:
        df["ap_top12_flag"] = np.nan
    if "data_source" not in df.columns:
        df["data_source"] = "proxy"

    # Filter merged to matched rows within enrich seasons only
    matched = merged_df[
        (merged_df["season"].isin(ENRICH_SEASONS)) &
        (merged_df["match_type"].isin(["CONFIRMED", "RANK_MATCH"]))
    ].copy()

    # Build lookup: (season, canonical_team_name) → real feature dict.
    # canonical_team_name is the stable identity key; team_id is NOT used here
    # because team_id assignments may be wrong for some schools (mapping conflicts).
    has_canonical = "canonical_team_name" in matched.columns
    real_lookup: dict[tuple, dict] = {}
    for _, row in matched.iterrows():
        if has_canonical and pd.notna(row.get("canonical_team_name")):
            key: tuple = (int(row["season"]), str(row["canonical_team_name"]))
        elif pd.notna(row.get("team_id")):
            # Fallback: use team_id for files built before canonical_team_name existed
            key = (int(row["season"]), int(row["team_id"]))
        else:
            continue
        real_lookup[key] = {
            "offense_rating":        row["offensive_efficiency"],
            "defense_rating":        row["defensive_efficiency"],
            "efficiency_margin":     row["efficiency_margin"],
            "kenpom_torvik_rating":  row["kenpom_torvik_rating"],
            "ap_rank_week6":         row.get("ap_rank_week6", np.nan),
            "ap_top12_flag":         row.get("ap_top12_flag", np.nan),
        }

    # Pass 1: apply real features to matched teams.
    # hist_df team_name is "T{id}" for proxy-era entries.  For modern (2013+)
    # seasons the canonical_team_name column (if present) takes priority;
    # otherwise fall back to team_id matching so pre-canonical files still work.
    hist_has_canonical = "canonical_team_name" in df.columns
    n_enriched = 0
    for idx, row in df.iterrows():
        if int(row["season"]) not in ENRICH_SEASONS:
            continue
        if hist_has_canonical and pd.notna(row.get("canonical_team_name")):
            key = (int(row["season"]), str(row["canonical_team_name"]))
        else:
            key = (int(row["season"]), int(row["team_id"]))
        if key in real_lookup:
            r = real_lookup[key]
            df.at[idx, "offense_rating"]       = r["offense_rating"]
            df.at[idx, "defense_rating"]       = r["defense_rating"]
            df.at[idx, "efficiency_margin"]    = r["efficiency_margin"]
            df.at[idx, "strength_of_schedule"] = r["kenpom_torvik_rating"]
            df.at[idx, "kenpom_torvik_rating"] = r["kenpom_torvik_rating"]
            if not pd.isna(r["ap_rank_week6"]):
                df.at[idx, "ap_rank_week6"]  = r["ap_rank_week6"]
            if not pd.isna(r["ap_top12_flag"]):
                df.at[idx, "ap_top12_flag"]  = r["ap_top12_flag"]
            df.at[idx, "data_source"] = "real"
            n_enriched += 1

    # Pass 2: impute scale for any 2013–2016 teams still on proxy scale
    # (teams that exist in hist_df but have no match in merged_team_stats)
    for season in ENRICH_SEASONS:
        season_mask = df["season"] == season
        enriched_mask = season_mask & (df["data_source"] == "real")
        proxy_mask    = season_mask & (df["data_source"] == "proxy")
        n_proxy = proxy_mask.sum()
        if n_proxy == 0:
            continue

        # Compute seed-tier medians from enriched peers
        enriched_slice = df[enriched_mask]
        seed_medians: dict[int, dict] = {}
        for seed_val, grp in enriched_slice.groupby("seed"):
            seed_medians[int(seed_val)] = {
                "offense_rating":     grp["offense_rating"].median(),
                "defense_rating":     grp["defense_rating"].median(),
                "efficiency_margin":  grp["efficiency_margin"].median(),
                "strength_of_schedule": grp["strength_of_schedule"].median(),
            }
        # Fall back to overall median if seed tier not present
        overall_medians = {
            "offense_rating":     df[enriched_mask]["offense_rating"].median(),
            "defense_rating":     df[enriched_mask]["defense_rating"].median(),
            "efficiency_margin":  df[enriched_mask]["efficiency_margin"].median(),
            "strength_of_schedule": df[enriched_mask]["strength_of_schedule"].median(),
        }

        for idx in df.index[proxy_mask]:
            seed_val = int(df.at[idx, "seed"])
            meds = seed_medians.get(seed_val, overall_medians)
            df.at[idx, "offense_rating"]       = meds["offense_rating"]
            df.at[idx, "defense_rating"]       = meds["defense_rating"]
            df.at[idx, "efficiency_margin"]    = meds["efficiency_margin"]
            df.at[idx, "strength_of_schedule"] = meds["strength_of_schedule"]
            # AP flag: explicitly 0 (no evidence of top-12 status for unresolved teams)
            df.at[idx, "ap_top12_flag"]        = 0
            df.at[idx, "data_source"]          = "imputed"

    return df, n_enriched


def _print_coverage(enriched_df: pd.DataFrame) -> None:
    """Print matched-team coverage by season."""
    print(f"\n{THN}")
    print(f"  MATCHED-TEAM COVERAGE BY SEASON")
    print(THN)
    print(f"  {'Season':>6}  {'Real':>6}  {'Imputed':>8}  {'Proxy':>6}  {'Total':>6}  {'Coverage':>9}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*9}")
    for season in ENRICH_SEASONS:
        grp  = enriched_df[enriched_df["season"] == season]
        n_r  = (grp["data_source"] == "real").sum()
        n_i  = (grp["data_source"] == "imputed").sum()
        n_p  = (grp["data_source"] == "proxy").sum()
        n_t  = len(grp)
        cov  = n_r / n_t if n_t else 0
        print(f"  {season:>6}  {n_r:>6}  {n_i:>8}  {n_p:>6}  {n_t:>6}  {cov:>9.1%}")

    # Print imputed teams (useful to know which teams got fallback treatment)
    imputed_rows = enriched_df[
        enriched_df["season"].isin(ENRICH_SEASONS) &
        (enriched_df["data_source"] == "imputed")
    ]
    if not imputed_rows.empty:
        print(f"\n  TEAMS USING IMPUTED SCALE  (matched cbb→Kaggle failed)")
        print(f"  {THN}")
        for _, r in imputed_rows.iterrows():
            print(f"  {r['season']}  seed {r['seed']:>2}  {r['team_name']:<12}  "
                  f"ADJOE≈{r['offense_rating']:.1f}  ADJDE≈{r['defense_rating']:.1f}  "
                  f"(seed-tier median)")
    else:
        print(f"\n  All tournament teams in 2013–2016 used real features.")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(SEP)
    print(f"{'INTEGRATE REAL FEATURES  —  2013–2016':^{W}}")
    print(SEP)

    # ── Validate inputs ───────────────────────────────────────────────────
    for path, name in [(HIST_CSV, "historical_team_ratings.csv"),
                       (MERGED_CSV, "merged_team_stats.csv")]:
        if not path.exists():
            sys.exit(f"Error: {path} not found.  Run prerequisite scripts first.")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\n  Loading {HIST_CSV.name} …", end=" ")
    hist_df   = pd.read_csv(HIST_CSV)
    hist_df["team_id"] = hist_df["team_id"].astype(int)
    print(f"{len(hist_df)} rows, {hist_df['season'].nunique()} seasons")

    print(f"  Loading {MERGED_CSV.name} …", end=" ")
    merged_df = pd.read_csv(MERGED_CSV)
    print(f"{len(merged_df)} rows")

    # ── Step 1: Capture baseline backtest results ─────────────────────────
    print(f"\n{THN}")
    print(f"  STEP 1 — Baseline backtest (proxy features)")
    print(THN)
    before_results = _run_backtests("BEFORE")
    for year, r in before_results.items():
        m = _extract_metrics(r)
        print(f"  {year}  {m['total_correct']:>2}/{m['total_possible']}  ({m['accuracy']:.1%})"
              f"  champ={'✓' if m['champ_correct'] else '✗'}  "
              f"FF={m['ff_correct']}/4  {m['pred_champ']}")

    # ── Step 2: Back up original CSV ──────────────────────────────────────
    print(f"\n  Backing up original → {BACKUP_CSV.name}")
    hist_df.to_csv(BACKUP_CSV, index=False)

    # ── Step 3: Enrich features ───────────────────────────────────────────
    print(f"\n{THN}")
    print(f"  STEP 2 — Enrich 2013–2016 with real ADJOE/ADJDE/BARTHAG")
    print(THN)

    enriched_df, n_enriched = _enrich_historical(hist_df, merged_df)
    _print_coverage(enriched_df)
    print(f"\n  Enriched {n_enriched} teams with real features.")

    # ── Step 4: Recompute ratings ─────────────────────────────────────────
    print(f"\n  Recomputing team_rating + champion_profile_score … ", end="")
    # Run rate_teams_by_season on ALL seasons (z-scores are within-season,
    # so pre-2013 seasons are unaffected by the enrichment).
    cols_to_keep = [c for c in enriched_df.columns
                    if c not in ("team_rating", "champion_profile_score",
                                 "_rating_zscore", "_champion_zscore", "profile_score")]
    base_df  = enriched_df[cols_to_keep].copy()
    rated_df = rate_teams_by_season(base_df)
    print("done")

    print(f"  Recomputing profile_score (1990–2016) … ", end="")
    # Merge profile_score back in (compute over the full dataset for stable percentiles)
    scored_df = compute_profile_scores_df(rated_df)
    print(f"done  (2013–16 enrich mean={scored_df[scored_df['season'].isin(ENRICH_SEASONS)]['profile_score'].mean():.4f})")

    # ── Step 5: Spot-check enriched values ───────────────────────────────
    print(f"\n{THN}")
    print(f"  SAMPLE — 2016 enriched teams (top 8 by team_rating)")
    print(THN)
    s16 = scored_df[scored_df["season"] == 2016].sort_values("team_rating", ascending=False).head(8)
    for _, r in s16.iterrows():
        src = r.get("data_source", "?")
        ps  = r.get("profile_score", float("nan"))
        print(f"  seed {r['seed']:>2}  {r['team_name']:<10}  "
              f"ADJOE={r['offense_rating']:>6.1f}  ADJDE={r['defense_rating']:>6.1f}  "
              f"team_rating={r['team_rating']:.4f}  "
              f"ps={ps:.4f}  [{src}]")

    # ── Step 6: Save ──────────────────────────────────────────────────────
    print(f"\n  Saving enriched ratings → {HIST_CSV.name} … ", end="")
    scored_df.to_csv(HIST_CSV, index=False)
    print("done")
    print(f"  Rows: {len(scored_df)}   Columns: {len(scored_df.columns)}")

    # ── Step 7: Re-run backtests with enriched features ───────────────────
    print(f"\n{THN}")
    print(f"  STEP 3 — Post-enrichment backtest (real features)")
    print(THN)

    # team_selector loads ratings fresh from disk on each run_backtest call —
    # no cache reset needed.
    after_results = _run_backtests("AFTER")
    for year, r in after_results.items():
        m = _extract_metrics(r)
        print(f"  {year}  {m['total_correct']:>2}/{m['total_possible']}  ({m['accuracy']:.1%})"
              f"  champ={'✓' if m['champ_correct'] else '✗'}  "
              f"FF={m['ff_correct']}/4  {m['pred_champ']}")

    # ── Step 8: Print comparison ──────────────────────────────────────────
    _print_comparison(before_results, after_results)

    print(f"\n{SEP}")
    print(f"  Enriched ratings saved → {HIST_CSV.name}")
    print(f"  Backup (proxy)         → {BACKUP_CSV.name}")
    print(SEP)


if __name__ == "__main__":
    main()
