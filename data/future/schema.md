# Future Bracket CSV Schema

Input format for `scripts/predict_future_bracket.py`.
One row per team. 64 rows total (16 seeds × 4 regions).

---

## Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `season` | int | Tournament year (e.g. 2026) |
| `team_name_raw` | str | Name exactly as it appears in your data source (KenPom, Barttorvik, etc.) |
| `canonical_team_name` | str | Normalized name used as the team identity throughout the system. Must be unique within the season. |
| `seed` | int | Tournament seed 1–16 |
| `region` | str | One of: `East`, `West`, `South`, `Midwest` |
| `offensive_efficiency` | float | Adjusted offensive efficiency (KenPom/Barttorvik scale, typically 90–130) |
| `defensive_efficiency` | float | Adjusted defensive efficiency (lower = better, typically 85–115) |
| `efficiency_margin` | float | `offensive_efficiency − defensive_efficiency`. Most important predictor. |

**Note:** `region` is listed as optional in some contexts but is required by the prediction engine to build the bracket structure.

---

## Optional Columns

| Column | Type | Description | Default if absent |
|--------|------|-------------|-------------------|
| `kenpom_rank` | int | KenPom national rank (1 = best) | Derived from `efficiency_margin` rank |
| `bart_torvik_rank` | int | Barttorvik national rank (1 = best) | Derived from `efficiency_margin` rank |
| `ap_top12_flag` | int | 1 if team was in AP Top 12 at any point this season, else 0 | 0 |
| `public_pick_pct` | float | Fraction of public brackets picking this team to win the championship (e.g. 0.22 = 22%) | Seed-based default |
| `conference` | str | Conference name (display only) | — |
| `recent_form` | str | Recent win/loss record (display only, e.g. "W8" = 8-game win streak) | — |
| `strength_of_schedule` | float | SOS score (display only) | — |

---

## Key Rules

1. **One row per team.** No duplicates within `(season, region, seed)`.
2. **64 teams total**: exactly 16 seeds (1–16) in each of 4 regions.
3. **`canonical_team_name`** is the identity key used downstream. Use consistent spelling across seasons. If a program changed names, always use the current canonical name.
4. **`efficiency_margin`** drives team ratings. The model normalizes the full field min–max, so relative gaps matter more than absolute values.
5. **`public_pick_pct`** should sum to ≤ 1.0 across the full field. If not provided, the engine uses historical seed-based defaults from `lib/bracket_strategy.py`.

---

## Efficiency Scale Reference

Typical KenPom adjusted efficiency ranges for NCAA tournament teams:

| Seed tier | Typical EM |
|-----------|-----------|
| #1 seeds  | +28 to +36 |
| #2 seeds  | +20 to +28 |
| #3–4 seeds | +12 to +22 |
| #5–8 seeds | +2 to +14  |
| #9–12 seeds | −8 to +4  |
| #13–16 seeds | −22 to −8 |

---

## Example Row

```
2026,Duke,Duke,1,East,122.3,89.8,32.5,2,1,1,0.22,ACC,W10,82.1
```

Interpretation: Duke, seed 1 in the East, has an offensive efficiency of 122.3, defensive efficiency of 89.8, efficiency margin of +32.5, is ranked #2 by KenPom and #1 by Barttorvik, was in the AP Top 12, 22% of public brackets pick them as champion, plays in the ACC, currently on a 10-game win streak, SOS of 82.1.

---

## Template File

`data/future/future_bracket_template.csv` — 64-row placeholder with 2026 season data.
Replace all values with real bracket data before running the prediction script.
