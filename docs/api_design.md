# API Design — March Madness Bracket Predictor

Lightweight REST API wrapping the Python prediction engine.
Recommended stack: **FastAPI** + **uvicorn** (Python, minimal dependencies).

---

## Suggested File Structure

```
api/
  main.py           # FastAPI app, route definitions
  models.py         # Pydantic request/response schemas
  engine.py         # Thin wrapper calling lib/ prediction code
  requirements.txt  # fastapi, uvicorn, pandas
```

---

## Endpoints

---

### GET /health

Check that the service is running and model data files are present.

**Request:** none

**Response:**
```json
{
  "status": "ok",
  "model_files": {
    "seed_probabilities": true,
    "optimal_bracket_structure": true
  },
  "version": "1.0.0"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### POST /predict-bracket

Generate a single best bracket from a 64-team field.
Teams are provided inline as a list; no CSV upload required.

**Request schema:**
```json
{
  "season": 2026,
  "mode": "balanced",
  "pool_size": 100,
  "teams": [
    {
      "canonical_team_name": "Duke",
      "seed": 1,
      "region": "East",
      "offensive_efficiency": 122.3,
      "defensive_efficiency": 89.8,
      "efficiency_margin": 32.5,
      "ap_top12_flag": 1,
      "public_pick_pct": 0.22
    }
  ],
  "public_picks_override": {
    "Duke": 0.22,
    "Kansas": 0.18
  }
}
```

**Field notes:**
- `season`: integer, display/naming only
- `mode`: `"conservative"` | `"balanced"` | `"upset_heavy"` (default: `"balanced"`)
- `pool_size`: integer, used for value-score strategy weighting (default: 100)
- `teams`: array of 64 team objects; `canonical_team_name`, `seed`, `region`, `offensive_efficiency`, `defensive_efficiency`, `efficiency_margin` are required per team
- `public_picks_override`: optional dict; wins over per-team `public_pick_pct` values

**Response schema:**
```json
{
  "season": 2026,
  "mode": "balanced",
  "champion": {
    "name": "Kansas",
    "seed": 1,
    "region": "West"
  },
  "final_four": [
    {"winner": "Kansas", "winner_seed": 1, "loser": "Gonzaga", "loser_seed": 2},
    {"winner": "Duke",   "winner_seed": 1, "loser": "Houston", "loser_seed": 3}
  ],
  "championship": {
    "winner": "Kansas", "winner_seed": 1,
    "loser": "Duke",    "loser_seed": 1
  },
  "upsets": [
    {"round": "R64", "winner": "NC State", "winner_seed": 11,
     "loser": "Xavier", "loser_seed": 6}
  ],
  "bracket": { ... }
}
```

**Example request:**
```bash
curl -X POST http://localhost:8000/predict-bracket \
  -H "Content-Type: application/json" \
  -d '{
    "season": 2026,
    "mode": "balanced",
    "pool_size": 100,
    "teams": [
      {"canonical_team_name":"Duke","seed":1,"region":"East",
       "offensive_efficiency":122.3,"defensive_efficiency":89.8,
       "efficiency_margin":32.5,"ap_top12_flag":1,"public_pick_pct":0.22},
      ...
    ]
  }'
```

---

### POST /generate-portfolio

Generate N diverse brackets optimized for pool-winning expected value.
Same input as `/predict-bracket` plus `n_brackets`.

**Request schema:**
```json
{
  "season": 2026,
  "mode": "balanced",
  "pool_size": 10000,
  "n_brackets": 5,
  "teams": [ ... ],
  "public_picks_override": {}
}
```

**Field notes:**
- `n_brackets`: number of portfolio brackets (1–20); default 5
- `pool_size` drives the composite scoring: large pools → weight value over probability

**Response schema:**
```json
{
  "season": 2026,
  "pool_size": 10000,
  "n_brackets": 5,
  "base_champion": {"name": "Kansas", "seed": 1},
  "brackets": [
    {
      "index": 1,
      "champion": "NC State",
      "seed": 11,
      "region": "East",
      "win_prob": 0.031,
      "public_pct": 0.001,
      "value_score": 31.0,
      "composite": 0.248,
      "rationale": "High-value contrarian pick (#11) — underrepresented in public brackets (0.1%)",
      "ev_note": "In a 10000-person pool: if NC State wins, ~10 people share the pot vs ~100 for a 1-seed",
      "bracket": { ... }
    }
  ]
}
```

**Example request:**
```bash
curl -X POST http://localhost:8000/generate-portfolio \
  -H "Content-Type: application/json" \
  -d '{
    "season": 2026,
    "mode": "balanced",
    "pool_size": 10000,
    "n_brackets": 5,
    "teams": [ ... ]
  }'
```

---

## Implementation Notes

### `api/engine.py` — bridge between API and lib/

```python
from lib.team_selector import simulate_bracket
from lib.bracket_strategy import extract_candidates, generate_portfolio, format_portfolio
from scripts.predict_future_bracket import _build_teams_override, _extract_public_picks

def predict(teams_payload, mode, pool_size, public_picks_override):
    df = pd.DataFrame(teams_payload)
    teams_override = _build_teams_override(df)
    public_picks   = _extract_public_picks(df, public_picks_override or {})
    bracket        = simulate_bracket(mode, _teams_override=teams_override)
    return bracket, public_picks

def portfolio(teams_payload, mode, pool_size, n_brackets, public_picks_override):
    bracket, public_picks = predict(teams_payload, mode, pool_size, public_picks_override)
    entries = generate_portfolio(bracket, n_brackets, pool_size, public_picks or None)
    return bracket, entries
```

### Error handling

| Scenario | HTTP status | message |
|----------|-------------|---------|
| Missing required team field | 422 | `"Missing field: efficiency_margin in team Duke"` |
| Wrong region value | 422 | `"Invalid region 'Northeast' — must be East/West/South/Midwest"` |
| Not 64 teams | 400 | `"Expected 64 teams, received 63"` |
| Duplicate seed/region | 400 | `"Duplicate team at East seed 1"` |
| Internal model error | 500 | `"Bracket simulation failed: {detail}"` |

### CORS

Enable CORS for the frontend domain:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["https://your-frontend.com"],
                   allow_methods=["*"], allow_headers=["*"])
```

---

## Running Locally

```bash
pip install fastapi uvicorn pandas
uvicorn api.main:app --reload --port 8000
```

Interactive docs auto-generated at: `http://localhost:8000/docs`
