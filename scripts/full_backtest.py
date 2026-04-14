"""
scripts/full_backtest.py
Run the full balanced backtest sweep across all available years (excluding 2020).
Prints per-year results and an aggregate summary.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.backtest import run_backtest

years = [y for y in range(1990, 2026) if y != 2020]
results = {}

for year in years:
    try:
        r = run_backtest(year, mode='balanced')

        correct  = r['total_correct']
        possible = r['total_possible']
        pct      = correct / possible if possible else 0.0

        pred_name = r['predicted_champion']['name']
        pred_seed = r['predicted_champion']['seed']
        act_name  = r['actual_champion']['name']
        act_seed  = r['actual_champion']['seed']

        champ_ok  = pred_name == act_name
        pred_ff   = {t['name'] for t in r['predicted_final_four']}
        act_ff    = {t['name'] for t in r['actual_final_four']}
        ff_ok     = len(pred_ff & act_ff)

        mark = '✓' if champ_ok else '✗'
        print(f"{year}  {correct}/{possible} ({pct:.1%})  champ={mark}  FF={ff_ok}/4")
        print(f"    pred: {pred_name} (seed {pred_seed})")
        print(f"    act : {act_name} (seed {act_seed})")

        results[str(year)] = {
            'correct':    correct,
            'possible':   possible,
            'champ':      champ_ok,
            'ff':         ff_ok,
            'pred_champ': pred_name,
            'act_champ':  act_name,
            'pred_seed':  pred_seed,
            'act_seed':   act_seed,
        }

    except Exception as e:
        results[str(year)] = {'error': str(e)}
        print(f"{year} ERROR: {e}")

# ── Save raw results ──────────────────────────────────────────────────────────
out = PROJECT_ROOT / 'data' / 'processed' / 'sweep_2025.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
valid = {y: v for y, v in results.items() if 'error' not in v}

total_correct  = sum(v['correct']  for v in valid.values())
total_possible = sum(v['possible'] for v in valid.values())
total_champs   = sum(1 for v in valid.values() if v['champ'])
total_ff       = sum(v['ff']       for v in valid.values())
n_years        = len(valid)

W = 60
print()
print('=' * W)
print(f"  FULL BACKTEST SUMMARY  ({min(valid)} – {max(valid)},  {n_years} seasons)")
print('=' * W)
print(f"  Overall accuracy  : {total_correct}/{total_possible}  ({total_correct/total_possible:.1%})")
print(f"  Champions correct : {total_champs}/{n_years}  ({total_champs/n_years:.1%})")
print(f"  Final Four (avg)  : {total_ff/(n_years*4):.1%}  ({total_ff} of {n_years*4} slots)")
print('=' * W)
print(f"  Results saved → {out}")
