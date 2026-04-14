import json
from pathlib import Path

data = json.loads(Path('/Users/ethankatz/march_madness/data/processed/sweep_2025.json').read_text())
valid = {y: v for y, v in data.items() if 'error' not in v}

prev_correct = 6
prev_acc = 1341 / 2205
prev_ff = 48 / 140

n = len(valid)
cur_correct = sum(1 for v in valid.values() if v['champ'])
cur_acc = sum(v['correct'] for v in valid.values()) / sum(v['possible'] for v in valid.values())
cur_ff = sum(v['ff'] for v in valid.values()) / (n * 4)

W = 62
print('=' * W)
print('RESULTS'.center(W))
print('=' * W)
print(f"{'Metric':<30}  {'Prev':>8}  {'Now':>8}")
print(f"{'─'*30}  {'─'*8}  {'─'*8}")
print(f"{'Overall accuracy':<30}  {prev_acc:>8.1%}  {cur_acc:>8.1%}")
print(f"{'Champions correct':<30}  {prev_correct:>5}/{n:<2}  {cur_correct:>5}/{n:<2}")
print(f"{'Champion accuracy':<30}  {prev_correct/n:>8.1%}  {cur_correct/n:>8.1%}")
print(f"{'Final Four accuracy':<30}  {prev_ff:>8.1%}  {cur_ff:>8.1%}")

print()
print('─' * W)
print('CORRECTLY PREDICTED CHAMPIONS')
print('─' * W)
for yr, v in sorted(valid.items()):
    if v['champ']:
        print(f"{yr}  {v['pred_champ']} (seed {v['pred_seed']})")

print()
print('─' * W)
print('CORRECT SEED, WRONG TEAM')
print('─' * W)
for yr, v in sorted(valid.items()):
    if not v['champ'] and v['pred_seed'] == v['act_seed']:
        print(f"{yr}  pred={v['pred_champ']} (s{v['pred_seed']})  act={v['act_champ']} (s{v['act_seed']})")

print('=' * W)
