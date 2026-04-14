import json
from pathlib import Path
from collections import Counter

new = json.loads(Path('/Users/ethankatz/march_madness/data/processed/sweep_2025.json').read_text())

old_path = Path('/tmp/sweep_2025_before_champ.json')
if old_path.exists():
    old = json.loads(old_path.read_text())
    old_seeds = [v['pred_seed'] for v in old.values() if 'error' not in v]
else:
    old_seeds = [2,3,2,3,2,2,2,2,2,3,2,2,3,2,4,2,3,2,2,3,2,3,2,2,2,2,2,2,2,2,3,2,3,3,3]

valid = {y: v for y, v in new.items() if 'error' not in v}
new_seeds = [v['pred_seed'] for v in valid.values()]
act_seeds = [v['act_seed'] for v in valid.values()]

W = 60
print('=' * W)
print('CHAMPION SELECTION — BEFORE vs AFTER'.center(W))
print('=' * W)
print(f"{'Metric':<32}  {'Before':>8}  {'After':>8}")
print(f"{'─'*32}  {'─'*8}  {'─'*8}")

n = len(valid)
old_champs = 3
new_champs = sum(1 for v in valid.values() if v['champ'])
old_acc = 1338 / 2205
new_acc = sum(v['correct'] for v in valid.values()) / sum(v['possible'] for v in valid.values())
old_ff = 48 / 140
new_ff = sum(v['ff'] for v in valid.values()) / (n * 4)

print(f"{'Overall accuracy':<32}  {old_acc:>8.1%}  {new_acc:>8.1%}")
print(f"{'Champions correct':<32}  {old_champs:>5}/{n:<2}  {new_champs:>5}/{n:<2}")
print(f"{'Champion accuracy':<32}  {old_champs/n:>8.1%}  {new_champs/n:>8.1%}")
print(f"{'Final Four accuracy':<32}  {old_ff:>8.1%}  {new_ff:>8.1%}")

print()
print('─' * W)
print('PREDICTED CHAMPION SEED DISTRIBUTION')
print('─' * W)

old_dist = Counter(old_seeds)
new_dist = Counter(new_seeds)
act_dist = Counter(act_seeds)

print(f"{'Seed':<6}  {'Before':>8}  {'After':>8}  {'Actual':>8}")
for s in sorted(set(old_seeds) | set(new_seeds) | set(act_seeds)):
    print(f"{s:<6}  {old_dist.get(s,0):>8}  {new_dist.get(s,0):>8}  {act_dist.get(s,0):>8}")
print('=' * W)
