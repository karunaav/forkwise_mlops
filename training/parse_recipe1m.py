import json, random
from pathlib import Path
from collections import defaultdict

random.seed(42)
LAYER1    = 'data/recipe1m/layer1.json'
DET_INGRS = 'data/recipe1m/det_ingrs.json'
OUT       = Path('data/processed')
HOLDOUT   = Path('data/production_holdout')
OUT.mkdir(parents=True, exist_ok=True)
HOLDOUT.mkdir(parents=True, exist_ok=True)

print('Loading layer1.json...')
layer1 = json.load(open(LAYER1))
print(f'Total recipes: {len(layer1):,}')

print('Loading det_ingrs.json...')
det_ingrs = json.load(open(DET_INGRS))
det_map   = {r['id']: r for r in det_ingrs}
print(f'Det ingrs: {len(det_map):,}')

def get_ingrs(rid):
    if rid not in det_map: return []
    return [i['text'].lower().strip()
            for i in det_map[rid].get('ingredients', [])
            if i.get('valid', True) and i.get('text', '').strip()]

print('Building co-occurrence (5-10 min)...')
cooccur = defaultdict(lambda: defaultdict(int))
for r in layer1:
    if r.get('partition') != 'train': continue
    ingrs = get_ingrs(r['id'])
    if len(ingrs) < 2: continue
    for a in ingrs:
        for b in ingrs:
            if a != b: cooccur[a][b] += 1
print(f'Unique ingredients: {len(cooccur):,}')

subs_cands = {a: [i for i, _ in sorted(co.items(), key=lambda x: -x[1])[:20]]
              for a, co in cooccur.items()}

def make_samples(partition, max_n):
    out = []
    for r in layer1:
        if r.get('partition') != partition: continue
        ingrs = get_ingrs(r['id'])
        if len(ingrs) < 3: continue
        for orig in ingrs:
            if orig not in subs_cands: continue
            repls = [c for c in subs_cands[orig] if c not in ingrs]
            if not repls: continue
            out.append({
                'recipe_id':   r['id'],
                'original':    orig,
                'replacement': random.choice(repls[:5]),
                'ingredients': ingrs,
                'cuisine':     'unknown',
            })
        if max_n and len(out) >= max_n: break
    random.shuffle(out)
    return out

train = make_samples('train', 50000)
val   = make_samples('val',   5000)
test  = make_samples('test',  10000)

# Split test: half for offline eval, half as production holdout (Prof. Fund's requirement)
mid = len(test) // 2
test_offline  = test[:mid]
test_holdout  = test[mid:]

for name, data in [('train', train), ('val', val), ('test_offline', test_offline)]:
    with open(OUT / f'{name}.json', 'w') as f: json.dump(data, f)
    print(f'{name}: {len(data):,} samples')

with open(HOLDOUT / 'holdout.json', 'w') as f: json.dump(test_holdout, f)
print(f'production_holdout: {len(test_holdout):,} samples (DO NOT TOUCH until demo)')
print('Done! Ready for training.')
