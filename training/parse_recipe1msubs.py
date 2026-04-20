import pickle, json, random, sys
from pathlib import Path

random.seed(42)

SUBS_DIR = Path('data/recipe1msubs')
LAYER1   = 'data/recipe1m/layer1.json'
OUT      = Path('data/processed')
HOLDOUT  = Path('data/production_holdout')
OUT.mkdir(parents=True, exist_ok=True)
HOLDOUT.mkdir(parents=True, exist_ok=True)

# Step 1: Load Recipe1M layer1.json for recipe context
print('Loading layer1.json...')
layer1 = json.load(open(LAYER1))
ctx_map = {}
for r in layer1:
    ingrs = [i['text'].lower().strip() for i in r.get('ingredients', [])]
    ctx_map[r['id']] = ingrs
print(f'Recipe1M context map: {len(ctx_map):,} recipes')

# Step 2: Inspect pkl structure first
print('Inspecting train_comments_subs.pkl...')
with open(SUBS_DIR / 'train_comments_subs.pkl', 'rb') as f:
    raw = pickle.load(f, encoding='latin1')

print(f'Type: {type(raw).__name__}  Len: {len(raw)}')
item = raw[0] if isinstance(raw, list) else list(raw.values())[0]
print(f'Item type: {type(item).__name__}')
if isinstance(item, dict):
    print(f'Item keys: {list(item.keys())}')
    print(f'Item sample: {item}')
else:
    print(f'Item value: {item}')
sys.exit(0)
