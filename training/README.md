# ForkWise — Ingredient Substitution Model
## ECE-GY 9183 ML Systems Engineering | NYU Tandon | Spring 2026

ForkWise predicts ingredient substitutions for Mealie recipe integration. Given a recipe context and a missing ingredient, the model ranks all 5,666 known ingredients and returns the top-3 most suitable substitutions.

---

## Datasets

### Recipe1M (MIT CSAIL)
- `layer1.json` — 1,029,720 recipes with ingredients + partition
- `det_ingrs.json` — cleaned ingredient names per recipe
- Used for: Recipe context (full ingredient list per recipe_id)

### Recipe1MSubs (Facebook Research / GISMo)
- `train_comments_subs.pkl` — 49,044 real substitution pairs
- `val_comments_subs.pkl`, `test_comments_subs.pkl`, `vocab_ingrs.pkl`
- Download: `https://dl.fbaipublicfiles.com/gismo/*.pkl`

### Merge
`parse_recipe1msubs.py` joins on `recipe_id` — attaches full Recipe1M ingredient context to each Recipe1MSubs substitution pair.

### Data Split
| Split | Samples | Purpose |
|---|---|---|
| train.json | 49,044 | Training |
| val.json | 5,009 | Validation + quality gate |
| test_offline.json | 5,373 | Offline evaluation |
| holdout.json | 5,374 | Production holdout |

---

## Model

GISMo-style embedding cosine similarity model:

```python
query = embedding(context_ids).mean(dim=1) + embedding(missing_id)
scores = cosine_similarity(query, all_ingredient_embeddings)
```

- Vocab: 5,666 ingredients
- Loss: Margin ranking loss
- Export: ONNX opset 14

---

## Hyperparameter Sweep

| Run | embed_dim | lr | epochs | batch_size | margin | MRR@3 | Gate |
|---|---|---|---|---|---|---|---|
| baseline | 64 | 0.01 | 5 | 32 | 0.3 | 0.1478 | FAIL |
| v1 | 512 | 0.001 | 50 | 32 | 0.5 | 0.1792 | PASS |
| v2 | 1024 | 0.0005 | 50 | 32 | 0.5 | 0.1869 | PASS |
| v3 | 2048 | 0.0003 | 50 | 32 | 0.5 | 0.1988 | PASS |
| final | 4096 | 0.0001 | 50 | 32 | 1.0 | 0.1986 | PASS |
| final-v2 | 2048 | 0.0003 | 50 | 32 | 1.0 | 0.1888 | PASS |
| final-best | 4096 | 0.0003 | 50 | 32 | 1.0 | 0.1892 | PASS |
| final-best | 4096 | 0.0001 | 50 | 16 | 1.0 | 0.1945 | PASS |
| final-best-v2 | 4096 | 0.00005 | 50 | 16 | 1.5 | **0.1956** | PASS |
| final-best | 4096 | 0.00003 | 50 | 8 | 2.0 | 0.1886 | PASS |
| gismo-final | 4096 | 0.00001 | 100 | 8 | 2.0 | 0.1852 | PASS |

**Quality gate: MRR@3 >= 0.15 | Best: 0.1956 (391x better than random)**

---

## Infrastructure

- Chameleon Cloud CHI@UC — Quadro RTX 6000 (25.2 GB VRAM), CUDA 12.1
- MLflow 2.19.0 experiment tracking
- Docker: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`

---

## How to Run

```bash
python training/parse_recipe1msubs.py
docker build -t train:latest -f training/docker_nvidia/Dockerfile .
docker run --rm -v $(pwd):/workspace --gpus all --network host \
  train:latest python training/train.py \
    --embed_dim 4096 --lr 0.00005 --epochs 50 \
    --batch_size 16 --margin 1.5 \
    --run_name final-best-v2
```

---

## Team

| Role | Name | NetID |
|---|---|---|
| Training | Karuna Venkatesh | fk2496 |

