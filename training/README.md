# ForkWise — Training

## Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script. Loads data, trains model, evaluates MRR@3, applies quality gate, saves checkpoint to MLflow + object storage |
| `evaluate.py` | Computes MRR@3, NDCG@3, per-cuisine fairness metrics (safeguarding requirement) |
| `model_stub.py` | `SubstitutionModel` class — shared with serving so checkpoint format matches |
| `watch_trigger.py` | K8S CronJob (every 30 min). Polls `data-proj01/triggers/` for retraining triggers from data team |
| `config.yaml` | Default hyperparameters. CLI args override these for sweep runs |
| `generate_synthetic_data.py` | Creates synthetic train/val/test data for immediate hyperparameter tuning |
| `requirements.txt` | Python dependencies |
| `docker_nvidia/Dockerfile` | NVIDIA CUDA container (RTX 6000 on CHI@UC) |

## Quick Start (on Chameleon CHI@UC node)

```bash
# 1. Generate synthetic data
python training/generate_synthetic_data.py

# 2. Build Docker container
docker build -t train:latest -f training/docker_nvidia/Dockerfile .

# 3. Run training
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  --shm-size=12g --network host \
  train:latest \
  python training/train.py \
    --config training/config.yaml \
    --dataset /workspace/data/processed/train.json \
    --embed_dim 128 \
    --run_name gismo-emb128-gpu \
    --mlflow_tracking_uri http://<HOST_IP>:5000
```

## Hyperparameter Sweep

Override config values via CLI:

```bash
# Larger embeddings
--embed_dim 256 --run_name gismo-emb256

# Lower learning rate for big models
--embed_dim 512 --lr 0.0005 --run_name gismo-emb512-lr5e4

# More epochs
--epochs 50 --run_name gismo-50ep

# Bigger batches (smoother gradients, better GPU util)
--batch_size 256 --run_name gismo-bs256

# Higher contrastive margin
--margin 1.0 --run_name gismo-margin1.0
```

## Quality Gate

- Threshold: MRR@3 >= 0.30
- Random baseline: ~0.10
- Models that pass: saved to `models-proj01/production/subst_model_current.pth` + registered in MLflow
- Models that fail: logged in MLflow with `quality_gate: failed` tag, not saved

## Object Storage Paths

```
READS:  data-proj01/raw/recipe1msubs/{train,val,test}.json
READS:  data-proj01/triggers/retrain_*.json
READS:  data-proj01/processed/train_v*.json
WRITES: models-proj01/checkpoints/subst_model_v{run_id}.pth
WRITES: models-proj01/production/subst_model_current.pth
WRITES: MLflow model registry
```

## Swap to Real Data

When data team uploads Recipe1MSubs, just change the `--dataset` path:

```bash
--dataset data-proj01/raw/recipe1msubs/train.json
```

Same model, same hyperparameters — no code changes needed.
