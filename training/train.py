#!/usr/bin/env python3
"""
train.py — Main training script for ForkWise ingredient substitution model.

Loads data (local disk or object storage), trains embedding model with
contrastive margin ranking loss, evaluates MRR@3, applies quality gate,
saves checkpoint and logs everything to MLflow.

Usage (local data — use this now per Hivansh):
  python train.py --config config.yaml \
                  --dataset /workspace/data/processed/train.json \
                  --run_name gismo-emb128-gpu \
                  --mlflow_tracking_uri http://<HOST_IP>:5000

Usage (object storage — swap to this when data team uploads):
  python train.py --config config.yaml \
                  --dataset data-proj01/raw/recipe1msubs/train.json \
                  --run_name gismo-real-data \
                  --mlflow_tracking_uri http://<HOST_IP>:5000

CLI overrides for hyperparameter sweep:
  python train.py --config config.yaml \
                  --embed_dim 256 --lr 0.0005 --epochs 30 \
                  --run_name sweep-emb256-lr5e4-30ep ...
"""

import argparse
import json
import os
import random
import sys
import time

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from datetime import datetime

from model_stub import SubstitutionModel
from evaluate import evaluate_model


# =====================================================================
# DATA LOADING
# =====================================================================

def get_s3_client():
    """Create boto3 S3 client for Chameleon object storage."""
    import boto3
    return boto3.client('s3',
        endpoint_url=os.getenv('OS_ENDPOINT'),
        aws_access_key_id=os.getenv('OS_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('OS_SECRET_KEY'))


def load_data(path):
    """
    Load JSON data from local disk or object storage.

    If path starts with 'data-proj01/', reads from object storage.
    Otherwise reads from local filesystem.

    This is how you swap from synthetic → real data without changing code:
      Local:   --dataset /workspace/data/processed/train.json
      Bucket:  --dataset data-proj01/raw/recipe1msubs/train.json
    """
    if path.startswith('data-proj01/'):
        s3 = get_s3_client()
        bucket_key = path.replace('data-proj01/', '')
        obj = s3.get_object(Bucket='data-proj01', Key=bucket_key)
        return json.loads(obj['Body'].read())
    return json.loads(open(path).read())


# =====================================================================
# VOCABULARY
# =====================================================================

def build_vocab(train_data):
    """
    Build ingredient vocabulary from training data.

    Scans all ingredients and assigns each a unique integer ID.
    Special tokens: <PAD>=0, <UNK>=1.

    Returns:
        dict mapping ingredient string → integer ID
    """
    ingredients = set()
    for record in train_data:
        ingredients.add(record['original'].lower().strip())
        ingredients.add(record['replacement'].lower().strip())
        for ing in record.get('ingredients', []):
            if isinstance(ing, str):
                ingredients.add(ing.lower().strip())

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for ing in sorted(ingredients):
        if ing and ing not in vocab:
            vocab[ing] = len(vocab)
    return vocab


# =====================================================================
# BATCH PREPARATION
# =====================================================================

def prepare_batch(records, vocab, context_len=20):
    """
    Convert a list of records into training tensors.

    For each record:
      - context_ids:  recipe ingredient IDs (padded to context_len)
      - missing_id:   the original ingredient to substitute
      - positive_id:  the correct substitution (ground truth)
      - negative_id:  a random wrong ingredient (contrastive negative)

    Returns tuple of 4 tensors.
    """
    contexts, missing_ids, positive_ids, negative_ids = [], [], [], []
    all_ingredients = list(vocab.keys())

    for record in records:
        original = record['original'].lower().strip()
        replacement = record['replacement'].lower().strip()
        orig_id = vocab.get(original, 1)
        repl_id = vocab.get(replacement, 1)

        # Build context from recipe ingredients
        recipe_ings = record.get('ingredients', [])
        ctx = []
        for ing in recipe_ings:
            if isinstance(ing, str):
                ctx.append(vocab.get(ing.lower().strip(), 1))
        ctx = ctx[:context_len]
        ctx += [0] * (context_len - len(ctx))   # pad with <PAD>=0

        # Random negative sampling
        neg_ing = random.choice(all_ingredients)
        while neg_ing in (original, replacement, '<PAD>', '<UNK>'):
            neg_ing = random.choice(all_ingredients)
        neg_id = vocab.get(neg_ing, 1)

        contexts.append(ctx)
        missing_ids.append(orig_id)
        positive_ids.append(repl_id)
        negative_ids.append(neg_id)

    return (torch.tensor(contexts), torch.tensor(missing_ids),
            torch.tensor(positive_ids), torch.tensor(negative_ids))


# =====================================================================
# TRAINING LOOP
# =====================================================================

def train_one_epoch(model, optimizer, data, vocab, config, device):
    """
    Train for one epoch using contrastive margin ranking loss.

    Loss function: MarginRankingLoss
      - positive_similarity should exceed negative_similarity by at least `margin`
      - Pushes correct substitution embeddings closer to the query
      - Pushes random wrong ingredients further away
    """
    model.train()
    random.shuffle(data)
    batch_size = config['batch_size']
    total_loss, num_batches = 0, 0

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        contexts, missing, positives, negatives = prepare_batch(
            batch, vocab, config.get('context_len', 20))

        # Move to GPU (or CPU)
        contexts = contexts.to(device)
        missing = missing.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        # Get embeddings
        ctx_embed = model.embedding(contexts).mean(dim=1)   # avg pool
        miss_embed = model.embedding(missing)
        pos_embed = model.embedding(positives)
        neg_embed = model.embedding(negatives)

        # Query = context + missing
        query = ctx_embed + miss_embed

        # Cosine similarity with positive and negative
        pos_sim = nn.functional.cosine_similarity(query, pos_embed)
        neg_sim = nn.functional.cosine_similarity(query, neg_embed)

        # Margin ranking loss: pos_sim should be > neg_sim + margin
        target = torch.ones(pos_sim.shape).to(device)
        loss = nn.functional.margin_ranking_loss(
            pos_sim, neg_sim, target,
            margin=config.get('margin', 0.5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# =====================================================================
# CHECKPOINT SAVE + REGISTER
# =====================================================================

def save_checkpoint(model, vocab, config, metrics, run_name):
    """
    Save model checkpoint locally and attempt object storage upload.

    Paths (from system_impl_guide):
      models-proj01/checkpoints/subst_model_v{run_id}.pth
      models-proj01/production/subst_model_current.pth  ← serving reads this
    """
    checkpoint_path = f'/tmp/{run_name}.pth'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': config,
        'metrics': metrics,
    }
    torch.save(checkpoint, checkpoint_path)
    mlflow.log_artifact(checkpoint_path)

    # Try object storage upload (skip if buckets not ready yet)
    try:
        s3 = get_s3_client()
        run_id = mlflow.active_run().info.run_id

        with open(checkpoint_path, 'rb') as f:
            s3.put_object(
                Bucket='models-proj01',
                Key=f'checkpoints/subst_model_v{run_id}.pth',
                Body=f)
        with open(checkpoint_path, 'rb') as f:
            s3.put_object(
                Bucket='models-proj01',
                Key='production/subst_model_current.pth',
                Body=f)
        print(f'  Checkpoint uploaded to models-proj01/production/')
    except Exception as e:
        print(f'  Object storage not available (OK for now): {e}')
        print(f'  Checkpoint saved locally: {checkpoint_path}')

    # Log model artifact to MLflow
    mlflow.pytorch.log_model(model, 'model')
    print(f'  Model logged to MLflow artifacts')


# =====================================================================
# MAIN TRAINING FUNCTION
# =====================================================================

def train(config, dataset_path, run_name, mlflow_uri):
    """Full training pipeline: load → train → evaluate → quality gate."""

    # ---- Device setup ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

    # ---- MLflow setup ----
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment('ingredient-substitution')

    with mlflow.start_run(run_name=run_name):

        # Log all hyperparameters for reproducibility
        mlflow.log_params({
            'embed_dim': config['embed_dim'],
            'hidden_dim': config.get('hidden_dim', 'N/A'),
            'context_len': config.get('context_len', 20),
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'lr': config['lr'],
            'margin': config.get('margin', 0.5),
            'quality_gate_mrr': config['quality_gate_mrr'],
            'dataset': dataset_path,
            'device': str(device),
            'torch_version': torch.__version__,
            'run_timestamp': datetime.utcnow().isoformat(),
        })
        if device.type == 'cuda':
            mlflow.log_param('gpu', torch.cuda.get_device_name(0))
        else:
            mlflow.log_param('gpu', 'none-cpu')

        # ---- Load data ----
        print(f'\nLoading data from {dataset_path}...')
        train_data = load_data(dataset_path)
        val_path = dataset_path.replace('train', 'val')
        val_data = load_data(val_path)
        print(f'  Train: {len(train_data)} records')
        print(f'  Val:   {len(val_data)} records')

        # ---- Build vocabulary ----
        vocab = build_vocab(train_data)
        vocab_size = len(vocab)
        mlflow.log_param('vocab_size', vocab_size)
        print(f'  Vocabulary: {vocab_size} ingredients')

        # Save vocab as artifact (serving needs this)
        vocab_path = '/tmp/vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=2)
        mlflow.log_artifact(vocab_path)

        # ---- Initialize model ----
        model = SubstitutionModel(
            vocab_size=vocab_size,
            embed_dim=config['embed_dim']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        param_count = sum(p.numel() for p in model.parameters())
        mlflow.log_param('param_count', param_count)
        print(f'  Parameters: {param_count:,}')

        # ---- TRAINING LOOP ----
        print(f'\nTraining for {config["epochs"]} epochs...')
        start_time = time.time()

        for epoch in range(config['epochs']):
            epoch_start = time.time()
            loss = train_one_epoch(
                model, optimizer, train_data, vocab, config, device)
            epoch_time = time.time() - epoch_start

            mlflow.log_metric('train_loss', loss, step=epoch)
            mlflow.log_metric('epoch_time_sec', epoch_time, step=epoch)
            print(f'  Epoch {epoch + 1:3d}/{config["epochs"]}: '
                  f'loss={loss:.4f}  time={epoch_time:.1f}s')

        total_time = time.time() - start_time
        mlflow.log_metric('total_training_time_sec', total_time)
        print(f'\nTraining complete in {total_time:.1f}s')

        # ---- EVALUATION ----
        model_cpu = model.cpu()
        print('\nEvaluating on validation set...')
        metrics = evaluate_model(model_cpu, val_data, vocab)
        mlflow.log_metrics(metrics)

        print('  Results:')
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f'    {k}: {v:.4f}')
            else:
                print(f'    {k}: {v}')

        # ---- QUALITY GATE ----
        threshold = config['quality_gate_mrr']
        print(f'\nQuality gate: MRR@3 >= {threshold}')

        if metrics['mrr_at_3'] >= threshold:
            mlflow.set_tag('quality_gate', 'passed')
            print(f'  PASSED — MRR@3 = {metrics["mrr_at_3"]:.4f} >= {threshold}')
            print(f'  Saving and registering model...')
            save_checkpoint(model_cpu, vocab, config, metrics, run_name)
        else:
            mlflow.set_tag('quality_gate', 'failed')
            print(f'  FAILED — MRR@3 = {metrics["mrr_at_3"]:.4f} < {threshold}')
            print(f'  Model NOT registered.')

        print('\nDone.')


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ForkWise ingredient substitution model training')

    parser.add_argument('--config', default='config.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--dataset',
                        default='/workspace/data/processed/train.json',
                        help='Path to training data (local or data-proj01/...)')
    parser.add_argument('--run_name', default='run',
                        help='MLflow run name (e.g. gismo-emb128-gpu)')
    parser.add_argument('--mlflow_tracking_uri',
                        default='http://localhost:5000',
                        help='MLflow tracking server URI')

    # CLI overrides for hyperparameter sweep
    parser.add_argument('--embed_dim', type=int, default=None,
                        help='Override embed_dim from config')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Override hidden_dim from config')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate from config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs from config')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch_size from config')
    parser.add_argument('--margin', type=float, default=None,
                        help='Override margin from config')

    args = parser.parse_args()

    # Load config and apply CLI overrides
    config = yaml.safe_load(open(args.config))
    if args.embed_dim is not None:
        config['embed_dim'] = args.embed_dim
    if args.hidden_dim is not None:
        config['hidden_dim'] = args.hidden_dim
    if args.lr is not None:
        config['lr'] = args.lr
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.margin is not None:
        config['margin'] = args.margin

    train(config, args.dataset, args.run_name, args.mlflow_tracking_uri)
