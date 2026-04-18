#!/usr/bin/env python3
"""
evaluate.py — Evaluation functions for ForkWise ingredient substitution model.

Computes:
  - MRR@3  (Mean Reciprocal Rank at 3)  — primary quality gate metric
  - NDCG@3 (Normalized Discounted Cumulative Gain at 3)
  - Per-cuisine MRR@3 — safeguarding / fairness requirement

MRR@3 explained:
  For each validation example, check if the correct substitution appears
  in the model's top 3 predictions.
    Rank 1 → score 1.000
    Rank 2 → score 0.500
    Rank 3 → score 0.333
    Not in top 3 → score 0.000
  Average across all examples = MRR@3.

Quality gate threshold: MRR@3 >= 0.30
  Justification: random baseline ~ 0.10 for vocab-size ranking.
  0.30 means the model is 3x better than random.
"""

import torch
import torch.nn.functional as F
import math
from collections import defaultdict


def get_top_k_substitutions(model, context_ids, missing_id, vocab, k=3):
    """Get top-k predicted substitutions for a single query."""
    model.eval()
    with torch.no_grad():
        ctx = torch.tensor([context_ids])
        miss = torch.tensor([missing_id])

        ctx_embed = model.embedding(ctx).mean(dim=1)
        miss_embed = model.embedding(miss)
        query = ctx_embed + miss_embed

        # Score every ingredient in the vocabulary
        all_ids = torch.arange(len(vocab))
        all_embeds = model.embedding(all_ids)
        scores = F.cosine_similarity(
            query.expand(len(vocab), -1), all_embeds)

        # Exclude special tokens and the query ingredient itself
        scores[0] = -1          # <PAD>
        scores[1] = -1          # <UNK>
        scores[missing_id] = -1

        top_k_indices = scores.topk(k).indices.tolist()
        top_k_scores = scores.topk(k).values.tolist()

    return top_k_indices, top_k_scores


def evaluate_model(model, val_data, vocab, k=3):
    """
    Evaluate model on validation set.

    Returns dict of metrics:
      - mrr_at_3:       float  — Mean Reciprocal Rank @ 3
      - ndcg_at_3:      float  — Normalized DCG @ 3
      - num_eval_examples: int
      - mrr_at_3_cuisine_<name>: float — per-cuisine breakdown (fairness)

    Per-cuisine metrics are the safeguarding requirement from the rubric:
    "MRR@3 logged separately per cuisine type in MLflow so model quality
     across cuisine groups is visible."
    """
    reciprocal_ranks = []
    ndcg_scores = []
    per_cuisine_rr = defaultdict(list)

    for record in val_data:
        original = record['original'].lower().strip()
        replacement = record['replacement'].lower().strip()
        cuisine = record.get('cuisine', 'unknown').lower()

        orig_id = vocab.get(original, 1)
        repl_id = vocab.get(replacement, 1)

        # Build context vector (pad/truncate to 20)
        context = []
        for ing in record.get('ingredients', []):
            if isinstance(ing, str):
                context.append(vocab.get(ing.lower().strip(), 1))
        context = context[:20]
        context += [0] * (20 - len(context))

        # Get model predictions
        top_k_ids, _ = get_top_k_substitutions(
            model, context, orig_id, vocab, k=k)

        # MRR@k
        rr = 0.0
        for rank, pred_id in enumerate(top_k_ids, 1):
            if pred_id == repl_id:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
        per_cuisine_rr[cuisine].append(rr)

        # NDCG@k
        dcg = 0.0
        for rank, pred_id in enumerate(top_k_ids, 1):
            if pred_id == repl_id:
                dcg = 1.0 / math.log2(rank + 1)
                break
        ndcg_scores.append(dcg)

    metrics = {
        'mrr_at_3': sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1),
        'ndcg_at_3': sum(ndcg_scores) / max(len(ndcg_scores), 1),
        'num_eval_examples': len(val_data),
    }

    # Per-cuisine MRR@3 (safeguarding: fairness across cuisine types)
    for cuisine, rrs in per_cuisine_rr.items():
        if len(rrs) >= 10:      # only log cuisines with enough samples
            key = f'mrr_at_3_cuisine_{cuisine[:20]}'
            metrics[key] = sum(rrs) / len(rrs)

    return metrics
