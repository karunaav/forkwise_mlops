import torch
import torch.nn.functional as F
import math
from collections import defaultdict

def get_top_k_substitutions(model, context_ids, missing_id, vocab, k=3):
    model.eval()
    with torch.no_grad():
        ctx  = torch.tensor([context_ids])
        miss = torch.tensor([missing_id])
        ctx_embed  = model.embedding(ctx).mean(dim=1)
        miss_embed = model.embedding(miss)
        query      = ctx_embed + miss_embed
        all_ids    = torch.arange(len(vocab))
        all_embeds = model.embedding(all_ids)
        scores = F.cosine_similarity(query.expand(len(vocab), -1), all_embeds)
        scores[0]          = -1
        scores[1]          = -1
        scores[missing_id] = -1
        return scores.topk(k).indices.tolist(), scores.topk(k).values.tolist()

def evaluate_model(model, val_data, vocab, k=3):
    rrs, ndcgs, per_cuisine = [], [], defaultdict(list)
    for rec in val_data:
        orig    = rec['original'].lower().strip()
        repl    = rec['replacement'].lower().strip()
        cuisine = rec.get('cuisine', 'unknown').lower()
        orig_id = vocab.get(orig, 1)
        repl_id = vocab.get(repl, 1)
        ctx = [vocab.get(i.lower().strip(), 1)
               for i in rec.get('ingredients', []) if isinstance(i, str)]
        ctx = ctx[:20]
        ctx += [0] * (20 - len(ctx))
        top_ids, _ = get_top_k_substitutions(model, ctx, orig_id, vocab, k=k)
        rr  = next((1.0/r for r, pid in enumerate(top_ids, 1) if pid == repl_id), 0.0)
        dcg = next((1.0/math.log2(r+1) for r, pid in enumerate(top_ids, 1)
                    if pid == repl_id), 0.0)
        rrs.append(rr)
        ndcgs.append(dcg)
        per_cuisine[cuisine].append(rr)
    metrics = {
        'mrr_at_3':          sum(rrs)   / max(len(rrs), 1),
        'ndcg_at_3':         sum(ndcgs) / max(len(ndcgs), 1),
        'num_eval_examples': len(val_data),
    }
    for c, rs in per_cuisine.items():
        if len(rs) >= 10:
            metrics[f'mrr_at_3_cuisine_{c[:20]}'] = sum(rs) / len(rs)
    return metrics
