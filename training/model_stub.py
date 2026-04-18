"""
model_stub.py — Shared SubstitutionModel class.

This file is used by BOTH training and serving so the checkpoint
format is always compatible. If you change the architecture here,
update it in serving/fastapi_pt/model_stub.py too (or symlink).

Architecture:
  1. Each ingredient in the vocabulary gets an embedding vector
  2. Recipe context = mean-pool of ingredient embeddings
  3. Query = context_embedding + missing_ingredient_embedding
  4. Score each candidate by cosine similarity with query
  5. Return top-k candidates ranked by score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubstitutionModel(nn.Module):
    """Embedding-based ingredient substitution ranking model."""

    def __init__(self, vocab_size=40, embed_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, context_ids, missing_id):
        """
        Args:
            context_ids: (batch, context_len) — ingredient IDs in the recipe
            missing_id:  (batch,)             — ingredient to substitute

        Returns:
            (batch, vocab_size) cosine similarity scores for every candidate
        """
        ctx_embed = self.embedding(context_ids).mean(dim=1)
        miss_embed = self.embedding(missing_id)
        query = ctx_embed + miss_embed

        all_embeds = self.embedding.weight          # (vocab_size, embed_dim)
        scores = F.cosine_similarity(
            query.unsqueeze(1), all_embeds.unsqueeze(0), dim=2)
        return scores

    def get_substitutions(self, context_ids, missing_id, k=3):
        """Get top-k substitutions with scores (inference helper)."""
        scores = self.forward(
            context_ids.unsqueeze(0), missing_id.unsqueeze(0))
        scores[0][0] = -1                           # exclude <PAD>
        scores[0][missing_id] = -1                  # exclude the missing ingredient
        top_k = scores[0].topk(k)
        return top_k.indices.tolist(), top_k.values.tolist()
