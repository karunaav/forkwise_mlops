import torch
import torch.nn as nn
import torch.nn.functional as F

class SubstitutionModel(nn.Module):
    def __init__(self, vocab_size=40, embed_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, context_ids, missing_id):
        ctx_embed  = self.embedding(context_ids).mean(dim=1)
        miss_embed = self.embedding(missing_id)
        query      = ctx_embed + miss_embed
        all_embeds = self.embedding.weight
        scores = F.cosine_similarity(
            query.unsqueeze(1), all_embeds.unsqueeze(0), dim=2)
        return scores

    def get_substitutions(self, context_ids, missing_id, k=3):
        scores = self.forward(
            context_ids.unsqueeze(0), missing_id.unsqueeze(0))
        scores[0][0]          = -1
        scores[0][missing_id] = -1
        top_k = scores[0].topk(k)
        return top_k.indices.tolist(), top_k.values.tolist()
