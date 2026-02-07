import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [batch_size, embedding_dim] (L2 normalized)
        labels:   [batch_size]
        """

        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # cosine similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # remove self-similarity
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        loss = -mean_log_prob_pos.mean()
        return loss
