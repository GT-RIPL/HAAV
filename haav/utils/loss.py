import torch
from torch.nn import functional as F


def contrastive_loss(x, x_ema, x_mem, tau=0.07):
    # x, x_ema: N x n_views x d
    # x_mem: K x d
    N, n_views, _ = x.shape

    x = x.reshape(n_views*N, -1)
    x_ema = x_ema.reshape(n_views*N, -1)
    sim = x @ x_ema.T  # n_views*N x n_views*N

    block = torch.ones((1, n_views), dtype=torch.bool)
    block = block.expand(n_views, n_views)
    idx_pos = torch.block_diag(*[block for _ in range(N)])
    idx_neg = ~idx_pos

    sim_pos = sim[idx_pos].reshape(n_views*N, n_views)
    sim_pos = sim_pos.reshape(-1, 1)
    sim_neg = sim[idx_neg].reshape(n_views*N, n_views*N-n_views)
    sim_neg = sim_neg[:, None, :].expand(-1, n_views, -1)
    sim_neg = sim_neg.reshape(-1, n_views*N-n_views)

    sim_neg_mem = x@x_mem.T  # n_views*N x K
    sim_neg_mem = sim_neg_mem[:, None, :].expand(-1, n_views, -1)
    sim_neg_mem = sim_neg_mem.reshape(-1, len(x_mem))

    logits = torch.cat([sim_pos, sim_neg, sim_neg_mem], dim=-1) / tau
    tgt = torch.zeros(len(logits), dtype=torch.long, device=x.device)
    loss = F.cross_entropy(logits, tgt)

    return loss
