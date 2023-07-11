import random
import math

import torch
from torch import nn
from torch.nn import functional as F


def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)
    if padding_idx is not None:
        out[padding_idx] = 0

    # use the magnitude of kaiming init with fan_in
    fan = nn.init._calculate_correct_fan(out, "fan_in")
    gain = nn.init.calculate_gain('leaky_relu', 0)
    std = gain / math.sqrt(fan)
    a = math.sqrt(3.0) * std
    out *= a
    
    return out


def drop_sequence(x, drop_rate=0.1):
    # x: N x S x d
    N, S, _ = x.shape
    mask_keep = torch.rand((N, S-1), device=x.device)  # N x S-1

    # keep at least one token
    padded_tokens = torch.sum(torch.abs(x[:, 1:]), dim=-1) == 0  # N x S-1
    mask_keep[padded_tokens] = -0.1
    idx = torch.argmax(mask_keep, dim=1)  # N
    mask_keep[torch.arange(N), idx] = 1.1

    # generate mask
    mask_keep = mask_keep > drop_rate
    assert torch.all(torch.sum(mask_keep * ~padded_tokens, dim=1) > 0)
    mask_keep = torch.cat([
        torch.ones(N, 1, dtype=torch.bool, device=x.device), mask_keep], dim=1
    )  # N x S

    return x * mask_keep.unsqueeze(-1).float()


def drop_view(x, drop_rate=0.1):
    # x: [[...] x n_views] x Nx
    mask_keep = torch.rand(len(x[0]))
    # keep at least one view
    mask_keep[torch.argmax(mask_keep)] = 1.1
    mask_keep = mask_keep > drop_rate
    x = [[xi[j] for j, k in enumerate(mask_keep) if k] for xi in x]
    
    return x


def shuffle(x):
    # x: [...] x Nx
    for i in range(len(x)-1):
        assert len(x[i]) == len(x[i+1]), f"got len(x[{i}]) = {len(x[i])}, len(x[{i+1}]) = {len(x[i+1])})"
    
    perm = list(range(len(x[0])))
    random.shuffle(perm)
    x = [[xi[j] for j in perm] for xi in x]

    return x, perm


def unshuffle(x, perm):
    for i in range(len(x)-1):
        assert len(x[i]) == len(perm), f"got len(x[{i}]) = {len(x[i])} != {len(perm)})"
    
    res = []
    for i, xi in enumerate(x):
        res_i = [None] * len(xi)
        for j, p in enumerate(perm):
            res_i[p] = xi[j]
        res.append(res_i)
    
    return res

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out