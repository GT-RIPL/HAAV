import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import PositionWiseFeedForward, drop_sequence


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model=512, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
        attention_module=None, attention_module_kwargs=None
    ):
        super(EncoderLayer, self).__init__()
        d_k = d_v = d_model//h
        self.mhatt = MultiHeadAttention(
            d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
            attention_module=attention_module, attention_module_kwargs=attention_module_kwargs
        )
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(
        self, N, padding_idx, d_model=512, h=8, d_ff=2048, dropout_s=0.1,
        identity_map_reordering=False, attention_module=None, attention_module_kwargs=None
    ):
        super(MultiLevelEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.dropout_s = dropout_s
    
    def forward(self, views):
        # views: [N x S x d] x n_views
        n_views,(N, S, d) = len(views), views[0].shape
        if self.training:
            views = [drop_sequence(v, self.dropout_s) for v in views]
        views = torch.stack(views).reshape(N*n_views, S, d)
        
        mask = (torch.sum(torch.abs(views), -1) == 0)
        mask = mask.unsqueeze(1).unsqueeze(1)  # (N*n_views, 1, 1, S)

        outs = []
        out = views
        for l in self.layers:
            out = l(out, out, out, mask)
            outs.append(out)
        outs = torch.stack(outs, dim=1)  # N*n_views x l x S x d
        
        outs = outs.reshape(n_views, N, len(self.layers), S, d)
        masks = mask.reshape(n_views, N, 1, 1, S)
        outs = [o for o in outs]
        masks = [m for m in masks]
        
        return outs, masks


class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, **kwargs):
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, **kwargs)

    def forward(self, input):
        return super(MemoryAugmentedEncoder, self).forward(input)
