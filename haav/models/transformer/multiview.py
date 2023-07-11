import random

import torch
from torch import nn

from models.transformer.utils import sinusoid_encoding_table


class ViewGenerator(nn.Module):
    def __init__(self, d_obj, d_vis, d_txt, d_out, txt_keys=None, topk=8, drop_rate=0.1):
        super().__init__()
        self.d_out = d_out

        # for objects O
        self.obj_mlp = nn.Sequential(
            nn.LayerNorm(d_obj), nn.Linear(d_obj, d_out), nn.Dropout(drop_rate)
        )

        # for vis_ctx
        self.vis_mlp = nn.Sequential(
            nn.LayerNorm(d_vis), nn.Linear(d_vis, d_out), nn.Dropout(drop_rate)
        )

        # for txt_ctx
        self.txt_keys = ("whole", "five", "nine") if txt_keys is None else txt_keys
        for k in self.txt_keys:
            mlp = nn.Sequential(
                nn.LayerNorm(d_txt), nn.Linear(d_txt, d_out), nn.Dropout(drop_rate)
            )
            setattr(self, f"txt_mlp_{k}", mlp)

            if k == "whole":
                num_embeddings = 1
            elif k == "five":
                num_embeddings = 5
            elif k == "nine":
                num_embeddings = 9
            pos = nn.Embedding.from_pretrained(
                sinusoid_encoding_table(num_embeddings, d_out), freeze=True
            )
            setattr(self, f"txt_pos_{k}", pos)

            if k == "whole":
                num_embeddings = topk * 8
            elif k == "five":
                num_embeddings = topk * 2
            elif k == "nine":
                num_embeddings = topk
            rank = nn.Embedding.from_pretrained(
                sinusoid_encoding_table(num_embeddings, d_out), freeze=True
            )
            setattr(self, f"txt_rank_{k}", rank)

        self.n_views = 2 + len(self.txt_keys)
    
    def forward(self, obj, vis_ctx, txt_ctx):
        views = []

        # object
        obj_embed = self.obj_mlp(obj)
        obj_mask = (torch.sum(torch.abs(obj), dim=-1) == 0)
        obj_embed[obj_mask] = 0.
        obj_cls = torch.sum(obj_embed, dim=1, keepdim=True)  # N x 1 x d
        obj_norm = torch.sum(~obj_mask, dim=-1, keepdim=True).unsqueeze(-1)  # N x 1 x 1
        obj_cls = obj_cls / (obj_norm.detach())
        obj_embed = torch.cat([obj_cls, obj_embed], dim=1)
        views.append(obj_embed)

        # vis_ctx
        vis = vis_ctx["grid"]
        vis_embed = self.vis_mlp(vis)
        vis_cls = torch.mean(vis_embed, dim=1, keepdim=True)
        vis_embed = torch.cat([vis_cls, vis_embed], dim=1)
        views.append(vis_embed)

        # txt_ctx
        for k in self.txt_keys:
            txt_k = getattr(self, f"txt_mlp_{k}")(txt_ctx[k]["embed"])
            pos_k = getattr(self, f"txt_pos_{k}")(txt_ctx[k]["pos"])
            rank_k = getattr(self, f"txt_rank_{k}")(txt_ctx[k]["rank"])
            embed_k = txt_k + pos_k + rank_k
            txt_cls = torch.mean(embed_k, dim=1, keepdim=True)
            embed_k = torch.cat([txt_cls, embed_k], dim=1)
            views.append(embed_k)
        
        # pad sequence
        max_len = max([v.shape[1] for v in views])
        N, _, d = views[0].shape
        for i, v in enumerate(views):
            diff_len = max_len - v.shape[1]
            if diff_len > 0:
                p = torch.zeros((N, diff_len, d), device=v.device)
                v = torch.cat([v, p], dim=1)
                views[i] = v
        
        return views
