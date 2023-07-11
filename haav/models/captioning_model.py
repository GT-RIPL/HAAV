import torch
from torch import distributions
import utils
from models.containers import Module
from models.beam_search import *


class CaptioningModel(Module):
    def __init__(self):
        super(CaptioningModel, self).__init__()

    def init_weights(self):
        raise NotImplementedError

    def step(self, t, prev_output, img, obj, ctx, seq, mode='teacher_forcing', **kwargs):
        raise NotImplementedError

    def forward(self, images, seq, *args):
        device = images.device
        b_s = images.size(0)
        seq_len = seq.size(1)
        state = self.init_state(b_s, device)
        out = None

        outputs = []
        for t in range(seq_len):
            out, state = self.step(t, state, out, images, seq, *args, mode='teacher_forcing')
            outputs.append(out)

        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1)
        return outputs

    def beam_search(self, img, obj, ctx, max_len: int, eos_idx: int, beam_size: int, out_size=1, return_probs=False, **kwargs):
        bs = BeamSearch(self, max_len, eos_idx, beam_size)
        return bs.apply(img, obj, ctx, out_size, return_probs, **kwargs)
