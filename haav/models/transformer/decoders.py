import torch
from torch import nn
from torch.nn import functional as F

from models.transformer.attention import MultiHeadAttention, AggregationAttention
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward, drop_view
from models.containers import Module, ModuleList


class MeshedDecoderLayer(Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, n_layers=3, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderLayer, self).__init__()
        d_k = d_v = d_model//h
        self.n_layers = n_layers

        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.cross_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.agg_layers = AggregationAttention(d_model, d_model//h, h)
        self.agg_views = AggregationAttention(d_model, d_model//h, h)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad

        n_views, (N, L, S, d) = len(enc_output), enc_output[0].shape
        enc_output = torch.stack(enc_output, dim=1)  # N x n_views x L x S x d
        enc_output = enc_output.reshape(L*n_views*N, S, d)
        mask_enc_att = torch.stack(mask_enc_att, dim=1)  # N x n_views x 1 x 1 x S
        mask_enc_att = mask_enc_att.unsqueeze(2).expand(N, n_views, L, 1, 1, S)
        mask_enc_att = mask_enc_att.reshape(L*n_views*N, 1, 1, S)
        
        self_att_exp = self_att.unsqueeze(1).expand(N, L*n_views, *self_att.shape[1:])
        self_att_exp = self_att_exp.reshape(L*n_views*N, *self_att.shape[1:])
        mask_pad_exp = mask_pad.unsqueeze(1).expand(N, L*n_views, *mask_pad.shape[1:])
        mask_pad_exp = mask_pad_exp.reshape(L*n_views*N, *mask_pad.shape[1:])
        enc_att = self.cross_att(
            self_att_exp, enc_output, enc_output, mask_enc_att
        ) * mask_pad_exp  # L*n_views*N x Sq x d

        enc_att = enc_att.reshape(n_views*N, L, *self_att.shape[1:])
        enc_att = torch.transpose(enc_att, 1, 2)  # n_views*N x Sq x L x d
        self_att_exp = self_att.unsqueeze(1).expand(N, n_views, *self_att.shape[1:])
        self_att_exp = self_att_exp.reshape(n_views*N, *self_att.shape[1:])
        mask_pad_exp = mask_pad.unsqueeze(1).expand(N, n_views, *mask_pad.shape[1:])
        mask_pad_exp = mask_pad_exp.reshape(n_views*N, *mask_pad.shape[1:])
        enc_att = self.agg_layers(self_att_exp, enc_att) * mask_pad_exp  # n_views*N x Sq x d

        enc_att = enc_att.reshape(N, n_views, *self_att.shape[1:])
        enc_att = torch.transpose(enc_att, 1, 2)  # N x Sq x n_views x d
        enc_att = self.agg_views(self_att, enc_att) * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class MeshedDecoder(Module):
    def __init__(
        self, vocab_size, max_len, N_dec, padding_idx, d_model=512, h=8, d_ff=2048, n_layers=3, dropout_v=0.1,
        self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None
    ):
        super(MeshedDecoder, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [MeshedDecoderLayer(d_model, h, d_ff, n_layers, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec
        self.n_layers = n_layers
        self.dropout_v = dropout_v

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, encoder_mask):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        if self.training:
            [encoder_output, encoder_mask] = drop_view([encoder_output, encoder_mask], self.dropout_v)

        f = self.word_emb(input) + self.pos_emb(seq)
        for l in self.layers:
            f = l(f, encoder_output, mask_queries, mask_self_attention, encoder_mask)
        log_p = F.log_softmax(self.fc(f), dim=-1)
        
        return log_p
