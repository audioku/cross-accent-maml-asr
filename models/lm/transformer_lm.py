import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.autograd import Variable

import numpy as np
import math

from utils import constant
from models.common_layers import MultiHeadAttention, PositionalEncoding, PositionwiseFeedForward, PositionwiseFeedForwardWithConv, get_subsequent_mask, get_non_pad_mask, get_attn_key_pad_mask, get_attn_pad_mask, pad_list

from utils.metrics import calculate_metrics
from torch.nn.parameter import Parameter

class TransformerLM(nn.Module):
    """
    Transformer LM class
    Implementation of Improving Language Understanding by Generative Pre-Training
    https://blog.openai.com/language-unsupervised/

    args:

    """

    def __init__(self, id2label, num_src_vocab, num_trg_vocab, num_layers, dim_emb, dim_model, dim_inner, num_heads, dim_key, dim_value, dropout, trg_max_length=1000):
        super(TransformerLM, self).__init__()

        self.sos_id = constant.SOS_TOKEN
        self.eos_id = constant.EOS_TOKEN
        self.id2label = id2label
        self.num_src_vocab = num_src_vocab
        self.num_trg_vocab = num_trg_vocab
        self.num_layers = num_layers
        self.dim_emb = dim_emb

        self.trg_embedding = nn.Embedding(num_src_vocab, dim_emb)
        
        self.positional_encoding = PositionalEncoding(
            dim_model, max_length=trg_max_length)

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(dim_model, dim_inner, num_heads,
                         dim_key, dim_value, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.out_linear = nn.Linear(dim_model, num_trg_vocab)
        self.x_logit_scale = 1.0

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def preprocess(self, padded_input):
        """
        Add SOS TOKEN and EOS TOKEN into padded_input
        """
        seq = [y[y != constant.PAD_TOKEN] for y in padded_input]
        eos = seq[0].new([self.eos_id])
        sos = seq[0].new([self.sos_id])
        seq_in = [torch.cat([sos, y], dim=0) for y in seq]
        seq_out = [torch.cat([y, eos], dim=0) for y in seq]
        seq_in_pad = pad_list(seq_in, self.eos_id)
        seq_out_pad = pad_list(seq_out, constant.PAD_TOKEN)
        assert seq_in_pad.size() == seq_out_pad.size()
        return seq_in_pad, seq_out_pad

    def forward(self, padded_input, input_lengths, verbose=False):
        seq_in_pad, seq_out_pad = self.preprocess(padded_input)

        output_length = seq_in_pad.size(1)

        decoder_output = self.dropout(self.trg_embedding(
            seq_in_pad) * self.x_logit_scale + self.positional_encoding(seq_in_pad))

        # decoder_output = decoder_output.sum(dim=2)
        # print(decoder_output.size())

        for layer in self.layers:
            decoder_output  = layer(decoder_output)

        logits = self.out_linear(decoder_output)
        # print(logits.size(), seq_in_pad.size(), seq_out_pad.size())
        return logits, seq_in_pad, seq_out_pad

class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x

class Attention(nn.Module):
    def __init__(self, dim_model, dim_value, n_head, attn_dropout=0, resid_dropout=0, scale=False):
        super(Attention, self).__init__()
        n_state = dim_model  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(dim_value, dim_value)).view(1, 1, dim_value, dim_value))
        
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, dim_model)
        self.c_proj = Conv1D(n_state, 1, dim_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def _attn(self, q, k, v):
        """
        q: 4x8x16x16
        k: 4x8x64x16
        v: 4x8x16x64
        """
        # print("q", q.size(), k.size(), v.size())
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        # print(">>", w.size(), self.b.size())
        b = self.b[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -1e9 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        """

        """
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a

class TransformerBlock(nn.Module):
    """
    Transformer Block (Decoder Layer for LM)
    """
    
    def __init__(self, dim_model, dim_inner, num_heads, dim_key, dim_value, dropout):
        super(TransformerBlock, self).__init__()

        # self.self_attn = MultiHeadAttention(num_heads, dim_model, dim_key, dim_value, dropout=dropout)
        self.self_attn = Attention(dim_model, dim_value, num_heads, scale=True)
        self.layer_norm_input_1 = nn.LayerNorm(dim_model)
        self.mlp = PositionwiseFeedForwardWithConv(dim_model, dim_inner, dropout=dropout)
        self.layer_norm_input_2 = nn.LayerNorm(dim_model)

    def forward(self, padded_input):
        attn = self.self_attn(padded_input)
        
        out = self.layer_norm_input_1(padded_input + attn)
        out_mlp = self.mlp(out)
        out_block = self.layer_norm_input_2(out + out_mlp)
        return out_block