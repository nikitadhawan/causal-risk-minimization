"""Shared transformer primitives."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, causal=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.causal = causal
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        def split_heads(t):
            return t.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        Q = split_heads(self.W_q(x))
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            scores = scores.masked_fill(causal_mask == 0, -1e10)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e10)
        scores = self.dropout(F.softmax(scores, dim=-1))

        out = torch.matmul(scores, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out)


class FFN(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_hidden, causal=True):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, causal=causal)
        self.sa_ln = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FFN(d_model, d_hidden)
        self.out_ln = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x):
        res = self.sa_ln(x + self.attention(x))
        return self.out_ln(res + self.ffn(res))
