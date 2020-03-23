import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_header, dim_hidden):
        super.__init__(MultiHeadAttention, self)
        self.num_header = num_header
        self.dim_hidden = dim_hidden
        self.dim_header = dim_hidden // num_header

        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        k = self.key(k).view(batch_size, -1, self.num_header, self.dim_header)
        q = self.query(q).view(batch_size, -1, self.num_header, self.dim_header)
        v = self.value(v).view(batch_size, -1, self.num_header, self.dim_header)

        # shape = batch * header * length * dim
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)

        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.dim_hidden)
        output = self.out(concat)

        return output

class SelfAttention(nn.Module):
    def __init__(self, num_header, dim_hidden, dim_ff=2048, residual=True):
        super.__init__(SelfAttention, self)
        self.residual = residual

        self.multiHeadAttention = MultiHeadAttention(num_header, dim_hidden)

        self.norm = nn.BatchNorm1d(dim_hidden)

        self.feedforward = nn.Seqential(
            nn.Linear(dim_hidden, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_hidden)
        )

    def forward(self, x):
        att = self.multiHeadAttention(x, x, x)
        x = x + att if self.residual else att

        x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        ff = self.feedforward(x)
        x = x + ff if self.residual else ff

        x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        return x


