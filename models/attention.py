import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp

class MultiHeadAttention(nn.Module):
    def __init__(self, num_header, dim_hidden):
        super(MultiHeadAttention, self).__init__()
        self.num_header = num_header
        self.dim_hidden = dim_hidden
        self.dim_header = dim_hidden // num_header

        self.query = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.value = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.key = nn.Linear(dim_hidden, dim_hidden, bias=False)

        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(dim_hidden, dim_hidden)

        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.out.weight)


    def forward(self, q, k, v, input_length):
        batch_size, max_len = q.size(0), q.size(1)
        mask = torch.ones(batch_size, max_len).cuda()
        for i in range(batch_size):
            mask[i, input_length[i]:] = 0
        mask = mask.lt(1).unsqueeze(1).expand(-1, max_len, -1)
        mask = mask.unsqueeze(1).expand(-1, self.num_header, -1, -1)

        k = self.key(k).view(batch_size, max_len, self.num_header, self.dim_header)
        q = self.query(q).view(batch_size, max_len, self.num_header, self.dim_header)
        v = self.value(v).view(batch_size, max_len, self.num_header, self.dim_header)

        # shape = batch * header * length * dim
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_header)
        scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        scores = torch.matmul(scores, v)

        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.dim_hidden)
        output = self.out(concat)

        return output

class SelfAttention(nn.Module):
    def __init__(self, num_header, dim_hidden, dim_ff=2048, drop_layer=0.0):
        super(SelfAttention, self).__init__()
        self.drop_layer = drop_layer

        self.multiHeadAttention = MultiHeadAttention(num_header, dim_hidden)

        self.norm = nn.LayerNorm(dim_hidden)

        self.fc1 = nn.Linear(dim_hidden, dim_ff)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(dim_ff, dim_hidden)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, x_lengths):
        drop_layer = (torch.rand(1)[0].item() < self.drop_layer)
        if drop_layer and self.training:
            return self.norm(x)

        att = self.multiHeadAttention(x, x, x, x_lengths)
        self.dropout(att)
        if self.training: att = att / (1 - self.drop_layer)
        x = x + att
        x = self.norm(x)

        ff = self.fc2(F.relu(self.fc1(x)))
        if self.training: ff = ff / (1 - self.drop_layer)
        x = x + ff
        x = self.norm(x)

        return x

class PositionEncoding(nn.Module):
    def __init__(self, pos_dim=40, max_len=2000):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, pos_dim, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * -(math.log(10000.0) / pos_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, length = x.size(0), x.size(1)
        pos_enc = self.pe[:, :length]
        pos_enc = pos_enc.repeat(batch_size, 1, 1)
        x_pos_enc = torch.cat((x, pos_enc), 2)
        return x_pos_enc

class SAN(nn.Module):
    def __init__(self, num_char, input_dim=120, downsample=3, pos_dim=40,
                 num_layer=10, num_header=8, hidden_dim=512, drop_layer=0.5):
        super(SAN, self).__init__()
        self.downsample = downsample

        self.embedding = nn.Linear(input_dim * self.downsample, hidden_dim - pos_dim)
        self.pos_enc = PositionEncoding(pos_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.atts = nn.ModuleList([
            SelfAttention(num_header,
                          hidden_dim,
                          drop_layer=(l + 1.0) / num_layer * drop_layer)
            for l in range(num_layer)])

        self.fc = nn.Linear(hidden_dim, num_char)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, x_lengths):
        n, t, d = x.size(0), x.size(1), x.size(2)
        pad_length = (self.downsample - t % self.downsample) % self.downsample
        padding = torch.FloatTensor(n, pad_length, d).zero_().cuda()
        x = torch.cat((x, padding), 1)
        x = x.reshape(n, x.size(1) // self.downsample, x.size(2) * self.downsample)

        x = self.embedding(x)
        x = self.pos_enc(x)
        for att_layer in self.atts:
            x = x + att_layer(x, x_lengths)

        x = x.transpose(0, 1) # x.shape = T * N * D

        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.fc(x)
        x = x.view(t, n, -1)

        x = F.log_softmax(x, dim=-1)
        return x, x_lengths / self.downsample
