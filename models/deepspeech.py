import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pdb import set_trace as bp


class DeepSpeech(nn.Module):
    def __init__(self, rnn_hidden, num_char):
        super(DeepSpeech, self).__init__()

        self.cnns = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2),padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),

            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1),padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

        self.rnns = nn.Sequential(
            BatchRNN(640, rnn_hidden, batch_norm=False),
            BatchRNN(rnn_hidden, rnn_hidden),
            BatchRNN(rnn_hidden, rnn_hidden),
            BatchRNN(rnn_hidden, rnn_hidden),
            BatchRNN(rnn_hidden, rnn_hidden)
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden),
            nn.Linear(rnn_hidden, num_char, bias=False)
        )

    def forward(self, x, x_lengths):
        x = x.unsqueeze(1).transpose(2,3)
        x = self.cnns(x)

        for m in self.cnns.modules():
            if type(m) == nn.modules.conv.Conv2d:
                x_lengths = ((x_lengths + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        x_lengths = x_lengths.int()

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # x.shape = N, C*D, T
        x = x.transpose(1, 2).transpose(0, 1).contiguous() # x.shape = T, N, H

        for rnn in self.rnns:
            x = rnn(x, x_lengths)

        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.fc(x)
        x = x.view(t, n, -1)

        x = F.log_softmax(x, dim=-1)

        return x, x_lengths

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.lstm = nn.GRU(input_size, hidden_size, bidirectional=True)

    def forward(self, x, x_lengths):
        if self.norm is not None:
            t, n = x.size(0), x.size(1)
            x = x.view(t * n, -1)
            x = self.norm(x)
            x = x.view(t, n, -1)
        x = pack_padded_sequence(x, x_lengths)
        x, h = self.lstm(x)
        x, _ = pad_packed_sequence(x)

        t, n = x.size(0), x.size(1)
        x = x.view(t, n, 2, -1).sum(2).view(t, n, -1)

        return x

#class DeepSpeechTransformer(nn.Module):
#    def __init__(self, num_char, num_header=8, num_layer=16, drop_layer=0.0):
#        super(DeepSpeechTransformer, self).__init__()
#
#        self.cnns = nn.Sequential(
#            nn.Conv2d(1, 16, kernel_size=(41, 11), stride=(2, 2),padding=(20, 5)),
#            nn.BatchNorm2d(16),
#            nn.Hardtanh(0, 20, inplace=True),
#
#            nn.Conv2d(16, 16, kernel_size=(21, 11), stride=(2, 1),padding=(10, 5)),
#            nn.BatchNorm2d(16),
#            nn.Hardtanh(0, 20, inplace=True)
#        )
#
#        self.embedding = nn.Linear(160, 472)
##         self.pos_enc = PositionEncoding()
#        hidden_dim = 512
#        self.att_layers = nn.ModuleList([
#            SelfAttention(num_header, hidden_dim,
#                          drop_layer=(l + 1.0) / num_layer * drop_layer)
#            for l in range(num_layer)
#        ])
#
#        self.fc = nn.Sequential(
#            nn.LayerNorm(hidden_dim),
#            nn.Linear(hidden_dim, num_char)
#        )
#
#    def forward(self, x, x_lengths):
#        x = x.unsqueeze(1).transpose(2,3)
#        x = self.cnns(x)
#
#        for m in self.cnns.modules():
#            if type(m) == nn.modules.conv.Conv2d:
#                x_lengths = ((x_lengths + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
#        x_lengths = x_lengths.int()
#
#        sizes = x.size()
#        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]).transpose(1, 2)
#        # x.shape = N * T * D
#        x = self.embedding(x)
#        x = self.pos_enc(x)
#        for att_layer in self.att_layers:
#            x = att_layer(x, x_lengths)
#        x = x.transpose(0, 1)
#        # x.shape = T * N * D
#
#        t, n = x.size(0), x.size(1)
#        x = x.contiguous().view(t * n, -1)
#        x = self.fc(x)
#        x = x.view(t, n, -1)
#
#        x = F.log_softmax(x, dim=-1)
#        return x, x_lengths


