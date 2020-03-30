import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformer import SelfAttention
from pdb import set_trace as bp


class Transducer(nn.Module):
    def __init__(self, transcript, predict, hidden_size, label_size):
        super(Transducer, self).__init__()
        self.transcriptor = transcriptor
        self.predictor = preditor

        joint_input_dim = transcriptor.modules[-1].out_features + predictor.modules[-1].out_features

        self.joint = nn.Sequential(
                        nn.Linear(joint_input_dim, hidden_size),
                        nn.Tanh(),
                        nn.Linear(hidden_size, label_size)
                    )

    def forward(self, feature, feat_len, label, label_len):
        transcript = self.transcriptor(feature, feat_len)

        pad_traget = F.pad(label, pad=(1,0,0,0), value=0)
        predict = self.predictor(pad_target, label_len.add(1))

        if transcript.dim() == 3 and predict.dim() == 3:
            transcript = transcript.unsqueeze(1)
            predict = predict.unsqueeze(2)

            t, u = transcript.size(1), predict.size(2)

            trancript = transcript.repeat([1, 1, u, 1])
            predict = predict.repeat([1, t, 1 ,1])
        else:
            assert predict.dim() == transcript.dim()

        concat_state = torch.concat((predict, transcript), dim=-1)
        output = self.joint(concat_state)
        return output


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
            BatchRNN(1312, rnn_hiden, batch_norm=False),
            BatchRNN(rnn_hiden, rnn_hiden),
            BatchRNN(rnn_hiden, rnn_hiden),
            BatchRNN(rnn_hiden, rnn_hiden),
            BatchRNN(rnn_hiden, rnn_hiden),
            BatchRNN(rnn_hiden, rnn_hiden)
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(rnn_hiden),
            nn.Linear(rnn_hiden, num_char, bias=False)
        )

    def forward(self, x, x_lengths):
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
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)

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

class DeepSpeechTransformer(nn.Module):
    def __init__(self, num_char):
        super(DeepSpeechTransformer, self).__init__()

        self.cnns = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2),padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),

            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1),padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

        self.transformers = nn.Sequential(
            SelfAttention(8, 1312),
            # SelfAttention(8, 1312),
            SelfAttention(8, 1312),
            SelfAttention(8, 1312),
            SelfAttention(8, 1312)
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(1312),
            nn.Linear(1312, num_char, bias=False)
        )

    def forward(self, x, x_lengths):
        x = self.cnns(x)

        for m in self.cnns.modules():
            if type(m) == nn.modules.conv.Conv2d:
                x_lengths = ((x_lengths + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        x_lengths = x_lengths.int()

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]).transpose(1, 2)

        # x.shape = N * T * D
        x = self.transformers(x)
        x = x.transpose(0, 1)
        # x.shape = T * N * D

        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.fc(x)
        x = x.view(t, n, -1)

        x = F.log_softmax(x, dim=-1)
        return x, x_lengths
