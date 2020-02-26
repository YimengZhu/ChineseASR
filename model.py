import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
    def __init__(self, num_char):
        super(DeepSpeech, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2),padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),

            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1),padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        
        self.rnns = nn.Sequential(
            BatchRNN(1280, 768, batch_norm=False),
            BatchRNN(768, 768),
            BatchRNN(768, 768),
            BatchRNN(768, 768),
            BatchRNN(768, 768),
            BatchRNN(768, 768)
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Linear(768, num_char, bias=False)
        )

    def forward(self, x, x_lengths):
        x = self.cnns(x)
        x = self.rnns(x, x_lengths)
        x = self.fc(x)
        if self.training:
            x = F.softmax(x, dim=-1)
        return x, x_lengths

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)

    def forward(self, x, x_lengths):
        if self.norm is not None:
            t, n = t.size(0), t.size(1)
            x = x.view(t * n, -1)
            x = self.norm(x)
            x = x.view(t, n, -1)

        x = pack_padded_sequence(x, x_lengths)
        x = self.lstm(x)
        x = pad_packed_sequence(x)
        
        t, n = x.size(0), x.size(1)
        x = x.view(t, n, 2, -1).sum(2).view(t, n, -1)

        return x
