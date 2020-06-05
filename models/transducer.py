import torch
import torch.nn as nn

class Transducer(nn.Module):
    def __init__(self, transcript, predict, hidden_size, label_size):
        super(Transducer, self).__init__()
        self.transcriptor = transcriptor
        self.predictor = preditor

        joint_input_dim = transcriptor.modules[-1].out_features + predictor.modules[-1].out_features

        self.joint = nn.Sequential(nn.Linear(joint_input_dim, hidden_size),
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

