import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp

class GatedCNN(nn.Module):
    def __init__(self, num_char):
        super(GatedCNN, self).__init__()
        modules = []

        modules.append(nn.Conv1d(120, 500, 48, 2, 97))
        modules.append(nn.Dropout(0.2))

        for _ in range(7):
            modules.append(nn.Conv1d(250, 500, 7, 1))
            modules.append(nn.Dropout(0.3))

        modules.append(nn.Conv1d(250, 2000, 32, 1))
        modules.append(nn.Dropout(0.5))
        modules.append(nn.Conv1d(1000, 2000, 1, 1))
        modules.append(nn.Dropout(0.5))
        modules.append(nn.Conv1d(1000, 2000, 1, 1))
        modules.append(nn.Dropout(0.5))

        self.convs = nn.ModuleList(modules)

        self.out = nn.utils.weight_norm(nn.Conv1d(1000, num_char, 1, 1))

    def forward(self, x, x_lengths):
        x = x.transpose(1,2)
        for layer in self.convs:
            x = layer(x)
            if type(layer) == nn.modules.Conv1d:
                x = F.glu(x, dim=1)
                x_lengths = (x_lengths - layer.kernel_size[0] + 2 * layer.padding[0]) // layer.stride[0] + 1
        x = self.out(x)
        x = F.log_softmax(x.permute(2, 0, 1), dim=-1)
        return x, x_lengths
