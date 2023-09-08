import torch
from torch.utils.data.dataset import Dataset
import torch.autograd as autograd
import torch.nn as nn

class MapLAI(torch.nn.Module):
    def __init__(self, in_channels):
        super(MapLAI, self).__init__()

        self.hidden_dim = 32
        # LAYERS
        self.lstm = nn.LSTM(in_channels, self.hidden_dim, 2, batch_first=True, bidirectional = True)
        self.hidden2out = nn.Linear(self.hidden_dim*2, 1)
        self.dropout = nn.Dropout(0.25)

        self.ReLU = nn.ReLU()

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2, batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(2, batch_size, self.hidden_dim)))

    def forward(self, x):

        out, (ht, ct) = self.lstm(x)
        out = self.ReLU(out)
        out = self.dropout(out)

        out = self.hidden2out(out)
        return out


class Loader(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :, :]

def myLoss(est, act):
    act = torch.nan_to_num(act, nan=-1.0)
    mask = ((act != -1.0) * (act != 0)).float()
    return torch.sum(((est - act) * mask) ** 2) / (torch.sum(mask) + 1e-8)