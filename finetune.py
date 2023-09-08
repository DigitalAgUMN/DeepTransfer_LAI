import scipy.io
import sys

sys.path.append("../")
import os
import numpy as np
import torch
from torch.autograd import Function
import torch.autograd as autograd
from torch.nn.functional import normalize
import torch.nn as nn
from torch.utils.data.dataset import Dataset

import glob, os

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

# BUILD MODEL
nd = 33
nb = 7
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")  # config.device
in_channels = 6
model = MapLAI(in_channels=in_channels)
checkpoint = torch.load("Pretraining.pt")  #
model.load_state_dict(checkpoint)
model = model.to(torch.device(device))

mat = scipy.io.loadmat(r"finetune_data.mat")
test_data = mat['finetune']
test_data[:,:-1] = test_data[:,:-1]*10000.0

test_data = test_data.reshape(-1, nd, test_data.shape[1])

img_mean = np.load(r"mean.npy")
img_std = np.load(r"std.npy")
idx = test_data[:,:,-1] == -1
test_data[:,:, -1] = test_data[:,:, -1] * 10.0
for i in range(nb):
    test_data[:, :, i] = (test_data[:, :, i] - img_mean[i]) / img_std[i]
test_data[idx,-1] = -1
test_data_loader = torch.utils.data.DataLoader(dataset=Loader(test_data), batch_size=2, shuffle=False, num_workers=0)

learning_rate = 1e-04
epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(model)
for epoch in range(1, epochs + 1):
    epoch_loss = 0
    for batch, train_data in enumerate(test_data_loader):
        optimizer.zero_grad()

        train_data = train_data.to(device)
        train_LAI = train_data[:, :, -1].flatten()
        train_data = train_data.to(torch.float32)
        train_data = train_data[:, :, :-1]

        pure = model(train_data).flatten()
        batch_loss = myLoss(pure, train_LAI)
        batch_loss.backward()

        optimizer.step()
        epoch_loss += batch_loss

    epoch_loss = epoch_loss / (batch + 1)
    print('Epoch:{}\tTrain Loss:{:.4f}'.format(epoch, epoch_loss))

torch.save(model.state_dict(), "Finetuning.pt")
