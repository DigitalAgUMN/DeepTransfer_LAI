import scipy.io
import sys

sys.path.append("../")
import os
import numpy as np
import torch

import glob, os

from utils import MapLAI, Loader, myLoss

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
