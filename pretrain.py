import sys
sys.path.append("../")

import torch
import glob, os
import scipy.io
import numpy as np

from utils import MapLAI, Loader, myLoss

torch.set_default_tensor_type('torch.cuda.FloatTensor')

data = scipy.io.loadmat(r"./MODIS_train.mat")
data = data['train_data']

nb = 7  # number of bands
img_mean = np.zeros([nb, 1])
img_std = np.zeros([nb, 1])
#normalize
for i in range(nb):
    img_mean[i] = np.nanmean(data[:, :, i])
    img_std[i] = np.nanstd(data[:, :, i])
    data[:, :, i] = (data[:, :, i] - img_mean[i])/img_std[i]

np.save(r"mean",img_mean)
np.save(r"std",img_std)

learning_rate = 1e-04
epochs = 100
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") #config.device

batch_size = 500
in_channels = 6

model = MapLAI(in_channels=in_channels)
model = model.to(torch.device(device))

print(model)

gamma = 0.2
step_size = 20
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # , weight_decay=weight_decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=gamma)

train_precent = np.int32(0.8 * data.shape[0])
indices = np.random.permutation(data.shape[0])
train_data = data[indices[:train_precent], :, :]
val_data = data[indices[train_precent:], :, :]

train_data = Loader(train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0,
                                          generator=torch.Generator(device=device))
val_data_loader = Loader(val_data)
val_data_loader = torch.utils.data.DataLoader(dataset=val_data_loader, batch_size=batch_size, shuffle=False, num_workers=0,
                                         generator=torch.Generator(device=device))

print("TRAIN MODEL")
train_loss = []

for epoch in range(1, epochs + 1):
    epoch_loss = 0
    val_epoch_LAI_loss = 0

    model.train()
    for batch, train_data in enumerate(data_loader):

        optimizer.zero_grad()
        train_LAI = train_data[:, :, -1].flatten()
        train_data = train_data.to(torch.float32)

        train_data = train_data[:,:,:-1]

        pure = model(train_data).flatten()
        batch_loss = myLoss(pure, train_LAI)
        batch_loss.backward()

        optimizer.step()
        epoch_loss += batch_loss

    epoch_loss = epoch_loss / (batch + 1)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        for batch, val_data in enumerate(val_data_loader):

            LAI_true = val_data[:, :, -1].flatten()
            val_data = val_data.to(torch.float32)
            val_data = val_data[:, :, :-1]

            LAI_pre = model(val_data).flatten()
            val_epoch_LAI_loss += myLoss(LAI_pre.flatten(), LAI_true)

    val_epoch_LAI_loss = val_epoch_LAI_loss / (batch + 1)
    print('Epoch:{}\tTrain Loss:{:.4f}\tVal Loss:{:.4f}'.format(epoch, epoch_loss,val_epoch_LAI_loss))

torch.save(model.state_dict(), "Pretraining.pt")
