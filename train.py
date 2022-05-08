import os
from tkinter import Variable
import numpy as np
from sklearn.cluster import KMeans
import segmentation_models as sm
from data.dataloader import CityScapesDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from net import UNet
from utils import LoadImage, CreateKMeans

# Initializaing UNet architecture and clustering
net = UNet(3, 15).float() # Using float precision
km = CreateKMeans(num_clusters=15)

criterion = nn.CrossEntropyLoss()
sm_loss = sm.losses.CategoricalFocalLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


train_data = CityScapesDataset(km = km,
                img_dir='/Users/saadhossain/VIP-Project/side/data/data_og/train')
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

# for i, batch in enumerate(train_loader):
#     print(i)


for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (img, seg) in enumerate(train_loader):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        img = img.permute(0, 3, 1, 2).float()
        seg = seg.permute(0, 3, 1, 2).float()
        outputs = net(img)
        loss = torch.Tensor(np.array(sm_loss(outputs.detach().numpy(), seg.detach().numpy()))).float()
        loss.requires_grad = True
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2:.3f}')
            running_loss = 0.0

print('Finished Training')