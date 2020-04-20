from torch.utils.data import Dataset, DataLoader
import torch
from collections import OrderedDict
import pandas as pd
import numpy as np
import time
import cv2
import sys
import os

from Dataset import *
from Model import *


#分別將 training set、validation set、testing set 用 readfile 函式讀進來
data_dir = sys.argv[1]
print("Reading data")

train_x, train_y = readfile(os.path.join(data_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))

val_x, val_y = readfile(os.path.join(data_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))

## Basic setting
batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

## Connect all data
train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

## Initialization
model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.SGD(model_best.parameters(), lr=0.01, momentum=0.9) # optimizer 使用 Adam
num_epoch = 700

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        #將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' %       (epoch + 1, num_epoch, time.time()-epoch_start_time,       train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))


torch.save(model_best.state_dict(), './best_sgdm_750.pth')
