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
pred_path = sys.argv[2]
model_path = sys.argv[3]

print("Reading data")
test_x = readfile(os.path.join(data_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

## Load model ....
model = Classifier().cuda()
model.load_state_dict(torch.load(model_path))
model.eval()


model.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)



#將結果寫入 csv 檔
with open(pred_path, 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))