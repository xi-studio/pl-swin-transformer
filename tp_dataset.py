from PIL import Image
import torch
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, utils
import glob
import csv
import time
from argparse import ArgumentParser
import gcsfs
import io


mytransform = transforms.Compose([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        ]
    )

def preprocess_image(x):
    tp = Image.open(x).convert('L').resize((256, 256))
    tp = np.array(tp)
    tp = mytransform(tp)

    return tp

class TP(Dataset):
    def __init__(self,transform=None):
        super(TP, self).__init__()


    def __getitem__(self, index):
        tp_path = 'data/era5/tp_data/rain_%05d.png' % index
        tp_t1   = 'data/era5/tp_data/rain_%05d.png' % (index + 1)
        tp_t2   = 'data/era5/tp_data/rain_%05d.png' % (index + 2)
        u_path  = 'data/era5/wind_data/wind_u_%05d.png' % (index + 1)
        v_path  = 'data/era5/wind_data/wind_v_%05d.png' % (index + 1)

        tp_img = preprocess_image(tp_path)
        u_img  = preprocess_image(u_path)
        v_img  = preprocess_image(v_path)
        t1_img = preprocess_image(tp_t1)
        t2_img = preprocess_image(tp_t2)

        return tp_img, t1_img, u_img, v_img, t2_img

    def __len__(self):
        return 6000


if __name__ == '__main__':
    dataset = TP()
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=8, pin_memory=True, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, pin_memory=True, shuffle=False, num_workers=4)
    for x in train_loader:
        a, b, c, d = x
        print(a.shape)
        print(len(x))
        break
