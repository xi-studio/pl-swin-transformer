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

#fs = gcsfs.GCSFileSystem(project='colorful-aia')
#fs.ls('era5_jy_test')
#with fs.open('era5_jy_test/radar_rain/Z_RADR_I_Z9010_20190701070601_P_DOR_SA_R_10_230_15.010_clean.png', 'rb') as f:
#    img = Image.open(io.BytesIO(f.read()))
#    print(img)

def preprocess_image(x):
    radar = Image.open(x[3:]).convert('L').resize((256, 256))
    radar = np.array(radar)
    radar = mytransform(radar)

    return radar

class Radars(Dataset):
    def __init__(self,transform=None):
        super(Radars, self).__init__()

        self.list = np.load('data/2019_rain_list.npy')
        print(self.list[0])


    def __getitem__(self, index):
        img_t0 = preprocess_image(self.list[index][0])
        img_t1 = preprocess_image(self.list[index][1])

        img_input = np.concatenate((img_t0, img_t1))
        img_input = img_input.reshape((2, 256, 256))

        img = Image.open(self.list[index][2][3:]).convert('L').resize((64, 64))
        img = np.array(img)
        img_output = mytransform(img)
         

        return img_input, img_output

    def __len__(self):
        return len(self.list)


if __name__ == '__main__':
    dataset = Radars()
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=8, pin_memory=True, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, pin_memory=True, shuffle=False, num_workers=4)
    for x in train_loader:
        print(len(x))
        break
