from PIL import Image
import torch
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import csv
import time
from argparse import ArgumentParser

mytransform = transforms.Compose([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        ]
    )

def preprocess_image(x):
    radar = Image.open(x).convert('L').resize((256, 256))
    radar = np.array(radar)
    radar = mytransform(radar)

    return radar

class Radars(Dataset):
    def __init__(self,transform=None):
        super(Radars, self).__init__()

        self.list = np.load('../data/2019_rain_list.npy')


    def __getitem__(self, index):
        img_t0 = preprocess_image(self.list[index][0])
        img_t1 = preprocess_image(self.list[index][1])
        img_t2 = preprocess_image(self.list[index][2])

        img_input = np.concatenate((img_t0, img_t1))
        img_input = img_input.reshape((2, 256, 256))
        img_output = img_t2

        return img_input, img_output

    def __len__(self):
        return len(self.list)


