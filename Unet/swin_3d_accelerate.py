import torch
import argparse
import os
import re

import numpy as np
import PIL
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor, Compose
import torchvision.datasets as datasets
from accelerate import Accelerator
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor
from swin_transformer_3d import SwinTransformer3D 

class Radars(Dataset):
    def __init__(self, filenames, transform=None):
        super(Radars, self).__init__()

        self.list = filenames 
        self.transform = transform

    def __getitem__(self, index):
        satelite = np.ones((5*8, 256, 256), dtype=np.float32)

        return satelite, satelite

    def __len__(self):
        return len(self.list)


def training_function(config):
    epoch_num     = config['num_epochs']
    batch_size    = config['batch_size']
    learning_rate = config['lr']
    #filenames     = np.load(config['filenames'])
    filenames     = np.arange(1000)
    
    train_tfm = Compose([ToTensor()])
    dataset = Radars(filenames, transform=train_tfm) 
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)


    #model = SwinTransformer3D(config)
    model = SwinTransformer3D(in_chans=5, 
                              patch_size=(2,4,4), 
                              embed_dim=48, 
                              window_size=(2,7,7), 
                              depths=[2, 2, 2]
                              )
    L1_loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    for epoch in range(epoch_num):
        model.train()
        for i, (x_train, y_train) in enumerate(train_loader):
            out = model(x_train)
            loss = L1_loss(out, x_train)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            if (i + 1) % 1 == 0:
                print(f"{accelerator.device} Train... [epoch {epoch + 1}/{epoch_num}, step {i + 1}/{len(train_loader)}]\t[loss {loss.item()}]")
       

def main(): 
    config = {"lr": 4e-5, "num_epochs": 3, "seed": 42, "batch_size": 4, "in_channels": 10, "mul_channels":64}
    config['filenames'] = 'data/meta/obs_meta.npy'
    training_function(config)

if __name__ == '__main__':
    main()
