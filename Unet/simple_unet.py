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

class Radars(Dataset):
    def __init__(self, file_names, transform=None):
        super(Radars, self).__init__()

        self.list = file_names 
        self.transform = transform


    def __getitem__(self, index):
        #img_t0 = preprocess_image(self.list[index])
        #img_t1 = preprocess_image(self.list[index + 1])
        img_t0 = np.ones((256, 256), dtype=np.float32)
        img_t1 = np.ones((256, 256), dtype=np.float32)

        if self.transform is not None:
            img_t0 = self.transform(img_t0)
            img_t1 = self.transform(img_t1)

        return img_t0, img_t1

    def __len__(self):
        return len(self.list)

class UNetModel(nn.Module):

    def __init__(self):
        super(UNetModel, self).__init__()

        #self.n_channels = hparams.n_channels
        #self.mul_channels = hparams.mul_channels
        self.n_channels = 1
        self.mul_channels = 64 

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.SiLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )
   
        def up(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                double_conv(in_channels, out_channels)
            )

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up0 = up(512, 512) 
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.out = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up0(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)

        return self.out(x)


def training_function():
    epoch_num = 4
    batch_size = 16
    learning_rate = 0.005

    #train_tfm = Compose([RandomResizedCrop(image_size, scale=(0.5, 1.0)), ToTensor()])
    train_tfm = Compose([ToTensor()])
    filenames = np.arange(1000)
    dataset = Radars(filenames, transform=train_tfm) 
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)


    model = UNetModel()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    for epoch in range(epoch_num):
        model.train()
        for i, (x_train, y_train) in enumerate(train_loader):
            out = model(x_train)
            loss = criterion(out, x_train)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            #if (i + 1) % 100 == 0:
            #    print(f"{accelerator.device} Train... [epoch {epoch + 1}/{epoch_num}, step {i + 1}/{len(train_loader)}]\t[loss {loss.item()}]")
            print(f"{accelerator.device} Train... [epoch {epoch + 1}/{epoch_num}, step {i + 1}/{len(train_loader)}]\t[loss {loss.item()}]")
        

def main(): 
    training_function()

if __name__ == '__main__':
    main()
