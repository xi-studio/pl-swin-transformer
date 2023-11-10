import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import lightning.pytorch as pl

from era5_dataset import TP

from config import get_config
from models import build_model


class SwModel(pl.LightningModule):
    def __init__(self, config):
        super(SwModel, self).__init__()
        self.batch_size = config.DATA.BATCH_SIZE
        self.num_workers = config.DATA.NUM_WORKERS

        self.l1 = build_model(config)

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), axis=1)
        x = self.l1(x)
        B, C= x.shape
        x = x.view(B, 1, 64, 64)
        
        return x

    def training_step(self, batch, batch_nb):
        x1, x2, x3, y = batch
        y_hat = self.forward(x1, x2, x3)
        loss = F.l1_loss(y_hat, y) 
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x1, x2, x3, y = batch
        y_hat = self.forward(x1, x2, x3)
        loss = F.l1_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)

    def __dataloader(self):
        dataset = TP() 
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, pin_memory=True, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers=self.num_workers)

        return {
            'train': train_loader,
            'val': val_loader,
        }

    def train_dataloader(self):
        return self.__dataloader()['train']

    def val_dataloader(self):
        return self.__dataloader()['val']
