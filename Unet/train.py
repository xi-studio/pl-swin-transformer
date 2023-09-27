import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser

from unet_base import UNetModel

def main(hparams):
    model = UNetModel(hparams)
    trainer = pl.Trainer(accelerator=hparams.accelerator, devices=hparams.devices, max_epochs=hparams.max_epochs, enable_checkpointing=True)
    
    trainer.fit(model)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=-1)
    parser.add_argument("--max_epochs", default=-1)
    parser.add_argument('--n_channels', type=int, default=2)

    args = parser.parse_args()

    main(args)
