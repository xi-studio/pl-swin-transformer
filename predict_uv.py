import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
import argparse
import glob
from PIL import Image
import numpy as np

from config import get_config
from models import build_model
from era5model import SwModel


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--accelerator', type=str, help='device')
    parser.add_argument('--devices', type=int, help='devices')
    parser.add_argument('--max_epochs', type=int, help='max epochs')


    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

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
        tp_path = 'data/era5/tp_data/rain_%05d.png' % (index + 6000)
        tp_t1   = 'data/era5/tp_data/rain_%05d.png' % (index + 1 + 6000)
        u_path  = 'data/era5/wind_data/wind_u_%05d.png' % (index + 6000)
        v_path  = 'data/era5/wind_data/wind_v_%05d.png' % (index + 6000)

        tp_img = preprocess_image(tp_path)
        u_img  = preprocess_image(u_path)
        v_img  = preprocess_image(v_path)
        t1_img = preprocess_image(tp_t1)

        return tp_img, u_img, v_img, t1_img 

    def __len__(self):
        return 2000

def predict_24(index):
    
    tp_path = 'data/era5/tp_data/rain_%05d.png' % (index + 6000)
    tp_t1   = 'data/era5/tp_data/rain_%05d.png' % (index + 1 + 6000)
    u_path  = 'data/era5/wind_data/wind_u_%05d.png' % (index + 6000)
    v_path  = 'data/era5/wind_data/wind_v_%05d.png' % (index + 6000)

    tp_img = preprocess_image(tp_path)
    u_img  = preprocess_image(u_path)
    v_img  = preprocess_image(v_path)
    t1_img = preprocess_image(tp_t1)

    tp_img = tp_img.unsqueeze(0)
    u_img  = u_img.unsqueeze(0)
    v_img  = v_img.unsqueeze(0)
    t1_img = t1_img.unsqueeze(0)

    return tp_img, u_img, v_img, t1_img


def main(args, config):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = SwModel.load_from_checkpoint(checkpoint_path='lightning_logs/tp_uv_1127/checkpoints/latest.ckpt', config=config)
    model.eval()
    
    model = model.to(device)

    dataset = TP()
    train_loader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=4)

    base = 0 
    for i, (x1, x2, x3, y) in enumerate(train_loader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        y = y.to(device)
        yp = model(x1, x2, x3)
        loss = torch.mean((yp - y)**2)
        base = base + loss.detach().numpy()
        print(i)
        #cy = torch.cat((y, yp), axis=0)
        #save_image(cy, 'data/predict_sample/tp_vs/%d.png' % i)
        if i > 1990:
            break
    print(base / 1990.0)
   
#    index = 0
#    loss_list = []
#    img_list = []
#    for i in range(240):
#        x1, x2, x3, y = predict_24(i) 
#        x1 = x1.to(device)
#        x2 = x2.to(device)
#        x3 = x3.to(device)
#        y  = y.to(device)
#        yp = model(x1, x2, x3)
#
#        loss = torch.mean((yp - y)**2)
#        loss_list.append(loss.detach().numpy())
#        print(i, ":", loss)
#        #cy = torch.cat((y, yp),axis=0)
#        #save_image(yp, 'data/predict_sample/tp_4_xr/%d_%03d.png' % (index+6000, i))
#
#
#    #print(np.array(loss_list))
#    np.save('data/predict_sample/uv_1127_6000_240_loss.npy', np.array(loss_list))
#

if __name__ == "__main__":
    args, config = parse_option()
    main(args, config)
