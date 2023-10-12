import time
import pickle
import argparse
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from dgmr import DGMR 
import os
import torch
import numpy as np
from torchvision import transforms, utils


mytransform = transforms.Compose([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        ]
    )


def preprocess_image(x):
    radar = Image.open(x[3:]).convert('L').resize((256, 256))
    radar = np.array(radar)
    radar = mytransform(radar)

    return radar


def load_model():

    # load single dgmr
    dgmr = DGMR.load_from_checkpoint('../data/dgmr.ckpt', strict=False,)
    dgmr.eval()
    dgmr = dgmr.to("cpu")
    print("single-dgmr loaded")

    return dgmr

if __name__ == '__main__':
    model = load_model()
    model.eval()

    path = '../data/2019_rain_list.npy'
    data_list = np.load(path)
    for item in data_list[:1000]:
        img0 = preprocess_image("../" + item[0])
        img1 = preprocess_image("../" + item[1])
        img2 = preprocess_image("../" + item[2])
        x = torch.cat((img0, img1), axis=0)
        x = x.reshape(1, 2, 1, 256, 256)
        print(x.shape)
        y = model(x)
