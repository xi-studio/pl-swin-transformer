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
from torchvision.utils import save_image


mytransform = transforms.Compose([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        ]
    )

my_resize = transforms.Resize((64,64), interpolation=transforms.InterpolationMode.BICUBIC, antialias='True')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    #dgmr = dgmr.to(device)
    print("single-dgmr loaded")

    return dgmr

if __name__ == '__main__':
    model = load_model()
    model.eval()

    path = '../data/2019_rain_list.npy'
    data_list = np.load(path)
    base = 0
    for i, item in enumerate(data_list[99:10000]):
        img0 = preprocess_image("../" + item[0])
        img1 = preprocess_image("../" + item[1])
        img2 = preprocess_image("../" + item[2])
        x = torch.cat((img0, img1), axis=0)
        x = x.reshape(1, 2, 1, 256, 256)
        #x = x.to(device)
        y = model(x)
        yp = y[0,0,0]

        #a = my_resize(y[0,0])
        #b = my_resize(img2.unsqueeze(0))
        
        #save_image(yp, 'sample/dsgan_%d.png' % i)
        #img2 = img2.to(device)

        res = torch.mean((yp - img2)**2)
        #res = torch.mean((a - b)**2)
        print(res.detach().numpy())
        res = torch.mean((yp - img1)**2)
        print(res.detach().numpy())
        #res = torch.mean((a - b)**2)
        base = base + res.detach().numpy()
        print(i)
        if i > 105:
            break

    #np.save('result_64.npy', np.array(base / 10000.0))
    #print(base / 10000.0)
