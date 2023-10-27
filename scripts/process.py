import numpy as np
import time
import glob
from PIL import Image

def checktime(data):
    for i, x in enumerate(data[:-1]):
        #print(i)
        #print(data[i])
        #print(data[i+1])
        a = time.mktime(time.strptime((data[i]).split('_')[4], "%Y%m%d%H%M%S"))
        b = time.mktime(time.strptime((data[i+1]).split('_')[4], "%Y%m%d%H%M%S"))
        if (b - a) > 600:
            return False

    return True

def checkrain(x):
    radar = np.array(Image.open(x))
    I = np.asarray(radar) / 255.0
    if np.sum(I > 0.1) > 2000:
        return True
    else:
        return False

def main():
    data_list = glob.glob('data/2019/*/*/*/*/*/*.png')
    data_list.sort()

    num = 0
    for i, x in enumerate(data_list[:-30]):
        time_list = data_list[i: i+25]

        flag_time = checktime(time_list[:])
        #print(time_list[0])
        flag_rain = checkrain(time_list[0])
        #print(flag_rain)
        if flag_rain and flag_time:
            num += 1
            print(num)
    print(num)


if __name__ == '__main__':
    main()
