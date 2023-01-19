import sys
import os
from PIL import Image, ImageDraw
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

def count_numnber_of_each_label():
    """ Returns an integer list of the number of labelled boxes of each class.
    The length of the list is 7 in order to the number of class is 7.
    """

    # annotations path
    dir_path = './dataset/annotations'

    # list to store files
    res = [0,0,0,0,0,0,0]

    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        annotation_file_path = os.path.join(dir_path, path)
        if os.path.isfile(annotation_file_path):
            with open(annotation_file_path, 'r') as f:
                lst = [int(x.split(" ")[0]) for x in f.readlines()]
                for n in lst:
                    res[n] += 1

    print('0(left):{} \n1(right):{} \n2(stop):{} \n3(crosswalk):{} \n4(uturn):{} \n5(traffic_light):{} \n6(ignore):{}'.
          format(res[0],res[1],res[2],res[3],res[4],res[5], res[6]))
    return res

def xywh2xyxy_np(x : np.array):
    y = np.zeros_like(x)
    y[...,0] = (x[...,0] - x[...,2]) / 2  # min_x
    y[...,1] = (x[...,1] - x[...,3]) / 2  # min_y
    y[...,2] = (x[...,0] + x[...,2]) / 2  # max_y
    y[...,3] = (x[...,1] + x[...,3]) / 2  # max_y
    return y

def drawBox(img):
    img *= 255
    if img.shape[0]  == 3:
        img_data = np.array(np.transpose(img, (1,2,0)), dtype=np.uint8)
        img_data = Image.fromarray(img_data)

    draw = ImageDraw.Draw(img_data)

    plt.imshow(img_data)
    plt.show()


