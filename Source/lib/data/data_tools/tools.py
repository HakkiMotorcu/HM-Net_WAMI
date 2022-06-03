import torch
import os
import numpy as np
import pandas as pd
import sys
import cv2

import json
from torch.utils.data import Dataset
from torchvision.io import read_image
from collections import OrderedDict
import math


def read_json(path):
    with open(path,"r")as file:
        ss=file.read()
    data=json.loads(ss)
    return data

def load_coco(path):
    #center x,y,w/2,h/2,class id
    data=read_json(path)
    keeper_temp=OrderedDict()
    for i,ob in enumerate(data["annotations"]):
        idx=ob["image_id"]
        if idx not in keeper_temp:
            keeper_temp[idx]={"path":None,"size":None,"ann":[]}
        cc=ob["category_id"]
        bbox=ob["bbox"]
        dd=[ob["track_id"],bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2,bbox[2]/2,bbox[3]/2,cc]
        keeper_temp[idx]["ann"].append(dd)

    keeper=OrderedDict()
    for dd in data["images"]:
        if dd["id"] in keeper_temp:
            keeper[dd["id"]]={}
            keeper[dd["id"]]["video_id"]=dd["video_id"]
            keeper[dd["id"]]["frame_id"]=dd["frame_id"]
            keeper[dd["id"]]["path"]=dd["file_name"]
            keeper[dd["id"]]["ann"]=keeper_temp[dd["id"]]["ann"]

    
    cats={}
    for aa in data["categories"]:
        ll=list(aa.values())
        if ll[1]!="ignore":
            cats[ll[0]-1]=ll[1]

    return list(keeper.values()),cats
    
def load_coco_test(path):
    #center x,y,w/2,h/2,class id
    data=read_json(path)
    keeper_temp=OrderedDict()
    for i,ob in enumerate(data["annotations"]):
        idx=ob["image_id"]
        if idx not in keeper_temp:
            keeper_temp[idx]={"path":None,"size":None,"ann":[]}
        cc=ob["category_id"]
        bbox=ob["bbox"]
        dd=[ob["track_id"],bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2,bbox[2]/2,bbox[3]/2,cc]
        keeper_temp[idx]["ann"].append(dd)
    
    keeper=OrderedDict()
    for dd in data["images"]:
        if dd["id"] in keeper_temp:
            keeper[dd["id"]]={}
            keeper[dd["id"]]["video_id"]=dd["video_id"]
            keeper[dd["id"]]["frame_id"]=dd["frame_id"]
            keeper[dd["id"]]["path"]=dd["file_name"]
            keeper[dd["id"]]["ann"]=keeper_temp[dd["id"]]["ann"]
        else:
            keeper[dd["id"]]={"path":None,"size":None,"ann":[]}
            keeper[dd["id"]]["path"]=dd["file_name"]
            keeper[dd["id"]]["video_id"]=dd["video_id"]
            keeper[dd["id"]]["frame_id"]=dd["frame_id"]

    




    cats={}
    for aa in data["categories"]:
        ll=list(aa.values())
        if ll[1]!="ignore":
            cats[ll[0]-1]=ll[1]

    return list(keeper.values()),cats

#CenterTrack
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    height*=2
    width*=2
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

#CenterTrack
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    # y, x = np.arange(-m, m + 1).reshape(-1, 1), np.arange(-n, n + 1).reshape(1, -1)
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

#CenterTrack
def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
