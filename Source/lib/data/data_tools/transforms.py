
from __future__ import division

import torch
import math
import sys
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings
import cv2
from torch.nn import functional as F

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x # W
        image = image.transpose((2, 0, 1)).astype("float32")
        return torch.from_numpy(image)

class Resize(object):
    """size=(h,w)"""
    def __init__(self, size):
        assert isinstance(size, int) or ( len(size) == 2)
        self.size = size
    #def __call__(self, img, obs,top=None,left=None):
    #    hi,wi=img.shape[:2]
    #    th,tw=self.size
    #    cy,cx=th/hi,tw/wi
    #    obs[:,(1,3)]*=cx
    #    obs[:,(2,4)]*=cy
    #    return cv2.resize(img, (self.size[1],self.size[0])),obs,0,0

    def __call__(self, img, obs,top=None,left=None):
        hi,wi=img.shape[:2]
        th,tw=self.size
        ks,kt=wi/hi,tw/th
        cy,cx=th/hi,tw/wi
        if ks>kt:
            obs[:,(1,3)]*=cx
            obs[:,(2,4)]*=cx
            imgN=np.zeros((self.size[0],self.size[1],3))
            imgR=cv2.resize(img, (int(wi*cx),int(hi*cx)))
            imgN[:imgR.shape[0],:imgR.shape[1]]=imgR
            return imgN,obs,0,0
        elif ks<kt:
            obs[:,(1,3)]*=cy
            obs[:,(2,4)]*=cy
            imgN=np.zeros((self.size[0],self.size[1],3))
            imgR=cv2.resize(img, (int(wi*cy),int(hi*cy)))
            imgN[:imgR.shape[0],:imgR.shape[1]]=imgR
            return imgN,obs,0,0
        else:
            cy,cx=th/hi,tw/wi
            obs[:,(1,3)]*=cx
            obs[:,(2,4)]*=cy
            return cv2.resize(img, (self.size[1],self.size[0])),obs,0,0

class RandomCrop(object):
    def __init__(self, output_size):
        """output_size=(h,w)"""
        assert isinstance(output_size, int) or ( len(output_size) == 2)
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    # def __call__(self, image, obs,  top=None,left=None):
    #     h, w = image.shape[:2]
    #     new_h, new_w = self.output_size
    #     if not top and left:
    #         top = np.random.randint(0, h - new_h)
    #         left = np.random.randint(0, w - new_w)
    #     image = image[top: top + new_h, left: left + new_w]
    #     obs[:,(1,2)] -= [left, top]
    #     idx=(obs[:,2]<new_h)*(obs[:,2]>0)*(obs[:,1]<new_w)*(obs[:,1]>0)
    #     return  image, obs[idx]

    def __call__(self, image, obs,  top=None,left=None):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        c=image.shape[-1] if len(image.shape)==3 else 1
        dh=h - new_h
        dw=w - new_w

        if top==None:
            if dh!=0:
                top = np.random.randint(0, abs(dh))
            else:
                top=0
        if left==None: 
            if dw!=0:
                left = np.random.randint(0, abs(dw))
            else:
                left=0

        if dh>0 and dw>0:
            imageZ=np.zeros((new_h,new_w,3)) if c==3 else np.zeros((new_h,new_w))
            imageZ = image[top: top + new_h, left: left + new_w]
            obs[:,(1,2)] -= [left, top]
            idx=(obs[:,2]<new_h-5)*(obs[:,2]>5)*(obs[:,1]<new_w-5)*(obs[:,1]>5)
            image=imageZ

        elif dh<=0 and dw>0:
            imageZ=np.zeros((new_h,new_w,3)) if c==3 else np.zeros((new_h,new_w))
            imageZ[top: top + h,:] = image[:, left: left + new_w]
            obs[:,(1,2)] -= [left, -top]
            idx=(obs[:,2]<new_h-5)*(obs[:,2]>5)*(obs[:,1]<new_w-5)*(obs[:,1]>5)
            image=imageZ
            
        elif dh>0 and dw<=0:
            imageZ=np.zeros((new_h,new_w,3)) if c==3 else np.zeros((new_h,new_w))
            imageZ[:,left: left + w] = image[top: top + new_h, :]
            obs[:,(1,2)] -= [-left, top]
            idx=(obs[:,2]<new_h-5)*(obs[:,2]>5)*(obs[:,1]<new_w-5)*(obs[:,1]>5)
            image=imageZ
           
        elif dh<0 and dw<0 :
            imageZ=np.zeros((new_h,new_w,3)) if c==3 else np.zeros((new_h,new_w))
            imageZ[top: top + h, left: left + w] = image
            obs[:,(1,2)] += [left, top]
            idx=(obs[:,2]<new_h-5)*(obs[:,2]>5)*(obs[:,1]<new_w-5)*(obs[:,1]>5)
            image=imageZ
            
        else:
            #print(5,dh>0 and dw>0,(dh>0) and (dw>0))
            return image, obs, 0,0
        #topN=top-np.random.randint(8)
        #top=topN if topN>=0 and np.random.random()>0.7 else top
        #leftN=left-np.random.randint(8)
        #left=leftN if leftN>=0 and np.random.random()>0.7 else left
        return  image, obs[idx],top,left

#functions
