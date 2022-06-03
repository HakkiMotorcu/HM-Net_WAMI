import torch
import os
import numpy as np
import sys
import cv2
import random
import math
from torch.utils.data import Dataset
from ..setup import Setup
from .data_tools.tools import gaussian_radius,gaussian2D,draw_umich_gaussian,load_coco,load_coco_test
from .data_tools.transforms import Resize,RandomCrop,ToTensor
import matplotlib.pyplot as plt



class BaseDataset(Dataset):
    def __init__(self, ann_path=None, root_path=None, setup=None,sampler=None,call_purp=None):
        self.min_rad=3
        self.setup=setup if setup!=None else Setup().parse([])


        self.ann_path = ann_path if ann_path else self.setup.ann_path
        self.root_path=root_path if root_path else self.setup.seq_path


        self.setup.mean = np.array(self.setup.mean, dtype=np.float32).reshape(1, 1, 3)
        self.setup.std = np.array(self.setup.std, dtype=np.float32).reshape(1, 1, 3)


        self.is_training=self.setup.purpose=="train"
        self.is_all= self.setup.generate_all_labels or self.is_training
        self.gen_pre_hm = self.gen_heatmap if self.is_training and call_purp!="val" else self.gen_pure_heatmap


        if self.is_all:
            self.labels,self.cats=load_coco(self.ann_path)
        else:
            self.labels,self.cats=load_coco_test(self.ann_path)

        #transforms
        self.crop=RandomCrop(self.setup.inp_res)
        self.resize=Resize(self.setup.inp_res)
        self.totensor=ToTensor()
        self.res_transform= self.crop if self.setup.is_crop else self.resize

        self.get_item = self.get_item_train if  self.is_all else self.get_item_test
        self.len= len(self.labels)

        self.track_names,self.track_info = None,None
        if not self.is_training:
            self.track_names,self.track_info=self.track_len_info()
            if sampler!=None:
                self.len = self.track_info[sampler]
                si=0
                for i in range(1,sampler):
                    si+=self.track_info[i]
                self.labels=self.labels[si:si+self.len]
            self.init_hm()


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.get_item(idx)

    def get_item_test(self, idx):
        img_path = os.path.join(self.root_path, self.labels[idx]["path"])
        image = cv2.imread(img_path)/255.

        pre_idx=max(0,idx-1)
        img_path = os.path.join(self.root_path, self.labels[pre_idx]["path"])
        pre_image = cv2.imread(img_path)/255.

        if self.setup.test_fix_res:
            image=cv2.resize(image,(self.setup.inp_res[1],self.setup.inp_res[0]))
            pre_image=cv2.resize(pre_image, (self.setup.inp_res[1],self.setup.inp_res[0]))
        else:
            new_h,new_w=self.setup.inp_res
            h,w,c=image.shape if len(image.shape)==3 else (image.shape[0],image.shape[1],1)

            imageZ=np.zeros((new_h,new_w,c))
            imageZ[:h,:w] = image
            image=imageZ
            imageZ=np.zeros((new_h,new_w,c))
            imageZ[:h,:w] = pre_image
            pre_image=imageZ
        if self.setup.norm_color:
            image = ((image - self.setup.mean) / self.setup.std)
            pre_image = ((pre_image - self.setup.mean) / self.setup.std)
        image=self.totensor(image)
        pre_image=self.totensor(pre_image)

        return {"img":image,"pre_img":pre_image,"pre_hm":self.pre_hm}

    def get_item_train(self,idx):
        #current image
        img_path = os.path.join(self.root_path, self.labels[idx]["path"])
        image = cv2.imread(img_path)
        obs = np.array(self.labels[idx]["ann"])
        image,obs,top,left=self.res_transform(image,obs)



        rfc=np.random.random()<self.setup.random_flip

        if rfc and self.is_training:
            image = image[:, ::-1, :].copy()
            obs[:,1]=self.setup.inp_res[1]-obs[:,1]-1

        image=image/255.
        kk=image.copy()
        if self.setup.norm_color:
            image = ((image - self.setup.mean) / self.setup.std)
        image=self.totensor(image)

        #previous image
        pre_obs=None
        if "pre_hm" in self.setup.keys or "tracking" in self.setup.keys:
            pre_image,pre_obs=self.get_pre_data(idx,previous_image_distance=random.randint(1,self.setup.max_frame_dist))

            pre_image,pre_obs,_,_=self.res_transform(pre_image,pre_obs,top,left) if self.setup.use_same_augments else self.crop(pre_image,pre_obs)

            if rfc and self.is_training:
                pre_image = pre_image[:, ::-1, :].copy()
                pre_obs[:,1]=self.setup.inp_res[1]-pre_obs[:,1]-1

            pre_image=pre_image/255.
            if self.setup.norm_color:
                pre_image = ((pre_image - self.setup.mean) / self.setup.std)
            pre_image=self.totensor(pre_image)


        sample=self.init_sample()
        if self.is_all:
            sample=self.gen_data(sample,obs,pre_obs,img=kk)
        sample["img"]=image

        if "pre_hm" in self.setup.keys or "tracking" in self.setup.keys:
            sample["pre_hm"]= self.gen_pre_hm(pre_obs) if self.is_all or idx==0 else []
            sample["pre_img"]=pre_image


        return sample

    def init_hm(self):
        _,obs=self.get_pre_data(0,previous_image_distance=0)

        if self.setup.test_fix_res:
            hi,wi=_.shape[:2]
            cy,cx=hi/self.setup.inp_res[0],wi/self.setup.inp_res[1]
            obs[:,(1,3)]*=cx
            obs[:,(2,4)]*=cy

        else:
            self.setup.real_res=_.shape[:2]
            ss=np.array(_.shape[:2])
            temp=ss//16
            temp2=ss%16
            ss[0]=16*temp[0] if temp2[0]==0 else temp[0]*16+16
            ss[1]=16*temp[1] if temp2[1]==0 else temp[1]*16+16
            self.setup.inp_res,self.setup.out_res=ss,ss.copy()

        self.pre_hm=self.gen_pure_heatmap(obs)

    def gen_heatmap(self,anns):
        heatmap=np.zeros((self.setup.regression_head_dims["hm"],self.setup.out_res[0],self.setup.out_res[1]), dtype=np.float32)
        num_objs = min(len(anns), self.setup.max_objs)
        for _,x,y,w,h,cls_id in anns:
            if cls_id!=0 and cls_id in self.setup.id2cls:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(self.min_rad, int(radius))

                #CT noise in center locations
                x += np.random.randn() * self.setup.hm_disturb * w
                y += np.random.randn() * self.setup.hm_disturb * h

                #CT random center eraser
                conf = 1 if np.random.random() > self.setup.lost_disturb else 0

                #HM random conf reduction
                conf = 1-np.random.random()*self.setup.rcr_max if conf and np.random.random() < self.setup.rcr_prob else conf

                #HM random center propagation
                if np.random.random()<self.setup.error_prop:
                    ecp1=int(np.random.randint(self.setup.error_prop_count, size=1))+1
                    #print(ecp1)
                    for i in range(ecp1):
                        ax,ay=x-np.random.randn()*w*0.05,y-np.random.randn()*0.05*h
                        draw_umich_gaussian(heatmap[(int(cls_id) - 1)%self.setup.regression_head_dims["hm"]], (ax,ay), radius,k=conf*1)

                draw_umich_gaussian(heatmap[(int(cls_id) - 1)%self.setup.regression_head_dims["hm"]], (x,y), radius,k=conf)



        return heatmap

    def gen_pure_heatmap(self,anns):
        heatmap=np.zeros((self.setup.regression_head_dims["hm"],self.setup.out_res[0],self.setup.out_res[1]), dtype=np.float32)
        num_objs = min(len(anns), self.setup.max_objs)
        for _,x,y,w,h,cls_id in anns:
            if cls_id!=0 and cls_id in self.setup.id2cls:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(self.min_rad, int(radius))
                draw_umich_gaussian(heatmap[(int(cls_id) - 1)%self.setup.regression_head_dims["hm"]], (x,y), radius,k=1)
        return heatmap

    def get_pre_data(self,idx,previous_image_distance=1):
        pre_idx=idx-previous_image_distance
        if self.labels[pre_idx]["video_id"]!=self.labels[idx]["video_id"]:
            pre_idx=idx
        pre_img_path = os.path.join(self.root_path, self.labels[pre_idx]["path"])
        pre_image = cv2.imread(pre_img_path)

        pre_obs = np.array(self.labels[pre_idx]["ann"])
        return pre_image,pre_obs

    def init_sample(self):
        sample={}
        sample["hm"]=np.zeros((self.setup.regression_head_dims["hm"],self.setup.out_res[0],self.setup.out_res[1]), dtype=np.float32) if self.is_all else []
        sample['ind'] = np.zeros((self.setup.max_objs), dtype=np.int64) if self.is_all else []
        sample['mask'] = np.zeros((self.setup.max_objs), dtype=np.float32) if self.is_all else []
        sample['cat'] = np.zeros((self.setup.max_objs), dtype=np.int64) if self.is_all else []
        sample['cat2'] = np.zeros((self.setup.max_objs), dtype=np.int64) if self.is_all else []
        sample['cls'] = np.zeros((self.setup.max_objs, self.setup.num_classes), dtype=np.float32) if self.is_all else []

        for head in self.setup.regression_head_dims:
            if head in self.setup.keys:
                sample[head] = np.zeros((self.setup.max_objs, self.setup.regression_head_dims[head]), dtype=np.float32) if self.is_all else []
                sample[head + '_mask'] = np.zeros((self.setup.max_objs, self.setup.regression_head_dims[head]), dtype=np.float32) if self.is_all else []
        return sample

    def gen_data(self,sample,anns,pre_anns=None,img=0):
        aaa=self.setup.norm_coef
        num_objs = min(len(anns), self.setup.max_objs)
        for ii in range(num_objs):
            track,x,y,w,h,cls_id=anns[ii]
            if cls_id!=0 and cls_id in self.setup.id2cls:
                ct_int=np.array([x,y]).astype(int)
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(self.min_rad, int(radius))
                draw_umich_gaussian(sample["hm"][(int(cls_id) - 1)%self.setup.regression_head_dims["hm"]], (x,y), radius)

                sample['cat'][ii] = 0 if self.setup.task=="mot2" else cls_id - 1 #
                sample['cat2'][ii] = cls_id - 1
                sample['cls'][ii][int(cls_id - 1)] = 1.0
                sample['ind'][ii] = ct_int[1] * self.setup.out_res[1] + ct_int[0]
                sample['mask'][ii] = 1.0

                if 'wh' in self.setup.keys:
                    sample['wh'][ii] = (1. * w, 1. * h) if not self.setup.norm_wh else (w/self.setup.inp_res[1]*aaa ,h/self.setup.inp_res[0]*aaa)
                    sample['wh_mask'][ii] = 1.0
                if 'reg' in self.setup.keys:
                    sample['reg'][ii] = np.array([x-ct_int[0],y-ct_int[1]])
                    sample['reg_mask'][ii] = 1.0

                if 'tracking' in self.setup.keys:
                    pre_ct=pre_anns[pre_anns[:,0]==track]
                    if len(pre_ct)>0:
                        sample['tracking_mask'][ii] = 1.0
                        sample['tracking'][ii] = (pre_ct[0,1:3] - ct_int) if not self.setup.norm_offset else ((pre_ct[0,1] - ct_int[0])/self.setup.inp_res[1]*aaa, (pre_ct[0,2] - ct_int[1])/self.setup.inp_res[0]*aaa)
                         
            else:
                region=sample["hm"][:,int(y-h):int(y+h)+1,int(x-w):int(x+w)+1]
                np.maximum(region,self.setup.ignore_val, out=region)

        return sample

    def track_len_info(self):
        cdict={}
        track_names=[]
        for img in self.labels:
            if img["video_id"] in cdict:
                cdict[img["video_id"]]+=1
            else:
                cdict[img["video_id"]]=1
            ss=img["path"].split("/")[0]
            if ss not in track_names:
                track_names.append(ss)
        return track_names,cdict
