from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import math
import time
import numpy as np
import torch
from progress.bar import Bar
import matplotlib.pyplot as plt
from lib.test.tracker_core import TrackerCore
from .tools import decoder,AverageMeter,gaussian_radius,gaussian2D,draw_umich_gaussian,mkdir

class BaseTester(object):

    def __init__(self,setup,model,dataset):
        self.setup = setup
        self.dataset=dataset
        dataset=dataset(setup=setup)

        #model setup
        self.model = model
        self.load_model()
        self.model = model.to(self.setup.device)
        self.model.eval()
        torch.cuda.empty_cache()


        self.track_info= dataset.track_info
        self.track_names= dataset.track_names
        self.tracker=TrackerCore(setup)
        self.init_all()
        self.get_img_shapes()
        #print(self.track_shapes,dataset.track_info,dataset.track_names)

    def run_all(self):
        with torch.no_grad():
            for track_id in self.track_info:
                print(self.str_out["run0"].format(track_name=self.track_names[track_id-1],tr_len=self.track_info[track_id]) )
                track_infos=[]
                tt=[]
                self.test_loader=self.init_data_loader(self.dataset(setup=self.setup,sampler=track_id))
                print("Test Resolution :",self.setup.inp_res)
                self.tracker.reset()
                for idx,batch in enumerate(self.test_loader):
                    #timerS
                    ss=time.perf_counter()
                    batch["pre_hm"]=self.pre_hm if idx!=0 or self.setup.pre_hm_init_zero else batch["pre_hm"]
                    for k in batch:
                        if k in self.setup.test_input:
                            batch[k] = batch[k].to(device= self.setup.device, non_blocking=True)
                    #timerS
                    output = self.model(batch)
                    torch.cuda.synchronize()
                    #timerS
                    info,rest=self.get_outputs(output)
                    #timerS
                    track_info=self.tracker.step(info)
                    #timerS

                    self.pre_hm=self.SGR(rest,batch)
                    #timerS
                    track_infos.append(track_info)
                    #timerE
                    torch.cuda.synchronize()
                    tt.append(time.perf_counter()-ss)
                    #print(rest[:,:,4])
                    if(idx+1)%20==0:
                        print(f"Batch: {idx+1}, Average FPS: {len(tt)/sum(tt):.3f}, {len(track_info)}")

                print(f"Average Track FPS: {len(tt)/sum(tt):.3f}")
                self.write_results(track_infos,track_id)
                if self.setup.save_video:
                    self.save_videos(track_infos,track_id,self.setup.fps)

    def init_data_loader(self,dataset):
        print("Init data pipeline:",end="")
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.setup.num_workers,
            pin_memory= self.setup.pin_memory,
            drop_last=False
        )
        print("Done")
        return data_loader

    def write_results(self,track_infos,track_id):
        print(self.str_out["wr0"],end="")
        hi,wi=self.track_shapes[track_id-1][:2]
        (ho,wo)=self.setup.inp_res if self.setup.test_fix_res else self.setup.real_res
        ratios=np.array([wi/wo,hi/ho,wi/wo,hi/ho])
        #print(ratios)
        with open(os.path.join(self.setup.test_path,self.track_names[track_id-1]+".txt"),"w") as file:
            ss=""
            for ii,frame in enumerate(track_infos):
                for obj in frame:
                    if obj["active"]!=0 and obj["score"]>0.05:
                        tmp=obj["bbox"]*ratios
                        xt,yt,w,h=np.round(tmp).astype(int)
                        #print(xt,yt,w,h)
                        ss+=f'{ii+1},{obj["tracking_id"]},{xt},{yt},{w},{h},{obj["score"]},{obj["class"]},0,0\n'
                        #xc,yc=xt+w/2,yt+h/2 
                        #if obj["class"]==0:
                        #    ss+=f'{ii+1},{obj["tracking_id"]},{xt},{yt},{w},{h},{1},{int(obj["class"])+1},1\n'
                        #else:
                        #    ss+=f'{ii+1},{obj["tracking_id"]},{xt},{yt},{w},{h},{1},{int(obj["class"])+1},1\n'
            file.write(ss[:-1])
        print("done")

    def save_videos(self,track_infos,track_id,fps=20,rad=3):
        print(self.str_out["sv0"],end="")
        #get images
        image_folder=os.path.join(self.setup.seq_path,self.track_names[track_id-1])
        images = [img for img in os.listdir(image_folder)]
        images.sort()

        #get dimensions
        hi,wi=self.track_shapes[track_id-1][:2]
        (ho,wo)=self.setup.inp_res if self.setup.test_fix_res else self.setup.real_res
        ratios=np.array([wi/wo,hi/ho,wi/wo,hi/ho])

        random_color_dict=np.random.randint(256,size=(400,3))

        video_file_path=os.path.join(self.setup.test_path,"videos",f"{self.track_names[track_id-1]}.mp4")
        out = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (wi,hi))

        for ii,filename in enumerate(images):
            img = cv2.imread(os.path.join(image_folder, filename))

            for obj in track_infos[ii]:
                if obj["active"]!=0 and obj["score"]>0.05:
                    tmp=obj["bbox"]*ratios
                    xt,yt,w,h=np.round(tmp).astype(int)
                    tr_id,clss,score=obj["tracking_id"],int(obj["class"]),obj["score"]
                    color=tuple(random_color_dict[int(tr_id)%400])
                    color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
                    if self.setup.use_video_bbox:
                        cv2.rectangle(img, (xt-1, yt-1), (xt+w+1, yt+h+1),color, 2)
                        y = yt - 15 if yt - 15 > 15 else yt + 15
                        cv2.putText(img, f"{self.setup.id2cls[clss+1]} {tr_id} %{score*100:.2f}", (xt, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[clss], 2)
                        cv2.arrowedLine(img,(int(xt+w/2),int(yt+h/2)) ,(int(xt+w/2+obj["tracking"][0]),int(yt+h/2+obj["tracking"][1])),[0,255,255], 4)
                    else:
                        x=int(xt+w/2)
                        y=int(yt+h/2)
                        img[y-rad:y+rad,x-rad:x+rad,0]=self.colors[clss][0]
                        img[y-rad:y+rad,x-rad:x+rad,1]=self.colors[clss][1]
                        img[y-rad:y+rad,x-rad:x+rad,2]=self.colors[clss][2]
                #print(obj["tracking"])
            out.write(img)

        out.release()

        print("done")

    def SGR(self,info,batch,booster=1.2):
        setup=self.setup
        pre_hm=np.zeros((1,setup.regression_head_dims["hm"],setup.inp_res[0],setup.inp_res[1]), dtype=np.float32)
        info=info[0]
        filtered_dets=info[info[:,4]>=self.setup.feed_thres]
        for line in filtered_dets:
            radius = gaussian_radius((math.ceil(line[3]), math.ceil(line[2])))
            radius = max(3, int(radius))
            draw_umich_gaussian(pre_hm[0,(int(line[5].astype(int)) - 1)%setup.regression_head_dims["hm"]], (line[-2:].astype(int)), radius, k= line[4]*booster)
        #plt.imshow(pre_hm[0].sum(0))
        #plt.colorbar()
        #plt.show()

        return torch.tensor(pre_hm)

    def get_outputs(self,output):
        det,ind=decoder(self.setup,output)
        det=det.detach().cpu().numpy()
        meta=[]

        for line in det[0]:
            meta.append({'score': line[4], 'class': line[5], 'ct': line[-2:], 'tracking':line[6:8], 'bbox':line[:4]})

        return meta,det

    def get_img_shapes(self):
        self.track_shapes=[]
        for i in self.track_names:
            track=os.path.join(self.setup.seq_path,i)
            self.track_shapes.append(cv2.imread(os.path.join(track,os.listdir(track)[0])).shape[:2])

    def load_model(self):
        check=torch.load(self.setup.load_model)
        self.model.load_state_dict(check['model'])

    def init_all(self):
        if self.setup.save_video:
            mkdir(os.path.join(self.setup.test_path,"videos"))
        self.colors=[[0,0,255],[255,0,0],[0,255,0],[0,255,255],[255,0,255],[255,255,0],[125,125,0],[125,0,125],[0,125,125],[125,0,0],[0,125,0],[0,0,125]]
        self.pre_hm=torch.zeros((self.setup.num_classes,self.setup.inp_res[0],self.setup.inp_res[1]), dtype=torch.float32).to(device= self.setup.device, non_blocking=True)
        self.str_out={
            "wr0":"Saving Track Output: ",
            "sv0":"Saving Video Output: ",
            "run0":"\nTrack_name: {track_name}\nTrack length: {tr_len}\nProceeding...\n",

        }
