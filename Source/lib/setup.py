from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
from .data.data_tools.tools import read_json

#https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

class Setup(object):


    def __init__(self):
        self.set_defaults()
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--task', default='mot', help='mot | detection(Not Ready)')
        self.parser.add_argument('--purpose', default='train', help='train | test')

        #dataset
        self.parser.add_argument('--dataset_conf', default='/home/arslan/Desktop/HM-Net/Source/data_conf/visdrone.json', help='path to config file')
        self.parser.add_argument('--dataset_purp', default='', help='path to config file')
        self.parser.add_argument('--is_validate',default=True, type=str2bool, nargs='?', const=True, help='')

        self.parser.add_argument('--max_objs', default=256, type=int, help='')
        self.parser.add_argument('--generate_all_labels', default=False, type=str2bool, nargs='?', const=True, help='To speed up testing discards un necesary labels. Effective only on testing')

        self.parser.add_argument('--ignore_val', default=1, type=float, help='')
        self.parser.add_argument('--hm_disturb', default=0.05, type=float, help='')
        self.parser.add_argument('--lost_disturb', default=0.1, type=float, help='')
        self.parser.add_argument('--error_prop', default=0.2, type=float, help='')
        self.parser.add_argument('--error_prop_count', default=5, type=int, help='')
        self.parser.add_argument('--rcr_max', default=0.8, type=float, help='')
        self.parser.add_argument('--rcr_prob', default=0.35, type=float, help='')
        self.parser.add_argument('--max_frame_dist', default=4, type=int, help='')
        self.parser.add_argument('--random_flip', default=0.5, type=float, help='')
        self.parser.add_argument('--inp_w', default=-1, type=int, help='-1 is default')
        self.parser.add_argument('--inp_h', default=-1, type=int, help='-1 is default')
        self.parser.add_argument('--down_ratio', default=1, type=int, help='not added yet 1:1 input output resolution is used')

        self.parser.add_argument('--bbox_w', default=11, type=int, help='training and testing w/o wh')
        self.parser.add_argument('--bbox_h', default=11, type=int, help='training and testing w/o wh')
        self.parser.add_argument('--use_same_augments',default=True,type=str2bool, nargs='?', const=True, help="previous image has same data augmentations")
        self.parser.add_argument('--norm_wh',default=True,type=str2bool, nargs='?', const=True, help="normalizes width and heihgt with image dimensions")
        self.parser.add_argument('--norm_offset',default=False,type=str2bool, nargs='?', const=True, help="normalizes offset vector with image sizes")
        self.parser.add_argument('--norm_color',default=False,type=str2bool, nargs='?', const=True, help="normalizes offset vector with image sizes")
        self.parser.add_argument('--ltrb',default=False,type=str2bool, nargs='?', const=True, help="not added yet")
        self.parser.add_argument('--is_crop', default=True,type=str2bool, nargs='?', const=True, help='')

        #trainer
        self.parser.add_argument('--exp_id', default='exp_1', help='name of the experiment')
        #self.parser.add_argument('--resume', default=False,type=bool help='resumes given experiment')
        self.parser.add_argument('--exp_root', default='../Experiments', help='experiments will be stored at this path')
        self.parser.add_argument('--epochs', default=30,type=int , help='')
        self.parser.add_argument('--max_iter', default=-1,type=int , help='max iter count in an epoch')
        self.parser.add_argument('--lr', default=0.00013,type=float , help='')
        self.parser.add_argument('--batch_size', default=2,type=int , help='')
        self.parser.add_argument('--loss_tag', default='generic_loss', help='check lib.loss.losses')
        self.parser.add_argument('--train_tag', default='base', help='check lib.train.trainers')
        self.parser.add_argument('--verbose_period', default=0,type=int , help='display period outputs on cmd')
        self.parser.add_argument('--save_period', default=1,type=int , help='')
        self.parser.add_argument('--is_save_model_batch', default=True,type=str2bool, nargs='?', const=True, help='')
        self.parser.add_argument('--save_model_batch', default=250,type=int, help='')
        self.parser.add_argument('--save_checkpoint', default=True,type=str2bool, nargs='?', const=True, help='')
        self.parser.add_argument('--load_optimizer', default=True,type=str2bool, nargs='?', const=True, help='')

        self.parser.add_argument('--start_from_zero', default=False,type=str2bool, nargs='?', const=True, help='')
        self.parser.add_argument('--use_logger', default=True,type=str2bool, nargs='?', const=True,   help='')

        #loss weights (update for new features)
        self.parser.add_argument("--use_default_weights",default=False,type=str2bool, nargs='?', const=True,help="use default dict of weights")
        self.parser.add_argument('--hm_weight', default=1,type=float , help='loss weight')
        self.parser.add_argument('--wh_weight', default=1,type=float , help='loss weight')
        self.parser.add_argument('--reg_weight', default=0.01,type=float , help='loss weight')
        self.parser.add_argument('--track_weight', default=1,type=float , help='loss weight')
        self.parser.add_argument('--cls_weight', default=1,type=float , help='loss weight')

        #tester
        self.parser.add_argument('--save_video', default=False,type=str2bool, nargs='?', const=True, help='video_test_output')
        self.parser.add_argument('--fps', default=4, type=int , help='output video fps')
        self.parser.add_argument('--use_video_bbox', default=True,type=str2bool, nargs='?', const=True, help='')
        self.parser.add_argument('--test_tag', default='base', help='check lib.test.testers')
        self.parser.add_argument('--results_path', default='../Results', help='test output path')
        self.parser.add_argument('--test_id', default='', help='name of the the test, leave empty for auto naming ')
        self.parser.add_argument('--feed_thres', default=0.1,type=float , help='feedback threshold')
        self.parser.add_argument('--det_thres', default=0.20,type=float , help='detection threshold')
        self.parser.add_argument('--max_age', default=5,type=int , help='')
        self.parser.add_argument('--is_challenge', default=True,type=str2bool, nargs='?', const=True, help='todo later')
        self.parser.add_argument('--is_public', default=False,type=str2bool, nargs='?', const=True, help='')
        self.parser.add_argument('--is_hungarian', default=True,type=str2bool, nargs='?', const=True,  help='')
        self.parser.add_argument('--pre_hm_init_zero', default=False,type=str2bool, nargs='?', const=True, help='')
        self.parser.add_argument('--nms', default=5, type=int , help='non maxima suppression window size, should be an odd number')
        self.parser.add_argument('--test_fix_res', default=True, type=str2bool, nargs='?', const=True, help='')
        self.parser.add_argument('--norm_coef',default=10,type=float,help="")

        #model
        self.parser.add_argument('--model_tag', default='HNetX', help='check lib.model.models')#prediction heads format
        self.parser.add_argument('--backbone_tag', default='HNet_v3', help='check lib.model.backbones')#model architecture
        self.parser.add_argument('--features', default=32,type=int , help='depth of model blocks')
        self.parser.add_argument('--load_model', default="" , help='model path')

        #system
        self.parser.add_argument('--gpus', default="0", help='set -1 for cpu')
        self.parser.add_argument('--visible_gpus', default="all" , help=" write 'all' for seeing all")
        self.parser.add_argument('--num_workers', default=1,type=int , help='data loader workers')
        self.parser.add_argument('--pin_memory', default=True,type=str2bool, nargs='?', const=True, help='')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true', help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=123,help='random seed')




    def parse(self, args=''):
        if args == '':
            setup = self.parser.parse_args()
        else:
            setup = self.parser.parse_args(args)

        assert setup.task in self.tasks, f"unidentified input: {setup.task}"
        assert setup.purpose in self.purposes, f"unidentified input: {setup.purpose}"

        dataset=read_json(setup.dataset_conf)

        purp=["test","train","val","challenge"]
        task_info=self.tasks[setup.task]
        setup.regression_head_dims = task_info["regression_head_dims"]
        setup.keys = task_info["keys"]

        data=dataset[setup.dataset_purp if setup.dataset_purp in purp else setup.purpose]

        setup.num_classes=len(data["classes"])
        setup.regression_head_dims["hm"]=setup.num_classes
        if setup.task=="mot2":
            setup.regression_head_dims["hm"]=1
            setup.regression_head_dims["cls"]=setup.num_classes

        setup.cls2id={data["class_name"][i]:i for i in range(len(data["class_name"]))}
        setup.id2cls={i:data["class_name"][i] for i in range(len(data["class_name"]))}
        setup.dataset_tag=dataset["tag"] #selector name needed for datasets.get_dataset

        if data["annotations_coco"].endswith(".json"):
            setup.ann_path=data["annotations_coco"]
        else:
            raise NotImplementedError
            ## TODO: create a converter to coco for custom datasets
        setup.seq_path=data["image_sequences"]
        setup.mean=dataset["mean"]
        setup.std=dataset["std"]

        if setup.inp_w ==-1 or setup.inp_h==-1:
            setup.inp_res=self.default_res
        else:
            assert setup.inp_w >=64 or setup.inp_h>=64, "invalid input resolution"
            setup.inp_res=(setup.inp_h,setup.inp_w)

        ## TODO: fix when we add down sample
        setup.out_res=setup.inp_res

        #rewrite same experiment again if true put here for just test purposes
        setup.accept_same_id=True

        #CUDA Settings
        setup.gpu_list=[int(i) for i in setup.gpus.split(",")]
        if setup.visible_gpus!="all":
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]=setup.visible_gpus
        if setup.is_validate:
            setup.val_ann_path=dataset["val"]["annotations_coco"]
            setup.val_seq_path=dataset["val"]["image_sequences"]
        #Model Settings
        setup.image_channels=[int(len(setup.mean)) for i in range(self.number_of_input_images)]
        #loss weightss
        setup.weight_dict= task_info["weight_dict"]#defaults
        if not setup.use_default_weights:
            if "hm" in setup.weight_dict:
                setup.weight_dict["hm"]=setup.hm_weight
            if "wh" in setup.weight_dict:
                setup.weight_dict["wh"]=setup.wh_weight
            if "reg" in setup.weight_dict:
                setup.weight_dict["reg"]=setup.reg_weight
            if "tracking" in setup.weight_dict:
                setup.weight_dict["tracking"]=setup.track_weight
            if "cls" in setup.weight_dict:
                setup.weight_dict["cls"]=setup.cls_weight
        #test
        setup.test_input=task_info["test_input"]
        print(setup)
        return setup


    def set_defaults(self):
        self.version="stable"
        self.versions=["stable","dev0"]
        self.purposes=["train","test","demo"]
        self.mot_settings={"regression_head_dims":{'hm': None, 'reg': 2, 'wh': 2, 'tracking': 2},
                           "keys":["reg","wh","tracking","pre_hm"], #controls creation of these optional features
                           "weight_dict" :{'hm': 1, 'wh': 1,'reg': 0.1,'tracking': 1,},
                           "test_input":["img","pre_hm","pre_img"]
        }
        self.wami_mot_settings={"regression_head_dims":{'hm': None, 'reg': 2, 'tracking': 2},
                           "keys":["reg","tracking","pre_hm"], #controls creation of these optional features
                           "weight_dict" :{'hm': 1, 'reg': 0.1,'tracking': 1},
                           "test_input":["img","pre_hm","pre_img"]
        }
        self.sat_mot_settings={"regression_head_dims":{'hm': None, 'wh': 2, 'tracking': 2},
                           "keys":["wh","tracking","pre_hm"], #controls creation of these optional features
                           "weight_dict" :{'hm': 1, 'wh': 1,'reg': 0.1,'tracking': 1,},
                           "test_input":["img","pre_hm","pre_img"]
        }
        self.mot2_settings={"regression_head_dims":{'hm': 1, 'wh': 2, 'tracking': 2,"reg" : 2, "cls" : None},
                           "keys":["wh","tracking","pre_hm","reg","cls"], #controls creation of these optional features
                           "weight_dict" :{'hm': 1, 'wh': 1,'reg': 0.1,'tracking': 1,"cls":1},
                           "test_input":["img","pre_hm","pre_img"]
        }
        self.tasks={"mot":self.mot_settings,"mot2":self.mot2_settings,"sat_mot":self.sat_mot_settings,"wami_mot":self.wami_mot_settings}
        self.default_res=(608,1088)
        self.number_of_input_images=2

if  __name__ == '__main__':
    setup=Setup()
    setup.parse([])
