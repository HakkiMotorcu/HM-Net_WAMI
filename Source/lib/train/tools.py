from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import json
import torch
import time
from progress.bar import Bar

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

def set_exp_path(setup):
    path=os.path.join(setup.exp_root,setup.exp_id)
    ret=mkdir(path)
    if not (ret or setup.accept_same_id):
        exps=os.listdir(setup.exp_root)
        inp=input(f"'{path}' already exists\nall existing experiments:\n '{os.listdir(setup.exp_root)}'\n specify new exp id to proceed: \n")
        if inp!="proceed":
            setup.exp_id=inp
            return set_exp_path(setup)
        else:
            setup.exp_path=path
            print(f"Command override\nCurrent experiment path: {path}")
            return False
    print(f"\nCurrent experiment path: {path}")
    setup.exp_path=path
    return True

def set_exp(setup):
    set_exp_path(setup)
    mkdir(os.path.join(setup.exp_path,"logs"))
    mkdir(os.path.join(setup.exp_path,"models"))

#From FairMOT
class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch)
        loss, loss_stats = self.loss(outputs, batch)
        
        return outputs, loss, loss_stats

#From FairMOT
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
