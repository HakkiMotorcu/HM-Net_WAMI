# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ffl import FastFocalLoss
from .RegWeightedL1Loss import RegWeightedL1Loss
from .WeightedClassLoss import WeightedClassLoss



class GenericLoss(torch.nn.Module):
  def __init__(self,setup):
    super(GenericLoss, self).__init__()
    self.crit = FastFocalLoss()
    self.crit_cat = WeightedClassLoss()
    self.crit_reg = RegWeightedL1Loss()
    self.weight_dict=setup.weight_dict
    self.regression_heads=list(setup.regression_head_dims.keys())

  def forward(self, output,batch):
    #output=batch['out']
    losses = {head: 0 for head in output.keys()}
    if 'hm' in output:
        losses['hm'] += self.crit(output["hm"], batch['hm'], batch['ind'],batch['mask'], batch['cat'])
    if "cls" in output:
        losses['cls'] += self.crit_cat(output["cls"], batch['cls'],batch['mask'], batch['ind'], batch['cat2'])
    for head in self.regression_heads:
        if head in output and head!="hm":
            losses[head] += self.crit_reg(output[head], batch[head + '_mask'],batch['ind'], batch[head])
    losses['tot'] = 0
    for head in self.weight_dict.keys():
        if head in output:
            losses['tot'] += self.weight_dict[head] * losses[head]
    return losses['tot'],losses
