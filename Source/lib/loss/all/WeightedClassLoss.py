from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
def _clamper(x):
  y = torch.clamp(x, min=1e-4, max=1-1e-4)
  return y
def _only_neg_loss(pred, gt):
  gt = torch.pow(1 - gt, 4)

  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * gt
  return neg_loss.sum()

class WeightedClassLoss(nn.Module):
  def __init__(self):
    super(WeightedClassLoss, self).__init__()
    self.loss = nn.CrossEntropyLoss(reduction='none',ignore_index=-1)
    self.alpha=1
    self.gamma=2
    self.only_neg_loss = _only_neg_loss

  def forward(self, output, target,mask, ind, cat):
    pred = _tranpose_and_gather_feat(output, ind)
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    #loss = self.loss(pred , target)
    #pt=torch.exp(-loss)
    #F_loss = self.alpha * torch.pow((1-pt),self.gamma) * loss
    #F_loss=F_loss.sum()
    #loss = F_loss / (mask.sum() + 1e-4)
    #return loss
    
    out=_clamper(output)
    
    pos_pred_pix = _tranpose_and_gather_feat(out, ind) # B x M x C
    neg_loss = self.only_neg_loss(pos_pred_pix, target) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
        return - neg_loss
    return - (pos_loss + neg_loss) / num_pos




    

def _gather_feat(feat, ind):
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  return feat

def _tranpose_and_gather_feat(feat, ind):
  feat = feat.permute(0, 2, 3, 1).contiguous()
  feat = feat.view(feat.size(0), -1, feat.size(3))
  feat = _gather_feat(feat, ind)
  return feat
