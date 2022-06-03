from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch import nn
import math
import numpy as np
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


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

def set_test_path(setup):
    ret=mkdir(setup.results_path)
    if setup.test_id=="":
        tot=len(os.listdir(setup.results_path))
        setup.test_id=f"test{tot+1}"
    path=os.path.join(setup.results_path,setup.test_id)
    setup.test_path=path
    ret=mkdir(setup.test_path)


#CenterTrack
def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
#CenterTrack
def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

#CenterTrack
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

#CenterTrack
def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()

      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = torch.true_divide(topk_inds, width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs
#CenterTrack
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = torch.true_divide(topk_inds, width).int().float()
    topk_xs   = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = torch.true_divide(topk_ind, K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decoder(setup,outputs):
    heat=outputs["hm"]
    batch, cat, height, width = heat.size()
    K=setup.max_objs
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat,kernel=setup.nms)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    #scores, inds, clses, ys, xs = _topk(torch.unsqueeze(heat[:,3],0), K=K)
    #scores, inds, clses, ys, xs = _topk(heat[:,:10], K=K)
    if "reg" in outputs:
        reg = _tranpose_and_gather_feat(outputs["reg"], inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5

    #bura hatalı olabilir[[[]]]
    if "cls" in outputs:
        cls_ = _tranpose_and_gather_feat(outputs["cls"], inds)
        _, clses = torch.max(cls_, 2)
        clses = clses.view(batch, K, 1).float()
        #print(clses)
    else:
        clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    if "wh" in outputs:
        wh = _tranpose_and_gather_feat(outputs["wh"], inds)
        # if setup.ltrb:
        #     wh = wh.view(batch, K, 4)
        # else:
        wh = wh.view(batch, K, 2)
        if setup.norm_wh:
            #wh[..., 0:1]*=setup.inp_res[1]/setup.norm_coef
            #wh[..., 1:2]*=setup.inp_res[0]/setup.norm_coef
            wh[..., 0:1]*=432/setup.norm_coef
            wh[..., 1:2]*=432/setup.norm_coef


        # if setup.ltrb:
        #     bboxes = torch.cat([xs - wh[..., 0:1],
        #                         ys - wh[..., 1:2],
        #                         xs + wh[..., 2:3],
        #                         ys + wh[..., 3:4]], dim=2)
        # else:
        bboxes = torch.cat([xs - wh[..., 0:1],
                                ys - wh[..., 1:2],
                                wh[..., 0:1]*2 ,
                                wh[..., 1:2]*2], dim=2)

    else:
        bboxes = torch.cat([xs - setup.bbox_w / 2,
                                ys - setup.bbox_h / 2,
                                 setup.bbox_w ,
                                 setup.bbox_h ], dim=2)
    if "tracking" in outputs:
        trc = _tranpose_and_gather_feat(outputs["tracking"], inds)

        trc = trc.view(batch, K, 2)
        if setup.norm_offset:
            trc[..., 0:1]*=setup.inp_res[1]/setup.norm_coef
            trc[..., 1:2]*=setup.inp_res[0]/setup.norm_coef
        txs =  trc[:, :, 0:1]
        tys =  trc[:, :, 1:2]

        detections = torch.cat([bboxes, scores, clses,txs,tys,xs,ys], dim=2)

    else:
        detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections, inds

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
