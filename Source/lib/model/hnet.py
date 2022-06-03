from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import get_backbone


class HNet(nn.Module):
    def __init__(self,setup):
        super(HNet, self).__init__()
        self.heads=setup.regression_head_dims
        self.backbone=get_backbone(setup.backbone_tag)(in_channels=[setup.num_classes,*setup.image_channels],features=setup.features)
        #self.wh1=nn.Conv2d(features,features,kernel_size=3, stride=1, padding=1, bias=True)
        for head in self.heads:
            head_conv=nn.Conv2d(setup.features,self.heads[head],kernel_size=1, stride=1, padding=0, bias=True)
            self.__setattr__(head, head_conv)


    def forward(self,sample):
        x=self.backbone(sample["pre_hm"],sample["pre_img"],sample["img"])
        out={}
        for head in self.heads:
            if "hm" in head:
                out[head]=torch.sigmoid(self.__getattr__(head)(x))
            else:
                out[head]=self.__getattr__(head)(x)
        return out


if __name__=="__main__":
    y, x=400,640
    # simple test run
    net = HNet(num_classes=11,heads={}).to("cuda")
    print(net)
    criterion = nn.MSELoss().to("cuda")
    optimizer = torch.optim.Adam(net.parameters())
    print('Network initialized. Running a test batch.')
    for _ in range(10):
        with torch.set_grad_enabled(True):
            batch = {"pre_hm":torch.empty(1, 11, y, x).normal_().to("cuda"),"pre_image":torch.empty(1, 3, y, x).normal_().to("cuda"),"image":torch.empty(1, 3,  y, x).normal_().to("cuda")}
            targets = torch.empty(1, 11,  y, x).normal_().to("cuda")
            out = net(batch)
            loss = criterion(out["hm"], targets)
            loss.backward()
            optimizer.step()
        print(out["hm"].shape)
