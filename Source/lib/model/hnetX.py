from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import get_backbone


class HNetX(nn.Module):
    def __init__(self,setup):
        super(HNetX, self).__init__()
        self.heads=setup.regression_head_dims

        self.backbone=get_backbone(setup.backbone_tag)(in_channels=[self.heads["hm"],*setup.image_channels],features=setup.features)
        if "wh" in self.heads:
            self.wh1=nn.Conv2d(setup.features,setup.features,kernel_size=5, stride=1, padding=2, bias=True)
            self.wh2=nn.Conv2d(setup.features,setup.features,kernel_size=3, stride=1, padding=1, bias=True)
        if "tracking" in self.heads:
            self.tr1=nn.Conv2d(setup.features,setup.features,kernel_size=5, stride=1, padding=2, bias=True)
            self.tr2=nn.Conv2d(setup.features,setup.features,kernel_size=3, stride=1, padding=1, bias=True)
        if "cls" in self.heads:
            self.ct1=nn.Conv2d(setup.features,setup.features,kernel_size=5, stride=1, padding=2, bias=True)
            self.ct2=nn.Conv2d(setup.features,setup.features,kernel_size=3, stride=1, padding=1, bias=True)
        for head in self.heads:
            head_conv=nn.Conv2d(setup.features,self.heads[head],kernel_size=1, stride=1, padding=0, bias=True)
            self.__setattr__(head, head_conv)

    def forward(self,sample):
        x=self.backbone(sample["pre_hm"],sample["pre_img"],sample["img"])
        out={}
        for head in self.heads:
            if "hm" in head:
                out[head]=torch.sigmoid(self.__getattr__(head)(x))
            elif "cls" in head:
                x=F.leaky_relu(self.ct1(x))
                x=F.leaky_relu(self.ct2(x))
                out[head]=torch.sigmoid(self.__getattr__(head)(x))
            elif "wh" in head:
                x=F.leaky_relu(self.wh1(x))
                x=F.leaky_relu(self.wh2(x))
                out[head]=self.__getattr__(head)(x)
            elif "tracking" in head:
                x=F.leaky_relu(self.tr1(x))
                x=self.tr2(x)
                out[head]=self.__getattr__(head)(x)
            else:
                out[head]=self.__getattr__(head)(x)
        return out

# if __name__=="__main__":
#     y, x=400,640
#     # simple test run
#     net = HNetX(num_classes=11,heads={}).to("cuda")
#     print(net)
#     criterion = nn.MSELoss().to("cuda")
#     optimizer = torch.optim.Adam(net.parameters())
#     print('Network initialized. Running a test batch.')
#     for _ in range(10):
#         with torch.set_grad_enabled(True):
#             batch = {"pre_hm":torch.ones(1, 11, y, x).normal_().to("cuda"),"pre_image":torch.ones(1, 3, y, x).normal_().to("cuda"),"image":torch.ones(1, 3,  y, x).normal_().to("cuda")}
#             targets = torch.empty(1, 11,  y, x).normal_().to("cuda")
#             out = net(batch)
#             loss = criterion(out["hm"], targets)
#             loss.backward()
#             optimizer.step()
#         print(out["hm"].shape)
