import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.nn import functional as F

from collections import OrderedDict
from .networks.resnet import resnet34,resnet18,resnet50

def create_resnet_encoder(enc_name,pretrained=True,num_classes=5,is_hm=False)  :
    backbone = resnet34(pretrained=False,in_size=num_classes) if is_hm else models.resnet34(pretrained=pretrained)
    modeldict=OrderedDict()
    layer_names=[]
    for name,layer in backbone.named_children():
        if name!="avgpool" and name!="fc":
            modeldict[f"{name}"]=layer
    return  nn.Sequential(modeldict)

def freeze_encoder(backbone):
    """ Freezing encoder parameters, the newly initialized decoder parameters are remaining trainable. """
    for param in backbone.parameters():
        param.requires_grad = False


class HNet_RL0(nn.Module):
    def __init__(self, num_classes=5, init_features=32,features=64,p=[0.5,0.8,0.8,0.8,0.9,0.9]):
        super(HNet_RL0, self).__init__()
        self.dropout_1= nn.Dropout2d(p=p[0])
        self.dropout_2= nn.Dropout2d(p=p[1])
        self.dropout_3= nn.Dropout2d(p=p[2])
        self.dropout_4= nn.Dropout2d(p=p[3])
        self.dropout_5= nn.Dropout2d(p=p[4])
        self.dropout_6= nn.Dropout2d(p=p[5])
        self.backboneA=create_resnet_encoder("encA",num_classes=num_classes,is_hm=True)
        self.backboneB=create_resnet_encoder("encB")
        self.backboneC=create_resnet_encoder("encC")
        freeze_encoder(self.backboneB)
        freeze_encoder(self.backboneC)
        self.poolA4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.poolB4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.poolC4 = nn.MaxPool2d(kernel_size=2, stride=2)

        features=3*features


        self.bottleneck = self._block(features *8, features*16, name="bottleneck")
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d((features * 2), features, kernel_size=2, stride=2)
        self.decoder1 = self._block((features * 2), features, name="dec1a")
        self.upconv1b = nn.ConvTranspose2d(features , features, kernel_size=2, stride=2)
        self.decoder1b = self._block(features*2 , features, name="dec1b")
        self.upconv1c = nn.ConvTranspose2d(features, features, kernel_size=2, stride=2)
        self.decoder1c = self._block(features , features, name="dec1c")



    def forward(self,pre_hm,pre_image,image):
        x,featuresA,featuresB,featuresC = self.forward_encoder(pre_hm,pre_image,image)

        bottleneck = self.bottleneck(torch.cat((self.dropout_1(self.poolA4(featuresA["layer4"])),self.poolB4(featuresB["layer4"]),self.poolC4(featuresC["layer4"])), dim=1).contiguous())

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4,torch.cat((self.dropout_2(featuresA["layer4"]),featuresB["layer4"],featuresC["layer4"]), dim=1)), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, torch.cat((self.dropout_3(featuresA["layer3"]),featuresB["layer3"],featuresC["layer3"]), dim=1)), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2,  torch.cat((self.dropout_4(featuresA["layer2"]),featuresB["layer2"],featuresC["layer2"]), dim=1)), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, torch.cat((self.dropout_5(featuresA["layer1"]),featuresB["layer1"],featuresC["layer1"]), dim=1)), dim=1)
        dec1 = self.decoder1(dec1)
        dec1b = self.upconv1b(dec1)
        dec1b = torch.cat((dec1b, torch.cat((self.dropout_6(featuresA["relu"]),featuresB["relu"],featuresC["relu"]), dim=1)), dim=1)
        dec1b = self.decoder1b(dec1b)
        dec1a = self.upconv1c(dec1b)
        dec1a = self.decoder1c(dec1a)
        return  dec1a

    def forward_encoder(self,pre_hm,pre_image,image):
        featuresA,featuresB,featuresC={},{},{}
        for name, child in self.backboneA.named_children():
            pre_hm = child(pre_hm)
            if "maxpool" not in name and "bn" not in name :
                featuresA[name] = pre_hm
        for name, child in self.backboneB.named_children():
            pre_image = child(pre_image)
            if "maxpool" not in name and "bn" not in name :
                featuresB[name] = pre_image
        for name, child in self.backboneC.named_children():
            image = child(image)
            if "maxpool" not in name and "bn" not in name :
                featuresC[name] = image
        return torch.cat((pre_hm,pre_image,image),dim=1), featuresA,featuresB,featuresC

    def _block(self,in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
# if __name__=="__main__":
#
#     # simple test run
#     net = HNet_V3L().to("cuda")
#     print(net)
#     criterion = nn.MSELoss().to("cuda")
#     optimizer = torch.optim.Adam(net.parameters())
#     print('Network initialized. Running a test batch.')
#     for _ in range(10):
#         with torch.set_grad_enabled(True):
#             batch = [torch.empty(1, 5, 512, 512).normal_().to("cuda"),torch.empty(1, 3, 512, 512).normal_().to("cuda"),torch.empty(1, 3, 512, 512).normal_().to("cuda")]
#             targets = torch.empty(1, 5, 512, 512).normal_().to("cuda")
#             out = net(*batch)
#             loss = criterion(out, targets)
#             loss.backward()
#             optimizer.step()
#         print(out.shape)
