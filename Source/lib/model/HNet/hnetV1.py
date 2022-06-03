
from collections import OrderedDict

import torch
import torch.nn as nn


class HNet_V1(nn.Module):

    def __init__(self, in_channels=[11,3,3],features=40,p=[0.01,0.01,0.01,0.01]):
        super(HNet_V1, self).__init__()
        self.dropout_1= nn.Dropout2d(p=p[0])
        self.dropout_2= nn.Dropout2d(p=p[1])
        self.dropout_3= nn.Dropout2d(p=p[2])
        self.dropout_4= nn.Dropout2d(p=p[3])
        self.dropout_b= nn.Dropout2d(p=p[3])

        #prev_heat_map encoder
        self.encoderA1 = self._block(in_channels[0], features, name="encA1")
        self.poolA1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderA2 = self._block(features, features * 2, name="encA2")
        self.poolA2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderA3 = self._block(features * 2, features * 4, name="encA3")
        self.poolA3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderA4 = self._block(features * 4, features * 8, name="encA4")
        self.poolA4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #pre_img encoder
        self.encoderB1 = self._block(in_channels[1], features, name="encB1")
        self.poolB1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderB2 = self._block(features, features * 2, name="encB2")
        self.poolB2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderB3 = self._block(features * 2, features * 4, name="encB3")
        self.poolB3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderB4 = self._block(features * 4, features * 8, name="encB4")
        self.poolB4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #img encoder
        self.encoderC1 = self._block(in_channels[2], features, name="encC1")
        self.poolC1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderC2 = self._block(features, features * 2, name="encC2")
        self.poolC2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderC3 = self._block(features * 2, features * 4, name="encC3")
        self.poolC3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderC4 = self._block(features * 4, features * 8, name="encC4")
        self.poolC4 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        #decoder heat_map
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")


    def forward(self, pre_hm,pre_img,img):
        encA1 = self.encoderA1(pre_hm)
        encA1 = self.dropout_1(encA1)
        encA2 = self.encoderA2(self.poolA1(encA1))
        encA2 = self.dropout_2(encA2)
        encA3 = self.encoderA3(self.poolA2(encA2))
        encA3 = self.dropout_3(encA3)
        encA4 = self.encoderA4(self.poolA3(encA3))
        encA4 = self.dropout_4(encA4)

        encB1 = self.encoderB1(pre_img)
        encB2 = self.encoderB2(self.poolB1(encB1))
        encB3 = self.encoderB3(self.poolB2(encB2))
        encB4 = self.encoderB4(self.poolB3(encB3))

        encC1 = self.encoderC1(img)
        encC2 = self.encoderC2(self.poolC1(encC1))
        encC3 = self.encoderC3(self.poolC2(encC2))
        encC4 = self.encoderC4(self.poolC3(encC3))

        bottleneck = self.bottleneck(self.poolA4(encA4)+self.poolB4(encB4)+self.poolC4(encC4))
        bottleneck = self.dropout_b(bottleneck)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4,encA4+encB4+encC4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3,encA3+encB3+encC3),  dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2,encA2+encB2+encC2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1,encA1+encB1+encC1), dim=1)
        dec1 = self.decoder1(dec1)

        return dec1

    def load_from_five(path):
        pass
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
# if __name__=='__main__':
#     model=HNet_V1E()
#
#     pretrained_dict=torch.load("../../model_archive/modelS4.pth")
#
#     model_dict = model.state_dict()
#     print(model_dict.keys())
#     print(pretrained_dict.keys())
#     # 1. filter out unnecessary keys
#     pretrained_dictN={}
#
#     for k in pretrained_dict.keys():
#         print(k)
#         if ".conv." in k or "encA1conv1" in k:
#             pass
#         elif k[7:] in model_dict.keys():
#             print(k)
#             pretrained_dictN[k[7:]] = pretrained_dict[k]
#
#     # 2. overwrite entries in the existing state dict
#     model_dict.update(pretrained_dictN)
#     # 3. load the new state dict
#     model.load_state_dict(model_dict)
#     torch.save(model.state_dict(),"../../model_archive/modelS4E.pth")

if __name__=="__main__":
     y, x=432,704
     # simple test run
     net = HNet_V1().to("cuda")
     print(net)
     criterion = nn.MSELoss().to("cuda")
     optimizer = torch.optim.Adam(net.parameters())
     import time
     print('Network initialized. Running a test batch.')
     for _ in range(10):
         with torch.set_grad_enabled(True):
             batch = [torch.empty(1, 11, y, x).normal_().to("cuda"),torch.empty(1, 3, y, x).normal_().to("cuda"),torch.empty(1, 3,  y, x).normal_().to("cuda")]
             targets = torch.empty(1, 40,  y, x).normal_().to("cuda")
             st=time.time()
             out = net(*batch)
             et=time.time()
             print(1/(et-st))
             loss = criterion(out, targets)
             loss.backward()
             optimizer.step()
         print(out.shape)
