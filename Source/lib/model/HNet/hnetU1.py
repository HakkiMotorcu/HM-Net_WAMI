
from collections import OrderedDict

import torch
import torch.nn as nn


class HNet_U1(nn.Module):

    def __init__(self, in_channels=(5,3,3), features=32,p=[0,0,0,0,0]):
        super(HNet_U1, self).__init__()
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

        features*=3
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

        bottleneck = self.bottleneck(torch.cat((self.poolA4(encA4),self.poolB4(encB4),self.poolC4(encC4)), dim=1).contiguous())
        bottleneck = self.dropout_b(bottleneck)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4,torch.cat((encA4,encB4,encC4), dim=1)), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3,torch.cat((encA3,encB3,encC3), dim=1)), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, torch.cat((encA2,encB2,encC2), dim=1)), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, torch.cat((encA1,encB1,encC1), dim=1)), dim=1)
        dec1 = self.decoder1(dec1)

        return dec1


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
