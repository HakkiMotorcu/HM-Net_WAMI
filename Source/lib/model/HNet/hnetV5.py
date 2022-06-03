
from collections import OrderedDict

import torch
import torch.nn as nn
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    def __init__(self,in_features,out_features,hidden_features,stride=1,padding=1,name=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=stride,padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        self.conv3 =  nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        identity = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out+identity)

        out = self.conv3(out)
        out = self.bn3(out)


        out = self.relu(out)

        return out


class HNet_V5 (nn.Module):
    def __init__(self, in_channels=[11,3,3],features=32,p=[0.01,0.01,0.01,0.01]):
        super(HNet_V5 , self).__init__()
        inC=[1,2,4,8,16]
        hhC=[1,2,4,8]
        self._block=Bottleneck
        #prev_heat_map encoder
        self.convA=nn.Conv2d(in_channels[0], features*inC[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.encoderA1 = self._block(features*inC[0], features*inC[1],features*hhC[0]  )
        self.poolA1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderA2 = self._block(features*inC[1], features * inC[2], features *hhC[1] )
        self.poolA2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderA3 = self._block(features * inC[2], features * inC[3], features * hhC[2] )
        self.poolA3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderA4 = self._block(features * inC[3], features * inC[4], features * hhC[3])
        self.poolA4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #pre_img encoder
        self.convB=nn.Conv2d(in_channels[1], features*inC[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.encoderB1 = self._block(features*inC[0], features*inC[1],features*hhC[0]  )
        self.poolB1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderB2 = self._block(features*inC[1], features * inC[2], features *hhC[1] )
        self.poolB2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderB3 = self._block(features * inC[2], features * inC[3], features * hhC[2] )
        self.poolB3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderB4 = self._block(features * inC[3], features * inC[4], features * hhC[3])
        self.poolB4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #img encoder
        self.convC=nn.Conv2d(in_channels[2], features*inC[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.encoderC1 = self._block(features*inC[0], features*inC[1],features*hhC[0]  )
        self.poolC1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderC2 = self._block(features*inC[1], features * inC[2], features *hhC[1] )
        self.poolC2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderC3 = self._block(features * inC[2], features * inC[3], features * hhC[2] )
        self.poolC3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderC4 = self._block(features * inC[3], features * inC[4], features * hhC[3])
        self.poolC4 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = self._block(features * inC[4], features * inC[4], features * hhC[3], name="bottleneck")

        #decoder heat_map
        self.upconv4 = nn.ConvTranspose2d(features * inC[4], features * inC[4], kernel_size=2, stride=2)
        self.decoder4 = self._block((features * inC[4]) * 2, features * inC[4],features * hhC[3] )
        self.upconv3 = nn.ConvTranspose2d(features *inC[4], features * inC[3], kernel_size=2, stride=2)
        self.decoder3 = self._block((features * inC[3]) * 2, features * inC[3], features * hhC[2] )
        self.upconv2 = nn.ConvTranspose2d(features * inC[3], features * inC[2], kernel_size=2, stride=2)
        self.decoder2 = self._block((features * inC[2]) * 2, features * inC[2], features * hhC[1] )
        self.upconv1 = nn.ConvTranspose2d(features * inC[2], features*inC[1], kernel_size=2, stride=2)
        self.decoder1 = self._block(features * inC[1]*2, features*inC[1], features*hhC[0] )


        self.encoderD1 = self._block(features*inC[1], features*inC[1],features*hhC[0])
        self.poolD1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderD2 = self._block(features*inC[1], features * inC[2], features *hhC[1] )
        self.poolD2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderD3 = self._block(features * inC[2], features * inC[3], features * hhC[2] )
        self.poolD3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoderD4 = self._block(features * inC[3], features * inC[4], features * hhC[3])
        self.poolD4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneckD = self._block(features * inC[4], features * inC[4], features *hhC[3], name="bottleneckD")
        #decoder heat_map
        self.upconvD4 = nn.ConvTranspose2d(features * inC[4], features * inC[4], kernel_size=2, stride=2)
        self.decoderD4 = self._block((features * inC[4]) * 2, features * inC[4],features * hhC[3] )
        self.upconvD3 = nn.ConvTranspose2d(features *inC[4], features * inC[3], kernel_size=2, stride=2)
        self.decoderD3 = self._block((features * inC[3]) * 2, features * inC[3], features * hhC[2] )
        self.upconvD2 = nn.ConvTranspose2d(features * inC[3], features * inC[2], kernel_size=2, stride=2)
        self.decoderD2 = self._block((features * inC[2]) * 2, features * inC[2], features * hhC[1] )
        self.upconvD1 = nn.ConvTranspose2d(features * inC[2], features*inC[1], kernel_size=2, stride=2)
        self.decoderD1 = self._block(features * inC[1]*2, features*inC[0], features*hhC[0] )



    def forward(self, pre_hm,pre_img,img):
        encA1 = self.convA(pre_hm)
        encA1 = self.encoderA1(encA1)
        encA2 = self.encoderA2(self.poolA1(encA1))
        encA3 = self.encoderA3(self.poolA2(encA2))
        encA4 = self.encoderA4(self.poolA3(encA3))

        encB1 = self.convB(pre_img)
        encB1 = self.encoderB1(encB1)
        encB2 = self.encoderB2(self.poolB1(encB1))
        encB3 = self.encoderB3(self.poolB2(encB2))
        encB4 = self.encoderB4(self.poolB3(encB3))

        encC1 = self.convC(img)
        encC1 = self.encoderC1(encC1)
        encC2 = self.encoderC2(self.poolC1(encC1))
        encC3 = self.encoderC3(self.poolC2(encC2))
        encC4 = self.encoderC4(self.poolC3(encC3))

        bottleneck = self.bottleneck(self.poolA4(encA4)+self.poolB4(encB4)+self.poolC4(encC4))


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

        encD1 = self.encoderD1(dec1)
        encD2 = self.encoderD2(self.poolD1(encD1))
        encD3 = self.encoderD3(self.poolD2(encD2))
        encD4 = self.encoderD4(self.poolD3(encD3))

        bottleneckD = self.bottleneckD(self.poolD4(encD4))

        decD4 = self.upconvD4(bottleneckD+bottleneck)
        decD4 = torch.cat((decD4,encD4+dec4), dim=1)
        decD4 = self.decoderD4(decD4)
        decD3 = self.upconvD3(decD4)
        decD3 = torch.cat((decD3,encD3+dec3),  dim=1)
        decD3 = self.decoderD3(decD3)
        decD2 = self.upconvD2(decD3)
        decD2 = torch.cat((decD2,encD2+dec2), dim=1)
        decD2 = self.decoderD2(decD2)
        decD1 = self.upconvD1(decD2)
        print(encD1.shape,dec1.shape)
        decD1 = torch.cat((decD1,encD1+dec1), dim=1)
        decD1 = self.decoderD1(decD1)
        return decD1

if __name__=="__main__":
    y, x=400,640
    # simple test run
    net = HNet_V5().to("cuda")
    print(net)
    criterion = nn.MSELoss().to("cuda")
    optimizer = torch.optim.Adam(net.parameters())
    print('Network initialized. Running a test batch.')
    for _ in range(10):
        with torch.set_grad_enabled(True):
            batch = [torch.empty(1, 11, y, x).normal_().to("cuda"),torch.empty(1, 3, y, x).normal_().to("cuda"),torch.empty(1, 3,  y, x).normal_().to("cuda")]
            targets = torch.empty(1, 32,  y, x).normal_().to("cuda")
            out = net(*batch)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        print(out.shape)
