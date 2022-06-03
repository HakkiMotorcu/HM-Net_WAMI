from .UNet.unet import UNet
from .HNet.hnetR0 import HNet_R0
from .HNet.hnetRL0 import HNet_RL0
from .HNet.hnetU0 import HNet_LTE as HNet_U0
from .HNet.hnetU1 import HNet_U1
from .HNet.hnetV0 import HNet_V0
from .HNet.hnetV1 import HNet_V1
from .HNet.hnetV2 import HNet_V2
from .HNet.hnetV3 import HNet_V3
from .HNet.hnetV4 import HNet_V4
from torch import nn

backbones={"unet":UNet,
        "hnetr0":HNet_R0,
        "hnetrl0":HNet_RL0,
        "hnetu0":HNet_U0,
        "hnetu1":HNet_U1,
        "hnetv0":HNet_V0,
        "hnetv1":HNet_V1,
        "hnetv2":HNet_V2,
        "hnetv3":HNet_V3,
        "hnetv4":HNet_V4
}

def get_backbone(tag,default="hnetv3"):
    tag=tag.lower().replace("_","")
    if isinstance(tag,str) and tag in backbones.keys():
        return backbones[tag]
    else:
        return backbones[tag.lower().replace("_","")]
