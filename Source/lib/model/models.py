
from .hnet import HNet
from .hnetX import HNetX
from torch import nn

models={
            "hnet":HNet,
            "hnetx":HNetX,
}

def get_model(tag,default="hnet"):
    tag=tag.lower().replace("_","")
    if isinstance(tag,str) and tag in models.keys():
        return models[tag]
    else:
        return models[tag.lower().replace("_","")]
