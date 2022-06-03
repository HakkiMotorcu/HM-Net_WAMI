"""
_______________________________________________________________________________________________
* Project Name : HNet
*
* Version : 0.1
* Date : 05.03.2021
*
* Developer : Hakkı Motorcu
*
* Information :
*
*  Version 0.1	- (05.03.2021) Initial Version
*
* All Rights Reserved.
_______________________________________________________________________________________________
* Purpose:
*  Handling general data operations such as vis drone to coco transformation, and creating
*  static or dynamic datasets if needed.
_______________________________________________________________________________________________
"""
from torch import nn
from .all.loss_lib import TverskyLoss,ComboLoss,FocalLoss,DiceBCELoss,IoULoss,FocalTverskyLoss
from .all.ffl import FastFocalLoss
from .all.generic_loss import GenericLoss
from .all.fl import FocalLoss as fl
from .all.RegWeightedL1Loss import RegWeightedL1Loss

loss_functions={"mse":nn.MSELoss,
                "bce":nn.BCELoss ,
                "bcewithsigmoid":nn.BCEWithLogitsLoss ,
                "mae":nn.L1Loss ,
                "crossentropy":nn.CrossEntropyLoss ,
                "nll":nn.NLLLoss ,
                "tverskyloss":TverskyLoss ,
                "comboloss":ComboLoss ,
                "focalloss":fl(alpha=2, gamma=4),
                "dicebceloss":DiceBCELoss ,
                "iouloss":IoULoss ,
                "focaltverskyloss":FocalTverskyLoss ,
                "fastfocalloss":FastFocalLoss ,
                "genericloss":GenericLoss ,
                "RegWeightedL1Loss":RegWeightedL1Loss
}

def get_loss(tag=None,default="MSE",alpha=None,gamma=None,setup=""):
    tag=tag.lower().replace("_","")
    if isinstance(tag,str) and tag in loss_functions.keys():
        return loss_functions[tag] if setup=="" else loss_functions[tag](setup)
    else:
        print("\nUnidenttified loss function\nLoss is set to default MSE")
        return loss_functions[default.lower().replace("_","")]()
