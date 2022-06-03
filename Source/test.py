from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import json
import torch

from lib.setup import Setup
from lib.model.models import get_model
from lib.data.datasets import get_dataset
from lib.test.testers import get_tester
from lib.test.tools import mkdir,set_test_path,AverageMeter

def tester(setup):
    set_test_path(setup)
    torch.manual_seed(setup.seed)
    torch.backends.cudnn.benchmark = not setup.not_cuda_benchmark


    print("\nInit dataset: ",end="")
    dataset = get_dataset(setup.dataset_tag)
    device_str='cuda' if setup.gpu_list[0] >= 0 else 'cpu'
    setup.device = torch.device(device_str)
    print("Done")

    print('Create model: ',end="")
    # model object with setup initialization
    model = get_model(setup.model_tag)(setup)

    print("Done")



    print("Test configuration: ",end="")
    #setup
    #get trainer for training
    Tester=get_tester(setup.test_tag)
    tester=Tester(setup,model,dataset)


    print("Done")
    print("Using: ",device_str)
    print(f"\nTest:\n{setup.test_id}")

    tester.run_all()


if __name__ == '__main__':
    setup=Setup().parse(["--task","mot","--purpose","test"])
    tester(setup)
