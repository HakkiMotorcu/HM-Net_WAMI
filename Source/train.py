from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import json
import torch

from lib.setup import Setup
from lib.loss.losses import get_loss
from lib.model.models import get_model
from lib.data.datasets import get_dataset
from lib.train.trainers import get_trainer
from lib.train.tools import mkdir,set_exp_path,set_exp
from lib.logger import Logger

def trainer(setup):

    set_exp(setup)

    torch.backends.cudnn.benchmark = not setup.not_cuda_benchmark
    logger = Logger(setup)
    torch.cuda.empty_cache()
    print("\nInit dataset: ",end="")
    dataset = get_dataset(setup.dataset_tag)(setup=setup)
    device_str='cuda' if setup.gpu_list[0] >= 0 else 'cpu'
    setup.device = torch.device(device_str)
    print("Done")
    print("Init data pipeline:",end="")
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=setup.batch_size,
        shuffle=True,
        num_workers=setup.num_workers,
        pin_memory= setup.pin_memory,
        drop_last=True
    )
    if setup.is_validate:
        val_dataset = get_dataset(setup.dataset_tag)(setup=setup,ann_path=setup.val_ann_path,root_path=setup.val_seq_path,call_purp="val")
        val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=setup.batch_size,
                        shuffle=True,
                        num_workers=setup.num_workers,
                        pin_memory= setup.pin_memory,
                        drop_last=True
                    )
    print("Done")
    print('Create model: ',end="")
    torch.manual_seed(setup.seed)
    # model object with setup initialization
    model = get_model(setup.model_tag)(setup)

    # loss function object with setup initialization
    loss = get_loss(setup.loss_tag,setup=setup)

    # optimizer init and merge with loss params
    optimizer = torch.optim.Adam(model.parameters(), setup.lr)

    print("Done")



    print("Training configuration: ",end="")
    #setup
    #get trainer for training
    Trainer=get_trainer(setup.train_tag)
    trainer=Trainer(setup,model,loss,optimizer)

    start_epoch=trainer.temp if trainer.is_load else 0
    print("Done")
    print("Using: ",device_str)
    print("\nTraining:")

    if setup.use_logger:
        for current_epoch in range(start_epoch,start_epoch+setup.epochs+1):
            logger.write('epoch: {} |'.format(current_epoch))

            ret, results = trainer.run_epoch("train",current_epoch,train_loader)

            for k, v in ret.items():
                logger.scalar_summary('train_{}'.format(k), v, current_epoch)
                logger.write('{} {:8f} | '.format(k, v))
            logger.write('\n')
            if setup.is_validate:
                with torch.no_grad():
                    ret, results = trainer.run_epoch("val",current_epoch,val_loader)
        logger.close()
    else:
        for current_epoch in range(start_epoch,start_epoch+setup.epochs+1):
            ret, results = trainer.run_epoch("train",current_epoch,train_loader)


if __name__ == '__main__':
    setup=Setup().parse([])
    setup.accept_same_id=True
    trainer(setup)
