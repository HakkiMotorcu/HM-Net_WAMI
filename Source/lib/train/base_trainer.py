from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
from .tools import ModelWithLoss,AverageMeter
import time
from progress.bar import Bar

class BaseTrainer(object):
    def __init__(self,setup,model,loss,optimizer):
        self.setup = setup
        self.model_with_loss = ModelWithLoss(model,loss)
        self.optimizer=optimizer
        self.is_load=False
        if self.setup.load_model!="":
            self.temp=self.load_model()
            self.is_load=True and not self.setup.start_from_zero
        self.set_env()

    def set_env(self):
        self.is_parallel=len(self.setup.gpu_list) > 1
        if self.is_parallel:
            self.model_with_loss = torch.nn.DataParallel(self.model_with_loss, device_ids=self.setup.gpu_list).to(self.setup.device)
        else:
            self.model_with_loss = self.model_with_loss.to(self.setup.device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=self.setup.device, non_blocking=True)

    def init_timers(self,data_loader,epoch):
        max_iter=len(data_loader)
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.setup.regression_head_dims}
        avg_loss_stats["tot"] = AverageMeter()
        bar = Bar(f'{self.setup.task}/{self.setup.exp_id} | Epoch: {epoch+1}', max=max_iter,check_tty=False, hide_cursor=False)
        return data_time, batch_time, avg_loss_stats, bar, max_iter

    def set_model(self,purpose):
        if purpose=="train":
            self.model_with_loss.train()
        else:
            self.model_with_loss.eval()
            torch.cuda.empty_cache()

    def run_epoch(self,purpose,epoch,data_loader):
        if self.is_load:
            epoch=self.temp
            self.is_load=False

        results = {}
        self.set_model(purpose)

        data_time, batch_time, avg_loss_stats, bar, max_iter=self.init_timers(data_loader,epoch)

        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            #if iter_id<2150:
            #    continue
            if iter_id >= max_iter:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device= self.setup.device, non_blocking=True)
            if purpose == 'train':
                self.optimizer.zero_grad()
            output, loss, loss_stats = self.model_with_loss(batch)
            
            loss = loss.mean()
            if purpose == 'train':
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = f' [{iter_id}/{max_iter}]|Tot: {bar.elapsed_td:} |ETA: {bar.eta_td:} '

            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['hm'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            Bar.suffix = Bar.suffix + f'|Data {data_time.val:.3f}s({data_time.avg:.3f}s)|Batch {batch_time.avg:.3f}s'


            if self.setup.verbose_period>0 and (iter_id % self.setup.verbose_period)+1==0:
                bar.next()
            else:
                bar.next()

            del output, loss, loss_stats, batch
            if purpose == 'train':
                if self.setup.is_save_model_batch and iter_id%self.setup.save_model_batch==0:
                    self.save_checkpoint(epoch,suffix="last")
            
        bar.finish()
        if purpose == 'train':
            if self.setup.save_checkpoint:
                if (epoch+1)%self.setup.save_period==0:
                    self.save_checkpoint(epoch,suffix=epoch)
            self.save_checkpoint(epoch,suffix="last")

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def save_checkpoint(self,epoch,suffix):
        model_dict=self.model_with_loss.module.model.state_dict() if isinstance(self.model_with_loss, torch.nn.DataParallel) else self.model_with_loss.model.state_dict()
        optimizer_dict=self.optimizer.state_dict()
        check={
            'epoch': epoch,
            'model': model_dict,
            'optimizer': optimizer_dict,
            }
        torch.save(check, os.path.join(self.setup.exp_root,self.setup.exp_id,"models",f"model_{suffix}.pth"))

    def load_model(self):
        checkpoint=torch.load(self.setup.load_model)
        self.model_with_loss.model.load_state_dict(checkpoint['model'])
        if self.setup.load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            #self.optimizer.param_groups[0]["lr"]=0.0002
            #print(self.optimizer.param_groups[0]["lr"])
        return checkpoint["epoch"]
