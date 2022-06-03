import torch
import torch.nn as nn

#https://github.com/mathiaszinnen/focal_loss_torch/blob/main/focal_loss/focal_loss.py
def _sigmoid(x):
    y = torch.clamp(x, min=1e-4, max=1-1e-4)
    return y

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target,d=0):
        x = _sigmoid(x)
        p_t = torch.where(target>d,  x, 1-x)
        fl = - 1 * torch.pow(1 - p_t, self.gamma) * torch.log(p_t)
        fl = torch.where(target>d, fl * self.alpha, fl)
        return self._reduce(fl)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x

 
