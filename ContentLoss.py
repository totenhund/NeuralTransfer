import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = None

    def forward(self, inp):
        self.loss = F.mse_loss(inp, self.target)
        return inp
