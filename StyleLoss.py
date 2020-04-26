import torch.nn as nn
import torch.nn.functional as F
import torch


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
        self.loss = None

    def forward(self, inp):
        G = self.gram_matrix(inp)
        self.loss = F.mse_loss(G, self.target)
        return inp

    def gram_matrix(self, inp):
        a, b, c, d = inp.size()
        features = inp.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)
