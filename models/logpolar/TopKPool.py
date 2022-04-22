from torch import nn

class TopKPool(nn.Module):
    def __init__(self, k, dim=-1):
        self.dim = dim
        self.k = k

    def forward(self, tens):
        return tens.topk(self.k, self.dim).values