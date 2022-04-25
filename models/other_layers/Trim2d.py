import torch
from torch import nn

class Trim2d(nn.Module):
    def __init__(self, trim_size=1):
        """
        Trims ("un-pads") the last two dimensions of a tensor by a certain 
            amount off each side in order to reduce feature map size.
            If this would lead to a negative size, the operation does nothing.
        Param:
            - trim_size: int | tuple
                The maximum number of pixels to remove from each side of each
                dimension.
        """
        super().__init__()
        if type(trim_size) not in (list, tuple):
            self.trim_size = (trim_size, trim_size)
        else:
            self.trim_size = trim_size
        
        assert self.trim_size[0] >= 0
        assert self.trim_size[1] >= 0
    
    def forward(self, x):
        d1, d2 = x.shape[-2:]
        t1, t2 = self.trim_size
        if d1 > 2*t1:
            x = x[..., t1:(d1-t1), :]
        if d2 > 2*t2:
            x = x[..., :, t2:(d2-t2)]
        return x


if __name__ == "__main__":
    trim = Trim2d((1, 3))
    x = torch.zeros((1, 1, 8, 7))
    x_trimmed = trim(x)
    assert x_trimmed.shape == (1, 1, 6, 1)

    x = torch.zeros((1, 1, 3, 6))
    x_trimmed = trim(x)
    assert x_trimmed.shape == (1, 1, 1, 6)