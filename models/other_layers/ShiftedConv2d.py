import torch
from torch import nn
import random


class ShiftedConv2d(nn.Module):
  def __init__(self, shifts, filters, stride=1, padding=0):
    """
    Note: Right now this happens over the input channels independently (e.g. output has C_in*C_out channels)
    Perform a convolution on pixels a certain distance away. This is equivalent 
        to convolving with a larger kernel such that the shift is the 
        distance from the "virtual" kernel's top left corner to the 
        active part of the kernel.

        For example,
        shift : (1, 4)
        kernel :
            K K K
            K K K

        is equivalent to convolution with the following virtual kernel:
            0 0 0 0 0 0 0
            0 0 0 0 K K K
            0 0 0 0 K K K
      
    """
    super().__init__()
    
    # assume shifts and filters are fixed
    # shifts are the distance from the kernel's top left corner 
    #   to the "actual" top left corner of the larger kernel
    self.shifts = torch.tensor(shifts)
    while len(filters.shape) < 4:
      filters = filters[None]
    self.filters = filters
    self.device = self.filters.device
    self.stride = stride

    # turn padding into 4-tuple
    assert type(padding) != str, "Padding type not supported."

    if type(padding) == int:
      padding = [padding]

    while len(padding) != 4:
      p = []
      for n in padding:
        p.append(n)
        p.append(n)
      padding = p
    
    padding[1] += filters[0].shape[-1] - 1 
    padding[3] += filters[0].shape[-2] - 1 

    self.pad_val = 0
    self.pad = nn.ConstantPad2d(tuple(padding), self.pad_val)
  

  def forward(self, tens, shift, filter_block):
    in_device = tens.device
    tens = tens.to(self.device)
    shape = tens.shape
    tens = tens.reshape((-1, 1, *shape[2:]))
    tens_padded = self.pad(tens)

    # shift tensor
    tens_padded = tens_padded.repeat(1, len(self.shifts), 1, 1)
    tens_padded = roll_multiple(tens_padded, self.shifts[..., 0], dim=-2, idx_dim=1, cyclic=False)
    tens_padded = roll_multiple(tens_padded, self.shifts[..., 1], dim=-1, idx_dim=1, cyclic=False)

    out = torch.conv2d(tens_padded, self.filters, stride=self.stride, groups=len(self.shifts))

    return out.reshape((shape[0], -1, *shape[2:])).to(in_device)


def roll_multiple(tens, idx, dim, idx_dim=None, cyclic=True):
    idx_dim = dim - 1 if idx_dim is None else idx_dim
    n = tens.shape[dim]
    rng_s = [1] * len(tens.shape)
    rng_s[dim] = n
    idx_s = [1] * len(tens.shape)
    idx_s[idx_dim] = len(idx)
    idx = idx.reshape(tuple(idx_s))

    rng = torch.arange(n).to(idx.device).reshape(tuple(rng_s))
    idx = rng - idx
    idx, _ = torch.broadcast_tensors(idx, tens)
    rolled = tens.gather(dim, idx % n)
    if cyclic:
        return rolled
    else:
        rolled[idx < 0] = 0
        rolled[idx >= n] = 0
        return rolled


if __name__ == "__main__":

    def mse(t1, t2):
        return ((t1 - t2)**2).mean().item()

    for stride in range(1, 3):
        for padding in range(1, 2):
            for kernel_size in range(3, 10):
                internal_size = (random.randint(2, kernel_size), random.randint(2, kernel_size))
                for i in range(kernel_size-internal_size[0]):
                    for j in range(kernel_size-internal_size[1]):
                        shift = (i, j)
                        x = torch.randn((20, 20))[None][None]
                        internal_kernel = torch.randn(internal_size)
                        kernel = torch.zeros((kernel_size, kernel_size)).float()[None, None]
                        print(kernel.shape, internal_kernel.shape, x.shape)
                        slices = [slice(shift[k], shift[k]+internal_kernel.shape[k]) for k in range(2)]
                        kernel[..., slices[0], slices[1]] = internal_kernel

                        print(kernel.shape, internal_kernel.shape, x.shape, i, j)
                        out = torch.conv2d(x, kernel, stride=stride, padding=padding)
                        sc = ShiftedConv2d([shift], internal_kernel[None], stride=stride, padding=padding)
                        out_sc = sc(x)
                        assert mse(out, out_sc[..., :out.shape[-2], :out.shape[-1]]) == 0