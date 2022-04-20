import torch
from torch import nn
import random

class ShiftedConv2d(nn.Module):
  def __init__(self, shifts, filters, stride=1, padding=0):
    """
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
    self.shifts = shifts
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

    self.pad_val = 0
    self.pad = nn.ConstantPad2d(tuple(padding), self.pad_val)
  

  def __forward_single_block(self, tens, shift, filter_block):
    shift = [-s for s in shift]
    while len(filter_block.shape) < 4:
      filter_block = filter_block[None]
    # filters : out_channels * in_channels * kH * kW
    # tens : batch * in_channels * iH * iW
    tens_padded = self.pad(tens)

    # shift tensor and fill empty spots with zeros
    tens_padded = tens_padded.roll(shifts=shift, dims=(-2, -1))
    start = [0, 0]
    stop = [0, 0]
    for i in range(2):
      start[i] = 0 if shift[i] >= 0 else tens.shape[i-2]+shift[i]
      stop[i] = start[i] + abs(shift[i])

    tens_padded[..., start[0]:stop[0], :] = self.pad_val
    tens_padded[..., :, start[1]:stop[1]] = self.pad_val
    return torch.conv2d(tens_padded, filter_block, stride=self.stride)


  def forward(self, tens):
    # tens = Batch * in_channels * iH * iW
    in_device = tens.device
    tens = tens.to(self.device)
    return torch.cat(
        [self.__forward_single_block(tens, s, filt) 
          for s, filt in zip(self.shifts, self.filters)], 
        dim=1
    ).to(in_device)



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