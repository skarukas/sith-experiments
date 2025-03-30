# relative_encoding.py - PyTorch layers for encoding vectors (x_1, x_2, ..., x_d)
#   such that f(x_1, x_2, ..., x_d) = f(x_1+k, x_2+k, ..., x_d+k) for all k
# Author: Stephen Karukas

import torch
from torch import nn
from math import pi
import random

EPS = 1e-8

class RelativeEncoding(nn.Module):
    def __init__(self, dim=-1, method="exponential", temperature=-1):
        """
        Create an encoding of the input vector such that:
        - f(x_1, x_2, ...) = f(x_1+k, x_2+k, ...) for all k
        And apply a set of weights to each output.

        Given a set of indices and weights that represent a distribution
        of some value over some domain, create a deterministic encoding
        that is invariant to translation over the domain.
        Compared to pooling, this similarly imposes strict invariance, but
        it retains the full distribution of the input data instead of some
        metric (average or max).
        """
        super().__init__()
        # method = exponential | linear
        self.method = method
        self.dim = dim
        self.temperature = temperature
    

    def forward(self, tens, weights=torch.tensor(1)):
        """
        tens: torch.Tensor
            - The tensor over which to perform the relative encoding. The domain of the distribution.
        weights: torch.Tensor
            - A weight matrix (broadcastable with tens) that stores the values of the distribution.
               The weight `w_i` for a given `x_i` will make `x_i` have less influence
               over the output. `w_i = 0` means that the value of `x_i`
               will have no effect on the output.
        """
        weights = weights.expand(tens.shape).to(tens.device)

        # Both methods satisfy the invariance property.
        if self.method == "linear":
            # A simple way to encode relative weight: subtract the mean.
            # x_i - w_mean(x, w)
            y = tens - (weights*tens).sum(self.dim, keepdim=True) / (EPS + weights.sum(self.dim, keepdim=True))
        else:
            # Modified softmax:
            # w_i*exp(x_i) / (sum(w*exp(x)) - w_i*exp(x_i))
            xp = weights * torch.exp(self.temperature*tens)
            sm = xp.sum(self.dim, keepdim=True)
            y = xp / (EPS + sm - xp)
        return y


class PeriodicRelativeEncoding(nn.Module):
    def __init__(self, period, dim=-1, real=False):
        """
        Create an encoding of the input vector such that:
        - `f(x_1, x_2, ...) = f(x_1+k, x_2+k, ...)` for all `k`
        - `f(x_1, x_2, ...) = f(x_1+a_1*m, x_2+a_2*m, ...)`
            for integer `(a_1, a_2, ...)` and period `m`.
        And apply a set of weights to each output.

        Given a set of indices and weights that represent a distribution
        of some value over a periodic axis (ex: a clock face), create a deterministic
        encoding that is invariant to rotation (= periodic translation).
        Compared to pooling, this similarly imposes strict invariance, but
        it retains the full distribution of the input data instead of some
        metric (average or max).
        
        - If `real` is `False`, the output will be a tensor where 
          `out.shape[dim] = 2*d`, with the real part of the function 
          as the first `d` elements and the imaginary part as the 
          next `d` elements. `d=inp.shape[dim]`
        - If `real` is `True`, only the real part will be returned.
        
        Note: the complex representation satisfies 
            `f(x) = f(x') iff x' = x + k + a*m` (integer vector `a`, real scalar `k`),
          but the real version does not. No smooth real periodic function 
          is bijective in [0, m) -- one output will map to multiple inputs, so
          the network would have to learn to disambiguate the output.
        """
        super().__init__()
        self.dim = dim
        self.real = real
        self.m = period
    

    def forward(self, tens, weights=torch.tensor(1)):
        """
        tens: torch.Tensor
            - The tensor over which to perform the relative encoding. The domain of the distribution.
        weights: torch.Tensor
            - A weight matrix (broadcastable with tens) that stores the values of the distribution.
               The weight `w_i` for a given `x_i` will make `x_i` have less influence
               over the output. `w_i = 0` means that the value of `x_i`
               will have no effect on the output.
        """
        weights = weights.expand(tens.shape).to(tens.device)
        # Modified complex softmax:
        # w_i*cexp(x_i) / (sum(w*cexp(x)) - w_i*cexp(x_i))
        # Where cexp(y) is a complex phasor with angle y
        xp = weights*torch.exp(-2j*pi*tens / self.m)
        sm = xp.sum(self.dim, keepdim=True)
        y = xp / (EPS + sm - xp)
        
        return y.real if self.real else (y.real, y.imag)


if __name__ == "__main__":
    # Test invariance of non-periodic layer
    eps = 1e-4
    x = torch.randn(10, 5, 6, 15)

    for method in ("linear", "exponential"):
        for dim in range(len(x.shape)):
            encoder = RelativeEncoding(dim=dim, method=method)
            out = encoder(x)
            assert out.shape == x.shape

            # add random number to all
            out1 = encoder(x + torch.randn(1))
            err = ((out1 - out)**2).sum()
            assert err < eps, err

    # Test invariance of periodic layer
    for real in (True, False):
        for dim in range(len(x.shape)):
            m = random.randint(5, 20)
            encoder = PeriodicRelativeEncoding(m, dim=dim, real=real)
            out = encoder(x)
            z = 2 - real
            assert out.shape[dim] == x.shape[dim]*z

            # add random number to all (k)
            out1 = encoder(x + torch.randn(1))
            err = ((out1 - out)**2).sum()
            assert err < eps, err

            # add random multiple of period to each (a_i * m)
            shape = [1] * len(x.shape)
            shape[dim] = x.shape[dim]
            shape = tuple(shape)
            a = torch.randint(0, 10, shape)
            out2 = encoder(x + a*m)
            err = ((out2 - out)**2).sum()
            assert err < eps, err
