import numpy as np
import torch
from functools import reduce
from operator import mul

TWO_PI = 2*np.pi
IDENTITY = lambda x: x

def unsqueeze_except(X, n_dim, dim=0):
    """
    Make X start at the dim'th dimension in an n_dim tensor.
    
    Example:
        unsqueeze_except(X, 4, 1) with X.shape = (128,) -> (1, 128, 1 1)
    """
    dim = (dim + n_dim) % n_dim
    assert n_dim - dim >= len(X.shape)
    res = X
    for i in range(dim):
        res = res[np.newaxis]
    for i in range(n_dim-dim-len(X.shape)):
        res = res[..., np.newaxis]
    return res


def safe_log(X):
    # only take the log of nonzero numbers
    out = torch.zeros_like(X)
    out[X > 0] = torch.log(X[X > 0])
    return out


def pad_periodic(x, padding, dim=0):
    # pad the end with values from the beginning
    if padding <= 0:
        return x
    else:
        return torch.cat((x, x[..., :padding]), dim=dim)


def prod(iterable):
    return reduce(mul, iterable, 1)