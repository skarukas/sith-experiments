from torch.utils.data import Dataset
import torch
import numpy as np

import sys
from math import log2
from datetime import datetime
from scipy.stats import zscore

import librosa


class Average:
    """
    Keep running average of a series of observations
    """
    def __init__(self):
        self.n = self.sum = 0
    
    def update(self, num, n=1):
        self.n += n
        self.sum += num

    def get(self):
        return self.sum / self.n if self.n else 0


def collate_examples_list(data):
    """
    Collate tensors with different sequence lengths 
        by collecting them into a list
    """
    X = [t[0] for t in data]
    targets = torch.tensor([t[1] for t in data])
    return (X, targets)


def collate_examples_pad(data):
    """
    Collate tensors with different sequence lengths by padding
        the beginning with zeros
    """
    inp, targets = zip(*data)

    # zero-pad input at beginning 
    batch_size = len(inp)
    lengths = [tens.shape[-1] for tens in inp]
    max_len = max(lengths)
    shape = (batch_size, *inp[0].shape[:-1], max_len)
    TensorType = torch.cuda.FloatTensor if 'cuda' in str(inp[0].device) else torch.FloatTensor
    
    padded = TensorType(*shape).fill_(0)
    for i in range(batch_size):
        l = lengths[i]
        padded[i, ..., -l:] = inp[i]
    targets = torch.tensor(targets, device=inp[0].device)
    return padded, targets


def curr_time_str():
    now = datetime.now().replace(microsecond=0)
    return now.isoformat(sep='_')


## audio methods
def normalize(X, method='minmax'):
    if method is None:
        return X
    elif method == "zscore":
        return zscore(X, axis=-1)
    else:
        # minmax
        d = X.max() - X.min()
        eps = 1e-4
        return (X - X.min()) / (d + eps)


def constant_q(x, sr=16000, fmin=100, fmax=6000, bins=50, hop_length=64):
    bins_per_octave = int(bins / log2(fmax/fmin) + 0.5)
    x = x.flatten()
    if len(x) == 0:
        x = np.zeros(128)
    X = librosa.cqt(
        x, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=bins, 
        bins_per_octave=bins_per_octave, pad_mode='constant'
    )
    X = torch.tensor(X[np.newaxis]).abs()
    return X


# modified from https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, fname, terminal_out=sys.stdout):
        self.terminal = terminal_out
        self.log = open(fname, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass