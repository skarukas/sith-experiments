from torch.utils.data import Dataset
import torch
import numpy as np

import os
from os import path
import glob
import sys
from math import log2
from datetime import datetime
from scipy.stats import zscore

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader
from audiotsm.io.array import ArrayWriter
import librosa
import matplotlib.pyplot as plt
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio


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


class FileDataset(Dataset):
    """
    Expects that dir contains a bunch of "torch.saved" files 
        in subdirectories.
    """
    def __init__(self, dir, device='cpu'):
        self.device = device
        gb_path = path.join(glob.escape(dir), "**/*")
        print(f"Using glob '{gb_path}'...", end=" ")

        gb = glob.glob(gb_path, recursive=True)
        self.files = [f for f in gb if path.isfile(f)]
        print(f"found {len(self.files)} files")

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            return torch.load(f, map_location=self.device)

    def __len__(self):
        return len(self.files)


class StretchAudioDataset(Dataset):
    """
    Expects that root_dir contains a bunch of wav files in directories 
        named by class. 
    Stretches the audio then applies a constant-Q transform and normalization.
    Not very efficient and maybe should only be used during testing.
    """
    def __init__(self, root_dir: str, speed: float, label_to_idx: dict, 
            norm: str='minmax', cqparams: dict={}, device='cpu'):
        self.device = device
        gb_path = path.join(glob.escape(root_dir), "**/*")
        print(f"Using glob '{gb_path}'...", end=" ")
        gb = glob.glob(gb_path, recursive=True)
        self.files = [f for f in gb if path.isfile(f)]
        self.num_channels = 1
        self.tsm = phasevocoder(self.num_channels, speed=speed)
        self.label_to_idx = label_to_idx
        self.cqparams = cqparams
        self.norm = norm
        print(f"found {len(self.files)} files")

    def __getitem__(self, idx):
        fname = self.files[idx]
        with WavReader(fname) as reader:
            writer = ArrayWriter(self.num_channels)
            self.tsm.run(reader, writer)
            sr = reader.samplerate
            stretched = writer.data
        label = fname.split("/")[-2]
        label_idx = self.label_to_idx[label]
        X = constant_q(stretched, sr=sr, **self.cqparams)
        X = normalize(X, self.norm)
        X = X.to(self.device)
        return (X, label_idx)

    def __len__(self):
        return len(self.files)


# modified version of 
#.  https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, root_dir, subset: str = None):
        super().__init__(root_dir, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(map(os.path.abspath, excludes))
            self._walker = [w for w in self._walker if os.path.abspath(w) not in excludes]
    
    def __getitem__(self, idx):
      fname = self._walker[idx]
      x = torchaudio.load(fname)[0]
      split = fname.split("/")
      word = split[-2]
      id = split[-1].rstrip(".wav")
      return (x, word, id)


class SCStretch(SPEECHCOMMANDS):
    def __init__(self, subset: str, root_dir: str, speed: float, label_to_idx: dict, 
            norm: str='minmax', cqparams: dict={}, device='cpu'):
        super().__init__(root_dir, download=False)

        def load_list(filename):
            filepath = path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self.files = load_list("validation_list.txt")
        elif subset == "testing":
            self.files = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(map(path.abspath, excludes))
            self.files = [w for w in self._walker if path.abspath(w) not in excludes]
        
        self.device = device
        self.num_channels = 1
        self.tsm = phasevocoder(self.num_channels, speed=speed)
        self.label_to_idx = label_to_idx
        self.cqparams = cqparams
        self.norm = norm
        print(f"found {len(self.files)} files")
    

    def __getitem__(self, idx):
        fname = self.files[idx]
        with WavReader(fname) as reader:
            writer = ArrayWriter(self.num_channels)
            self.tsm.run(reader, writer)
            sr = reader.samplerate
            stretched = writer.data[0]
        label = fname.split("/")[-2]
        label_idx = self.label_to_idx[label]
        X = constant_q(stretched, sr=sr, **self.cqparams)
        X = normalize(X, self.norm)
        X = X.to(self.device)
        return (X, label_idx)


    def __len__(self):
        return len(self.files)


## audio methods
def normalize(X, method='minmax'):
    if method is None:
        return X
    elif method == "zscore":
        return zscore(X, axis=1)
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



if __name__ == "__main__":
    # test stretch dataset
    label_to_idx = {
        "one": 0,
        "two": 1,
    }
    st = StretchAudioDataset("data/wav_dataset", 1, label_to_idx=label_to_idx)
    plt.imshow(st[0][0][0], aspect="auto")
    plt.show()