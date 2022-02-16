from torch.utils.data import Dataset
import torch
import numpy as np
import random

import os
from os import path
import sys
import glob

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader
from audiotsm.io.array import ArrayWriter, ArrayReader
import scipy.io.wavfile as wavfile
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio

from morlet import phase_pow_multi
from scipy import signal
from util import constant_q, normalize


MAX_INT16 = 2**15
DEFAULT_SR = 16000


class SCStretch(SPEECHCOMMANDS):
    def __init__(self, subset: str, root_dir: str, speed: float, transform_params: dict, device='cpu'):
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
        self.speed = speed
        self.transform_params = transform_params
        print(f"found {len(self.files)} files")
    

    def __getitem__(self, idx):
        fname = self.files[idx]
        #x, sr = torchaudio.load(fname)
        sr, x = wavfile.read(fname)
        maxl = self.transform_params['maxl']
        x = np.pad(x, (int(np.floor((maxl - x.shape[0])/2)),
                int(np.ceil((maxl - x.shape[0])/2))), 'constant')

        stretched = stretch_audio(x, self.speed)

        # extract features
        label, id = fname.split("/")[-2:]
        id = id.rstrip(".wav")
        label_idx = self.transform_params['label_to_idx'][label]
        X = transform(stretched, self.transform_params, sr)
        X = X.to(self.device)

        return (X, label, label_idx, id)


    def __len__(self):
        return len(self.files)

SHUFFLE_SEED = 11111


class StretchedAudioMNIST(Dataset):
    """
    Expects that root_dir contains a bunch of wav files that have the
         class as the first char of the filename
    Has the same output format as StretchSC
    """
    def __init__(self, subset: str, root_dir: str, speed: float, transform_params: dict, device='cpu', split=(0.7, 0.15, 0.15)):
        self.device = device
        gb_path = path.join(glob.escape(root_dir), "**/*")
        print(f"Using glob '{gb_path}'...", end=" ")
        gb = glob.glob(gb_path, recursive=True)
        
        allfiles = [f for f in gb if path.isfile(f) and f.endswith(".wav")]
        # apply fixed random tr/te/val split
        random.seed(SHUFFLE_SEED)
        random.shuffle(allfiles)
        nfiles = len(allfiles)
        s0 = int(split[0] * nfiles)
        s1 = int((split[0]+split[1]) * nfiles)
        splitfiles = allfiles[:s0], allfiles[s0:s1], allfiles[s1:]
        print([*map(len, splitfiles)])
        idx = ["training", "validation", "testing"].index(subset)

        self.files = splitfiles[idx]
        self.transform_params = transform_params
        self.speed = speed
        print(f"found {len(self.files)} files")


    def __getitem__(self, idx):
        fname = self.files[idx]
        sr, x = wavfile.read(fname)
        maxl = self.transform_params['maxl']
        x = np.pad(x, (int(np.floor((maxl - x.shape[0])/2)),
                int(np.ceil((maxl - x.shape[0])/2))), 'constant')

        stretched = stretch_audio(x, self.speed)

        # extract features
        short_name = fname.split("/")[-1]
        label = short_name.split("_")[0]
        label_idx = self.transform_params['label_to_idx'][label]

        id = short_name.rstrip(".wav")
        
        X = transform(stretched, self.transform_params, sr)
        X = X.to(self.device)

        return (X, label, label_idx, id)

    def __len__(self):
        return len(self.files)


def stretch_audio(x, speed):
    factor = 1 / speed
    reader = ArrayReader(x[np.newaxis])
    writer = ArrayWriter(1)
    tsm = phasevocoder(1, speed=speed)
    tsm.run(reader, writer)
    stretched = writer.data[0].astype(np.int16)*factor
    return stretched


def transform(x, transform_params, sr=DEFAULT_SR):
    """
    Code adapted from AudioMNIST notebook
    """

    # pad audio data (not strictly necessary, 
    #   but helps short files not run into issues during morlet transform)

    if transform_params['method'] == "morlet":
        # compute morlet transform
        X = phase_pow_multi(
            transform_params['morlet_freqs'], x, samplerates=sr, widths=5,
            to_return='power', time_axis=-1,
            conv_dtype=np.complex64, freq_name='freqs'
        )
    else:
        # constant-Q transform
        X = constant_q(
            x, sr=sr, 
            fmin=transform_params['fmin'], fmax=transform_params['fmax'], 
            bins=transform_params['nbins'], hop_length=transform_params['hop_length']
        )[0]
    # resample 2D features
    resample_factor = transform_params['resample_factor']
    if resample_factor is not None:
        X = signal.resample(X, X.shape[1]//resample_factor, axis=1)
    
    X_norm = normalize(X, transform_params['norm_method'])
    X_norm = torch.tensor(X_norm)
    X_norm[~X_norm.isfinite()] = 0

    if len(X_norm.shape) == 2:
        X_norm = X_norm[np.newaxis]
    return X_norm