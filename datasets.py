from torch.utils.data import Dataset
import torch
import numpy as np

import os
from os import path
import sys

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader
from audiotsm.io.array import ArrayWriter, ArrayReader
import scipy.io.wavfile as wavfile
from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio

from morlet import phase_pow_multi
from scipy import signal
from util import constant_q, normalize


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
        self.num_channels = 1
        self.speed = speed
        self.tsm = phasevocoder(self.num_channels, speed=speed)
        self.transform_params = transform_params
        print(f"found {len(self.files)} files")
    

    def __getitem__(self, idx):
        fname = self.files[idx]
        #x, sr = torchaudio.load(fname)
        sr, x = wavfile.read(fname)
        maxl = self.transform_params['maxl']
        x = np.pad(x, (int(np.floor((maxl - x.shape[0])/2)),
                int(np.ceil((maxl - x.shape[0])/2))), 'constant')

        factor = 1 / self.speed
        reader = ArrayReader(x[np.newaxis])
        writer = ArrayWriter(self.num_channels)
        self.tsm.run(reader, writer)

        stretched = writer.data[0].astype(np.int16)*factor
        stretched /= MAX_INT16

        # extract features
        label, id = fname.split("/")[-2:]
        id = id.rstrip(".wav")
        (X, label_idx, id) = transform((stretched, label, id), self.transform_params)
        X = X.to(self.device)

        return (X, label, label_idx, id)


    def __len__(self):
        return len(self.files)

MAX_INT16 = 2**15
DEFAULT_SR = 16000

def transform(data, transform_params):
    """
    Code adapted from AudioMNIST notebook
    """
    (x, label, id) = data[:3]
    sr = data[3] if len(data) > 3 else DEFAULT_SR
    label_idx = transform_params['label_to_idx'][label]

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
    if not X_norm.isfinite().all():
        print(label, id)
        X_norm = torch.zeros_like(X_norm)

    if len(X_norm.shape) == 2:
        X_norm = X_norm[np.newaxis]
    return (X_norm, label_idx, id)