from util import SubsetSC, normalize, constant_q
import os
from os.path import join, exists
import numpy as np
import yaml
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from scipy import signal
from tqdm import tqdm
import functools
from multiprocess import Pool
from morlet import phase_pow_multi
import matplotlib.pyplot as plt
import sys

DEFAULT_SR = 16000


def get_label_indices(dataset):
    # collect and assign labels
    idx = 0
    label_idx = {}
    maxl = 0
    for dat in tqdm(dataset):
        x, label = dat[:2]
        maxl = max(len(x.flatten()), maxl)
        if label not in label_idx:
            label_idx[label] = idx
            idx += 1
    return label_idx, maxl


max_int16 = 2**15
"""
class ProcessedDataset(Dataset):
    def __init__(self, inner: Dataset, transform_params: dict):
        self.inner = inner
        self.transform_params = transform_params
        self.freqs = np.logspace(
            np.log(transform_params['fmin']), 
            np.log(transform_params['fmax']), 
            transform_params['nbins'], 
            base=np.e
        )

    def __getitem__(self, idx):
        item = self.inner[idx]
        (x, label, id) = item[:3]
        sr = item[3] if len(item) > 3 else DEFAULT_SR
        label_idx = self.transform_params['label_to_idx'][label]
        dat = x / max_int16
        maxl = self.transform_params['maxl']
        dat = np.pad(dat, (int(np.floor((maxl - dat.shape[0])/2)),
                    int(np.ceil((maxl - dat.shape[0])/2))), 'constant')
        X = phase_pow_multi(
            self.freqs, dat,  samplerates=sr, widths=5,
            to_return='power', time_axis=-1,
            conv_dtype=np.complex64, freq_name='freqs',
        )
        resample_factor = self.transform_params['resample_factor']
        X = signal.resample(X, X.shape[1]//resample_factor, axis=1)
        X = normalize(X, self.transform_params['norm_method'])
        return (X, label_idx, id)


    def __len__(self, idx):
        return len(self.inner)
"""

def transform(data, params):
    """
    Code adapted from AudioMNIST notebook
    """
    (x, label, id) = data[:3]
    sr = data[3] if len(data) > 3 else DEFAULT_SR
    label_idx = transform_params['label_to_idx'][label]

    # pad audio data (not strictly necessary, 
    #   but helps short files not run into issues during morlet transform)
    maxl = transform_params['maxl']
    x = x.flatten()
    currl = len(x)
    dat = np.pad(x, (int(np.floor((maxl - currl)/2)),
                int(np.ceil((maxl - currl)/2))), 'constant')

    if transform_params['method'] == "morlet":
        # compute morlet transform
        X = phase_pow_multi(
            transform_params['morlet_freqs'], dat, samplerates=sr, widths=5,
            to_return='power', time_axis=-1,
            conv_dtype=np.complex64, freq_name='freqs'
        )
    else:
        # constant-Q transform
        X = constant_q(
            dat, sr=sr, 
            fmin=transform_params['fmin'], fmax=transform_params['fmax'], 
            bins=transform_params['nbins'], hop_length=transform_params['hop_length']
        )[0]
    # resample 2D features
    resample_factor = transform_params['resample_factor']
    if resample_factor is not None:
        X = signal.resample(X, X.shape[1]//resample_factor, axis=1)
    
    X = normalize(X, transform_params['norm_method'])
    X = torch.tensor(X)
    if len(X.shape) == 2:
        X = X[np.newaxis]
    return (X, label_idx, id)


def transform_and_save(dirname, transform_params, data):
    x, label, id = data
    path = join(dirname, label, id + ".pt") # out_dir/label1/myfile0b83a31b.pt
    if not exists(path): 
        (X, label_idx, id) = transform(data, transform_params)
        #plt.imshow(X, aspect='auto')
        #plt.savefig('example.png')
        #sys.exit(0)
        
        item = (X, label_idx)
        torch.save(item, path)


def save_data(dataset, dirname, transform_params, num_workers=4):

    for label in transform_params['label_to_idx'].keys():
        label_dir = join(dirname, label)
        os.makedirs(label_dir, exist_ok=True)

    pool = Pool(num_workers)
    func = functools.partial(transform_and_save, dirname, transform_params)
    mapper = tqdm(
        map(func, dataset),
        #pool.imap_unordered(func, dataset),
        total=len(dataset),
        leave=False
    )
    # iterate through
    _ = [*mapper]



if __name__ == "__main__":
    # load datasets, expect in form (x, label, id), where id is filename
    #   or (x, label, id, sr)
    sc_root_dir = "data"
    train, val, test = SubsetSC(sc_root_dir, "training"), SubsetSC(sc_root_dir, "validation"), SubsetSC(sc_root_dir, "testing")

    # output format = torch.saved (X, target_idx) saved with id filename, in target folders
    #   e.g. out_dir/train/label1/example.pt
    out_dir = join(sc_root_dir, "SpeechCommands/processed_cqt_zscore")

    # assign indices based on training data
    print("Getting all labels")
    maxl = 16000
    label_idx = {}
    for i, lab in enumerate(['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']):
        label_idx[lab] = i
    #label_idx, maxl = get_label_indices(train)
    labels = list(label_idx.keys())
    print("Num labels:", len(labels))
    print(labels)
    print("maxl:", maxl)

    # cqt zscore
    transform_params = {
        "method": "constant_q",
        "fmin": 100,
        "fmax": 8000, # Nyquist of 16000Hz
        "nbins": 50,
        "hop_length": 64,
        "norm_method": "zscore",
        "maxl": maxl,
        "resample_factor": None,
        "label_to_idx": label_idx
    } 

    # morlet zscore
    """ transform_params = {
        "method": "morlet",
        "fmin": 100,
        "fmax": 8000, # Nyquist of 16000Hz
        "nbins": 50,
        "norm_method": "zscore",
        "maxl": maxl,
        "resample_factor": 100,
        "label_to_idx": label_idx
    } """

    os.makedirs(out_dir, exist_ok=True)
    # write out transformation details
    f = open(join(out_dir, "transform_params.yaml"), "w")
    yaml.safe_dump(transform_params, f)

    if transform_params['method'] == "morlet":
        transform_params['morlet_freqs'] = np.logspace(
            np.log(transform_params['fmin']), 
            np.log(transform_params['fmax']), 
            transform_params['nbins'], 
            base=np.e
        )

    #save_data(train, join(out_dir, "train"), transform_params)
    #save_data(val, join(out_dir, "val"), transform_params)
    save_data(test, join(out_dir, "test"), transform_params)
