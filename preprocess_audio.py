from util import SubsetSC, normalize, constant_q
from datasets import SCStretch, StretchedAudioMNIST
import os
from os.path import join, exists
import numpy as np
import yaml
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from tqdm import tqdm
import functools
from multiprocess import Pool
from morlet import phase_pow_multi
import matplotlib.pyplot as plt
import sys


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


def transform_and_save(dirname, transform_params, data):
    (X, label, label_idx, id) = data
    path = join(dirname, label, id + ".pt") # out_dir/label1/myfile0b83a31b.pt
    item = (X, label_idx)
    torch.save(item, path)


def save_data(dataset, dirname, transform_params, num_workers=4):

    for label in transform_params['label_to_idx'].keys():
        label_dir = join(dirname, label)
        os.makedirs(label_dir, exist_ok=True)

    #pool = Pool(num_workers)
    func = functools.partial(transform_and_save, dirname, transform_params)
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size, num_workers=num_workers, collate_fn=lambda x: x)
    # iterate through
    for b in tqdm(dataloader):
        for x in b:
            func(x)



if __name__ == "__main__":

    # assign indices based on training data
    """     print("Getting all labels")
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
    }  """

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

    # audiomnist params
    label_idx = {}
    for i in range(10):
        label_idx[str(i)] = i

    transform_params = {
        "method": "morlet",
        "fmin": 1000,
        "fmax": 24000, # Nyquist of 48k
        "nbins": 50,
        "norm_method": "zscore",
        "maxl": 50000,
        "resample_factor": 200,
        "label_to_idx": label_idx
    }


    # load datasets, expect in form (x, label, id), where id is filename
    #   or (x, label, id, sr)
    """     sc_root_dir = "data"
    train = SCStretch("training", sc_root_dir, 1.0, transform_params, 'cpu')
    val = SCStretch("validation", sc_root_dir, 1.0, transform_params, 'cpu')
    test = SCStretch("testing", sc_root_dir, 1.0, transform_params, 'cpu') """

    # audioMNIST stuff
    amn_root = "data/AudioMNIST"
    amn_raw = join(amn_root, "raw")
    split = (0.7, 0.15, 0.15)
    train = StretchedAudioMNIST("training", amn_raw, 1.0, transform_params, 'cpu', split)
    val = StretchedAudioMNIST("validation", amn_raw, 1.0, transform_params, 'cpu', split)
    test = StretchedAudioMNIST("testing", amn_raw, 1.0, transform_params, 'cpu', split)
    

    # output format = torch.saved (X, target_idx) saved with id filename, in target folders
    #   e.g. out_dir/train/label1/example.pt
    #out_dir = join(sc_root_dir, "SpeechCommands/processed_cqt_zscore")
    out_dir = join(amn_root, "processed")

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

    save_data(train, join(out_dir, "train"), transform_params)
    save_data(val, join(out_dir, "val"), transform_params)
    save_data(test, join(out_dir, "test"), transform_params)
