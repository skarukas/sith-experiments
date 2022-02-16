import sys
import yaml
from os.path import join
import argparse
from tqdm import tqdm

from models.util import get_model
from util import Average, collate_examples_pad, Logger, FileDataset
from datasets import SCStretch

import numpy as np
import torch
from torch.utils.data import DataLoader


def evaluate(model, dataloader, progress_bar=False):
    """
    Evaluate a model on data.
    """
    loss = Average()
    acc = Average()
    model.eval()
    if progress_bar:
        dataloader = tqdm(dataloader, desc="Evaluation", leave=False)
    batch_stats = {}
    for (X, label) in dataloader:
        # compute loss
        prediction = model(X)
        curr_loss = model.loss_function(prediction, label)
        loss.update(curr_loss.item())

        # compute accuracy
        curr_acc = model.accuracy(prediction, label)
        acc.update(curr_acc.item())
        batch_stats = {
            "loss": loss.get(),
            "acc":  acc.get()
        }
        if progress_bar:
            dataloader.set_postfix(batch_stats)

    return batch_stats


def parse_args():
    parser = argparse.ArgumentParser()
    default_data_dir = "data/"
    default_experiment_path = "out/SITH_train/sithcon_gsc_cqt_new_1098838/1"

    parser.add_argument("--ddir", 
        type=str, default=default_data_dir, 
        help="Directory where the data is found. " \
            + "May be dependent upon the implementation of the Dataset."
    )
    parser.add_argument("--mpath", 
        type=str, default=default_experiment_path,
        help="Directory including saved model dict and YAML parameter file.")
    parsed = parser.parse_args()
    return vars(parsed)


# SpeechCommands mapping
label_to_idx = {
    "backward": 0,
    "bed": 1,
    "bird": 2,
    "cat": 3,
    "dog": 4,
    "down": 5,
    "eight": 6,
    "five": 7,
    "follow": 8,
    "forward": 9,
    "four": 10,
    "go": 11,
    "happy": 12,
    "house": 13,
    "learn": 14,
    "left": 15,
    "marvin": 16,
    "nine": 17,
    "no": 18,
    "off": 19,
    "on": 20,
    "one": 21,
    "right": 22,
    "seven": 23,
    "sheila": 24,
    "six": 25,
    "stop": 26,
    "three": 27,
    "tree": 28,
    "two": 29,
    "up": 30,
    "visual": 31,
    "wow": 32,
    "yes": 33,
    "zero": 34,
}


if __name__ == "__main__":
    argp = parse_args()
    experiment_path = argp['mpath']
    test_data_dir = argp['ddir']
    
    transform_params_path = "data/SpeechCommands/processed_cqt_zscore/transform_params.yaml"
    f = open(transform_params_path)
    transform_params = yaml.safe_load(f)

    if transform_params['method'] == "morlet":
        transform_params['morlet_freqs'] = np.logspace(
            np.log(transform_params['fmin']), 
            np.log(transform_params['fmax']), 
            transform_params['nbins'], 
            base=np.e
        )

    sys.stdout = Logger(join(experiment_path, "evaluate_out.txt"), sys.stdout)
    sys.stderr = Logger(join(experiment_path, "evaluate_err.txt"), sys.stderr)


    ## import param file as dict
    f = open(join(experiment_path, "train_config.yaml"))
    config = yaml.safe_load(f)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", config['device'])


    ## get model and load state dict
    model = get_model(config)
    # assume model state dict stored as something like MyClassifier.pt
    state_dict_path = join(experiment_path, config['model']['classname'] + ".pt")
    state_dict = torch.load(open(state_dict_path, "rb"))
    model.load_state_dict(state_dict)

    
    ## evaluate
    # Note: slower speeds may take a LONG time
    speeds = [0.1, 0.2, 0.4, 0.8, 1, 1.25, 2.50, 5, 10]
    # to avoid CUDA out of memory error for the longer (=slower) samples, 
    #   each batch size is dependent upon the size of the data.
    # make this value smaller if you get CUDA out of memory error

    batch_size_normal_speed = 32
    for speed in reversed(speeds):
        # create stretched version of SpeechCommands dataset
        dataset = SCStretch(
            "testing", test_data_dir, speed, 
            transform_params=transform_params, 
            device=config['device']
        )
        batch_size = int(speed * batch_size_normal_speed) 
        dataloader = DataLoader(
            dataset, batch_size, shuffle=True, 
            collate_fn=lambda data: collate_examples_pad([(x[0], x[2]) for x in data])
        )
        stats = evaluate(model, dataloader, progress_bar=True)
        print(f"For speed={speed}, acc={stats['acc']}, loss={stats['loss']}")
