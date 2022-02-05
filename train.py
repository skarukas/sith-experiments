import sys
import yaml
from os.path import join, abspath
import os
import argparse
import atexit
import dill
import shutil
from datetime import datetime
from tqdm import trange, tqdm
import random

from models.util import get_model
from util import Average, FileDataset

import torch
from torch.utils.data import DataLoader


# global parameters, automatically saved on exit
train_history = None
model = None
config = None


def train_loop(model, train_dataloader, config, val_dataloader=None):
    lr = config['optimizer']['params']['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr)
    epochs = trange(config['num_epochs'], desc="Epoch")

    for epoch in epochs:
        epoch_stats = {}
        train_loss = Average()
        val_loss = Average()
        train_acc = Average()
        val_acc = Average()
        
        model.train()
        batches = tqdm(train_dataloader, leave=False, desc="Batch")
        for (X, label) in batches:
            optimizer.zero_grad()
            # compute training loss
            prediction = model(X)
            loss = model.loss_function(prediction, label)
            train_loss.update(loss.item())

            loss.backward()
            optimizer.step()

            # compute training accuracy
            acc = model.accuracy(prediction, label)
            train_acc.update(acc.item())
            batch_stats = {
                "train_loss": loss.item(),
                "train_acc":  acc.item()
            }

            batches.set_postfix(batch_stats)
        
        epoch_stats = {
            "train_loss": train_loss.get(),
            "train_acc":  train_acc.get()
        }
        

        if val_dataloader:
            model.eval()
            for (X, label) in val_dataloader:
                # compute validation loss
                prediction = model(X)
                vloss = model.loss_function(prediction, label)
                val_loss.update(vloss.item())

                # compute validation accuracy
                acc = model.accuracy(prediction, label)
                val_acc.update(acc.item())

            epoch_stats = {
                **epoch_stats,
                "val_loss":   val_loss.get(),
                "val_acc":    val_acc.get(),
            }

        train_history.append(epoch_stats)
        config['execution']['epochs_completed'] += 1
        epochs.set_postfix(epoch_stats)
            

def parse_args():
    parser = argparse.ArgumentParser()
    default_out_dir = f"out/test_out/{curr_time_str()}"
    parser.add_argument("--out_dir", type=str, default=default_out_dir, 
        help="Output directory for this experiment only.")
    parser.add_argument("--param", type=str, help="Base parameter file")
    parsed = parser.parse_args()
    return vars(parsed)


def create_dir(dir):
    try:
        os.makedirs(dir, exist_ok=False)
        print(f"Created output directory '{dir}'")
    except OSError:
        overwrite = input(f"Warning: directory '{dir}' exists. "
            + "Are you sure you want to overwrite its contents? [y/n] ")
        if overwrite.lower() in ("yes", "y"):
            shutil.rmtree(dir)
            os.makedirs(dir, exist_ok=False)
        else:
            sys.exit(1)
    
WARNING = '\033[93m'
ENDC = '\033[0m'

# store model and history on program exit
def cleanup():
    if model is not None:
        model_class = config['model']['classname']
        torch.save(model.state_dict(), join(out_dir, model_class + ".pt"))
        store_variable(train_history, 'train_history')
        # output parameter file within the folder
        config['execution']['local_stop'] = curr_time_str()
        f = open(join(out_dir, 'train_config.yaml'), "w")
        yaml.safe_dump(config, f)


def store_variable(var, fname):
    if var is not None:
        f = open(join(out_dir, f"{fname}.dill"), "wb")
        dill.dump(var, f)


def curr_time_str():
    now = datetime.now().replace(microsecond=0)
    return now.isoformat(sep='_')


def generate_synthetic_data(dirname, n=100, num_classes=35, shape=None):
    for i in range(n):
        if shape is None:
            i_shape = (1, 50, random.randint(64, 256))
        X = torch.randn(i_shape)
        label = random.randint(0, num_classes-1)
        obj = (X, label)
        torch.save(obj, join(dirname, f"example_file_{i}.pt"))


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
    padded = torch.zeros((batch_size, *inp[0].shape[:-1], max_len), dtype=inp[0].dtype)
    for i in range(batch_size):
        l = lengths[i]
        padded[i, ..., -l:] = inp[i]
    targets = torch.tensor(targets)
    return padded, targets


if __name__ == "__main__":
    config = parse_args()
    param_file = config['param']
    out_dir = config['out_dir'] = abspath(config['out_dir'])
    f = open(param_file)

    # import param file as dict
    train_params = yaml.safe_load(f)
    print(f"Loaded {param_file}")
    
    config = {
        **train_params,
        **config,
        'execution': {
            'epochs_completed': 0,
            'local_start': curr_time_str(),
            'program_exit': 'FAILURE'
        }
    }
    train_data_dir = config['train_data_dir'] = abspath(config['train_data_dir'])
    if 'val_data_dir' in config:
        val_data_dir = config['val_data_dir'] = abspath(config['val_data_dir'])
    else:
        val_data_dir = None


    # make output directory solely for the experiment
    create_dir(out_dir)

    atexit.register(cleanup)

    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training using {config['device']}")

    collate = collate_examples_list if config.get('collate') == 'single' else collate_examples_pad
    # load data
    print("Loading training data")
    train_data = FileDataset(train_data_dir, device=config['device'])
    train_dataloader = DataLoader(
        train_data, config['batch_size'], shuffle=True, 
        collate_fn=collate
    )
    val_dataloader = None
    if val_data_dir is not None:
        print("Loading validation data")
        val_data = FileDataset(val_data_dir, device=config['device'])
        val_dataloader = DataLoader(
            val_data, config['batch_size'], shuffle=True, 
            collate_fn=collate
        )
    else:
        print("No validation dataset given.")

    
    model = get_model(config)
    config['model']['classname'] = model.__class__.__name__
    train_history = []
    train_loop(model, train_dataloader, config, val_dataloader)

    config['execution']['program_exit'] = 'SUCCESS'