import sys
from time import sleep
import yaml
from os.path import join
import os
import argparse
import atexit
import dill
import shutil
from datetime import datetime
from tqdm import trange

import models.sithcon_utils as sutil
from dataloading import FileDataset
from util import Average

import torch

# global parameters, automatically saved on exit
train_history = None
model = None
train_params = None


def train_loop():
    train_history = []
    train_dataloader = None
    optimizer = None
    num_epochs = 10
    val_dataloader = None
    

    epochs = trange(num_epochs, bar_format='{l_bar}{bar:10}{r_bar}{bar:-510}')
    for epoch in epochs:
        train_loss = Average()
        val_loss = Average()
        train_acc = Average()
        val_acc = Average()
        
        model.train()
        for (X, label) in train_dataloader:
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
        train_params['execution']['epochs_completed'] += 1
        epochs.set_postfix(epoch_stats)
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory for this experiment only.")
    parser.add_argument("--param", type=str, help="Base parameter file")
    parsed = parser.parse_args()
    return vars(parsed)
    

# store model and history on program exit
def cleanup():
    store_variable(train_history, 'train_history')
    store_variable(model, 'model_checkpoint')

    # output parameter file within the folder
    train_params['execution']['local_stop'] = curr_time_str()
    f = open(join(out_dir, 'train_params.yaml'), "w")
    yaml.safe_dump(train_params, f)


def store_variable(var, fname):
    if var is not None:
        dill.dump(var, join(out_dir, f"{fname}.dill"))


def curr_time_str():
    now = datetime.now().replace(microsecond=0)
    return now.isoformat(sep=' ')


if __name__ == "__main__":
    config = parse_args()
    param_file = config['param']
    out_dir = config['out_dir']
    f = open(param_file)

    # import param file as dict
    train_params = yaml.safe_load(f)
    
    train_params['execution'] = {
        'epochs_completed': 0,
        'local_start': curr_time_str()
    }
    train_data_dir = train_params['train_data_dir']
    val_data_dir = train_params.get('val_data_dir')
    print(f"Loaded {param_file}")

    # make output directory for the experiment
    try:
        os.makedirs(out_dir, exist_ok=False)
        print(f"Created output directory '{out_dir}'")
    except OSError:
        overwrite = input(f"Warning: directory '{out_dir}' exists. Are you sure you want to overwrite its contents? [y/n] ")
        if overwrite.lower() in ("yes", "y"):
            shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=False)
        else:
            sys.exit(1)

    atexit.register(cleanup)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training using {config['device']}")

    # load data
    print("Loading training data")
    train_data = FileDataset(train_data_dir, device=config['device'])
    if val_data_dir is not None:
        print("Loading validation data")
        val_data = FileDataset(val_data_dir, device=config['device'])
    else:
        print("No validation dataset given.")

    train_loop()