import sys
import yaml
from os.path import join, abspath
import os
import argparse
import atexit
import dill
import shutil
from datetime import datetime
from tqdm import trange

from models import get_model
from dataloading import FileDataset
from util import Average

import torch
from torch.utils.data import DataLoader

# global parameters, automatically saved on exit
train_history = None
model = None
config = None


def train_loop(model, train_dataloader, config, val_dataloader=None):
    lr = config['optimizer']['params']['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr)
    epochs = trange(config['num_epochs'], bar_format='{l_bar}{bar:10}{r_bar}{bar:-510}')

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
        config['execution']['epochs_completed'] += 1
        epochs.set_postfix(epoch_stats)
            

def parse_args():
    parser = argparse.ArgumentParser()
    default_out_dir = f"out/{curr_time_str()}"
    parser.add_argument("--out_dir", type=str, default=default_out_dir, help="Output directory for this experiment only.")
    parser.add_argument("--param", type=str, help="Base parameter file")
    parsed = parser.parse_args()
    return vars(parsed)


def create_dir(dir):
    try:
        os.makedirs(dir, exist_ok=False)
        print(f"Created output directory '{dir}'")
    except OSError:
        overwrite = input(f"Warning: directory '{dir}' exists. Are you sure you want to overwrite its contents? [y/n] ")
        if overwrite.lower() in ("yes", "y"):
            shutil.rmtree(dir)
            os.makedirs(dir, exist_ok=False)
        else:
            sys.exit(1)
    

# store model and history on program exit
def cleanup():
    store_variable(train_history, 'train_history')
    store_variable(model, 'model_checkpoint')

    # output parameter file within the folder
    config['execution']['local_stop'] = curr_time_str()
    f = open(join(out_dir, 'train_config.yaml'), "w")
    yaml.safe_dump(config, f)


def store_variable(var, fname):
    if var is not None:
        dill.dump(var, join(out_dir, f"{fname}.dill"))


def curr_time_str():
    now = datetime.now().replace(microsecond=0)
    return now.isoformat(sep='_')


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

    # load data
    print("Loading training data")
    train_data = FileDataset(train_data_dir, device=config['device'])
    train_dataloader = DataLoader(train_data, config['batch_size'], shuffle=True)
    if val_data_dir is not None:
        print("Loading validation data")
        val_data = FileDataset(val_data_dir, device=config['device'])
        val_dataloader = DataLoader(val_data, config['batch_size'], shuffle=True)
    else:
        print("No validation dataset given.")


    model = get_model(config['model']).to(config['device'])
    train_history = []
    train_loop(model)

    config['execution']['program_exit'] = 'SUCCESS'