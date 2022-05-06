import sys
import yaml
from os.path import join, abspath
import os
import argparse
import atexit
import dill
import shutil
from tqdm import trange, tqdm
import random


from models.util import get_model
import util
from util import Average, Logger
from evaluate import evaluate
from datasets import get_dataset

import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity


CHECKPOINT_FREQ = 1
CHECKPOINT_DIRNAME = "checkpoints"

# global parameters, automatically saved on exit
train_history = None
model = None
config = None

RECORD_PROFILE = True
PROFILE_NBATCH = 10


def train_loop(model, train_dataloader, config, val_dataloader=None):
    lr = config['optimizer']['params']['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr)
    epochs = trange(config['num_epochs'], desc="Epoch")

    for epoch in epochs:
        epoch_stats = {}
        train_loss = Average()
        train_acc = Average()
        
        model.train()

        ## profile model computation on a few batches
        if RECORD_PROFILE and epoch == 0:
            print("Profiling model...")
            with profile(activities=[
                ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                profile_memory=True, record_shapes=True) as prof:
                for i, (x, l) in enumerate(train_dataloader):
                    if i == PROFILE_NBATCH:
                        break
                    optimizer.zero_grad()
                    with record_function(f"batch_{i}"):
                        yh = model(x)
                        loss = model.loss_function(yh, l)
                    with record_function(f"batch_{i}_backward"):
                        loss.backward()

            prof.export_chrome_trace(join(log_dir, "training_chrome_trace.json"))
            print("Operations sorted by CUDA time:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

        ## train
        batches = tqdm(train_dataloader, leave=False, desc="Batch", mininterval=10)
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
                "train_acc":  acc.item(),
                "avg_tr_acc": train_acc.get(),
                "avg_tr_loss": train_loss.get()
            }

            batches.set_postfix(batch_stats, refresh=False)
        
        epoch_stats = {
            "train_loss": train_loss.get(),
            "train_acc":  train_acc.get()
        }
        

        if val_dataloader:
            val_stats = evaluate(model, val_dataloader)
            epoch_stats = {
                **epoch_stats,
                "val_loss": val_stats['loss'],
                "val_acc":  val_stats['acc']
            }

        train_history.append(epoch_stats)
        config['execution']['epochs_completed'] += 1
        epochs.set_postfix(epoch_stats)

        if (epoch+1) % CHECKPOINT_FREQ == 0:
            checkpoint_dir = join(out_dir, CHECKPOINT_DIRNAME)
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_class = config['model']['classname']
            torch.save(model.state_dict(), join(checkpoint_dir, f"{model_class}_epoch{epoch}.pt"))
            save_progress()

            

def parse_args():
    parser = argparse.ArgumentParser()
    default_out_dir = f"out/test_out/{util.curr_time_str()}"
    parser.add_argument("--out_dir", type=str, default=default_out_dir, 
        help="Output directory for this experiment only.")
    parser.add_argument("--param", type=str, help="Base parameter file")
    parsed = parser.parse_args()
    return vars(parsed)


def create_dir(dir):
    try:
        os.makedirs(dir, exist_ok=False)
        print(f"Created output directory '{dir}'")
    except:
        try:
            overwrite = input(f"Warning: directory '{dir}' exists. " \
                + "Are you sure you want to overwrite its contents? [y/n] ")
        except EOFError:
            overwrite = "y"
        if overwrite.lower() in ("yes", "y"):
            shutil.rmtree(dir)
            os.makedirs(dir, exist_ok=False)
        else:
            sys.exit(1)
    
WARNING = '\033[93m'
ENDC = '\033[0m'

# store model and history on checkpoint / program exit
def save_progress():
    if model is not None:
        model_class = config['model']['classname']
        torch.save(model.state_dict(), join(out_dir, model_class + ".pt"))
        if train_history is not None and len(train_history) > 0:
            f = open(join(out_dir, "train_history.dill"), "wb")
            dill.dump(train_history, f)
            config['execution']['stats'] = train_history[-1]
        # output parameter file within the folder
        config['execution']['local_stop'] = util.curr_time_str()
        f = open(join(out_dir, 'train_config.yaml'), "w")
        yaml.safe_dump(config, f)


def generate_synthetic_data(dirname, n=100, num_classes=35, shape=None):
    shutil.rmtree(dirname, ignore_errors=True)
    os.makedirs(dirname, exist_ok=False)
    for i in range(n):
        if shape is None:
            i_shape = (1, 50, random.randint(4, 10))
        X = torch.randn(i_shape)
        label = random.randint(0, num_classes-1)
        obj = (X, label)
        torch.save(obj, join(dirname, f"example_file_{i}.pt"))


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
            'local_start': util.curr_time_str(),
            'program_exit': 'FAILURE'
        }
    }
    train_data_dir = config['train_data_dir']
    if isinstance(train_data_dir, str):
        train_data_dir = config['train_data_dir'] = abspath(config['train_data_dir'])
    if 'val_data_dir' in config:
        val_data_dir = config['val_data_dir']
        if isinstance(val_data_dir, str):
            val_data_dir = config['val_data_dir'] = abspath(config['val_data_dir'])
    else:
        val_data_dir = None

    #generate_synthetic_data(train_data_dir, 10)
    
    # (recursively) make output directory, solely for the experiment
    log_dir = join(out_dir, "log")
    create_dir(log_dir)

    # hacky way to redirect to special output files, 
    #   as shell redirection with slurm was getting odd with all the
    #   directories that needed to exists
    sys.stderr = Logger(join(log_dir, "stderr.txt"), sys.stderr)
    sys.stdout = Logger(join(log_dir, "stdout.txt"), sys.stdout)

    atexit.register(save_progress)

    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training using {config['device']}")

    collate = None # util.collate_examples_list if config.get('collate') == 'single' \
    #else util.collate_examples_pad
    # load data
    print("Loading training data")
    train_data = get_dataset(train_data_dir, device=config['device'])
    train_dataloader = DataLoader(
        train_data, config['batch_size'], shuffle=True, 
        collate_fn=collate
    )
    val_dataloader = None
    if val_data_dir is not None:
        print("Loading validation data")
        val_data = get_dataset(val_data_dir, device=config['device'])
        val_dataloader = DataLoader(
            val_data, config['batch_size'], shuffle=True, 
            collate_fn=collate
        )
    else:
        print("No validation dataset given.")

    
    model = get_model(config)
    print("Model Architecture:")
    print(model)
    config['model']['classname'] = model.__class__.__name__
    config['model']['num_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters:", config['model']['num_params'])
    train_history = []
    train_loop(model, train_dataloader, config, val_dataloader)

    config['execution']['program_exit'] = 'SUCCESS'
