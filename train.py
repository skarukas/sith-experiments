import sys
import yaml
import models.sithcon_utils as sutil
from os.path import join
import os
import argparse
import atexit
import dill


train_history = None
model = None

def train_loop():
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="out", help="Base parameter file")
    parser.add_argument("--param", type=str, help="Base parameter file")
    parsed = parser.parse_args()
    return vars(parsed)
    

# store model and history on program failure
def cleanup():
    store_variable(train_history, 'train_history')
    store_variable(model, 'model_checkpoint')

def store_variable(var, fname):
    if var is not None:
        dill.dump(var, join(out_dir, f"{fname}.dill"))

atexit.register(cleanup)


if __name__ == "__main__":
    config = parse_args()
    param_file = config['param']
    out_dir = config['out_dir']
    f = open(param_file)

    # import param file as dict
    params = yaml.safe_load(f)
    print(f"Loaded {param_file}")

    # make output directory for the experiment
    try:
        os.makedirs(out_dir, exist_ok=False)
        print(f"Created output directory '{out_dir}'")
    except OSError:
        overwrite = input(f"Warning: directory '{out_dir}' exists. Are you sure you want to overwrite its contents? [y/n] ")
        if overwrite.lower() not in ("yes", "y"):
            sys.exit(1)
    