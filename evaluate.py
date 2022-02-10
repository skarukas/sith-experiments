from util import Average, StretchAudioDataset
import sys
import yaml
from os.path import join
import argparse

from models.util import get_model
from util import collate_examples_list

import torch
from torch.utils.data import DataLoader


def evaluate(model, dataloader):
    loss = Average()
    acc = Average()
    model.eval()
    for (X, label) in dataloader:
        # compute loss
        prediction = model(X)
        curr_loss = model.loss_function(prediction, label)
        loss.update(curr_loss.item())

        # compute accuracy
        curr_acc = model.accuracy(prediction, label)
        acc.update(curr_acc.item())

    return {
        "loss": loss.get(),
        "acc":  acc.get()
    }


def parse_args():
    parser = argparse.ArgumentParser()
    default_data_dir = "data/SpeechCommands/processed/test"
    parser.add_argument("--ddir", type=str, default=default_data_dir)
    default_experiment_path = "out/SITH_train/sithcon_gsc_3layer_lr_0_1079330"
    parser.add_argument("--mpath", type=str, default=default_experiment_path)
    parsed = parser.parse_args()
    return vars(parsed)


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

    # import param file as dict
    f = open(join(experiment_path, "train_config.yaml"))
    config = yaml.safe_load(f)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(config)

    state_dict_path = join(experiment_path, config['model']['classname'] + ".pt")
    state_dict = torch.load(open(state_dict_path, "rb"))
    model.load_state_dict(state_dict)

    speeds = [0.1, 0.2, 0.4, 0.8, 1, 1.25, 2.50, 5, 10]
    for speed in speeds:
        dataset = StretchAudioDataset(test_data_dir, speed, label_to_idx=label_to_idx, device=config['device'])
        dataloader = DataLoader(
            dataset, 32, shuffle=False, 
            collate_fn=collate_examples_list
        )
        stats = evaluate(model, dataloader)
        print(f"For speed={speed}, acc={stats['acc']}, loss={stats['loss']}")