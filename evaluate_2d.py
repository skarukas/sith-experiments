import sys
import yaml
from os.path import join
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import TransformedMNIST
from evaluate import evaluate
from util import collate_examples_pad, Logger
from models.util import get_model


if __name__ == "__main__":
    experiment_path = "out/Deep_LP_train/lp_mnist_med_1153406/0"
    data_dir = "data"

    results_file = open(join(experiment_path, "evaluate_results.yaml"), "w")
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
    state_dict = torch.load(open(state_dict_path, "rb"), map_location=config['device'])
    model.load_state_dict(state_dict)

    
    ## evaluate
    transforms = [
        # normal
        dict(),
        # angle
        dict(max_angle_deg=5),
        dict(max_angle_deg=15),
        dict(max_angle_deg=30),
        dict(max_angle_deg=45),
        dict(max_angle_deg=60),
        dict(max_angle_deg=90),
        # scale
        dict(max_scale=0.2, min_scale=0.2),
        dict(max_scale=0.5, min_scale=0.5),
        dict(max_scale=0.8, min_scale=0.8),
        dict(max_scale=1.5, min_scale=1.5),
        dict(max_scale=2, min_scale=2),
        # translation
        dict(max_translate=2),
        dict(max_translate=5),
        dict(max_translate=10),
        dict(max_translate=20),
    ]
    batch_size = config['batch_size']
    results = []
    for transform_dict in tqdm(transforms):
        # create stretched version of SpeechCommands dataset
        dataset = TransformedMNIST(
            data_dir, device=config['device'], 
            download=True, train=False,
            **transform_dict
        )
        dataloader = DataLoader(
            dataset, batch_size, shuffle=True, 
            collate_fn=collate_examples_pad
        )
        stats = evaluate(model, dataloader, progress_bar=True)
        print(f"\nFor transform={transform_dict}:\n acc={stats['acc']}, loss={stats['loss']}")
        # write after every transform just in case
        results.append({ "transform": transform_dict,  **stats })
        yaml.safe_dump(results, results_file)