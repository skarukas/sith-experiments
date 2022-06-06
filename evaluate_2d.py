import sys
import yaml
from os.path import join
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
from datasets import get_dataset, CIFAR10_Tensor, FastMNIST, TransformedImageDataset, TransformedMNIST
from evaluate import evaluate
from util import collate_examples_pad, Logger, scale_ntau
from models.util import get_model
import matplotlib.pyplot as plt
from itertools import chain

DEFAULT_TRANSFORMS = [
        # scale
        #dict(scale=3),
        #dict(scale=4),
        dict(scale=2),
        dict(scale=1),
        dict(scale=1.5),
        dict(scale=0.8),
        dict(scale=0.6),
        dict(scale=0.5),

        # angle
        dict(angle=5),
        dict(angle=15),
        dict(angle=30),
        dict(angle=45),
        dict(angle=60),
        dict(angle=90),
        # translation
        dict(t_x=1, t_y=1, out_size=(28, 28)),
        dict(t_x=2, t_y=2, out_size=(28, 28)),
        dict(t_x=3, t_y=3, out_size=(28, 28)),
        dict(t_x=4, t_y=4, out_size=(28, 28)),
        dict(t_x=-1, t_y=-1, out_size=(28, 28)),
        dict(t_x=-2, t_y=-2, out_size=(28, 28)),
        dict(t_x=-3, t_y=-3, out_size=(28, 28)),
        dict(t_x=-4, t_y=-4, out_size=(28, 28))
    ]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
    else:
        experiment_path = "out/Deep_LP_train/lp_mnist_output_prelinear_ab_1191050/control-med"
    data_dir = "data"

    results_file = open(join(experiment_path, "evaluate_scale_results.yaml"), "w")
    sys.stdout = Logger(join(experiment_path, "evaluate_out.txt"), sys.stdout)
    sys.stderr = Logger(join(experiment_path, "evaluate_err.txt"), sys.stderr)

    ## import param file as dict
    f = open(join(experiment_path, "train_config.yaml"))
    config = yaml.safe_load(f)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", config['device'])


    #scale_ntau(config, scale=1.5)

    ## get model and load state dict
    model = get_model(config)
    # assume model state dict stored as something like MyClassifier.pt
    state_dict_path = join(experiment_path, config['model']['classname'] + ".pt")
    state_dict = torch.load(open(state_dict_path, "rb"), map_location=config['device'])
    model.load_state_dict(state_dict)

    #config["val_data_dir"]["type"] = "FastMNIST"

    inner_dataset = get_dataset(config["val_data_dir"], device=config['device'])
    ## evaluate

    transforms = DEFAULT_TRANSFORMS
    #transforms = [dict(scale=1)]
    n_angles = 24
    #transforms = [dict(angle=i*(360/n_angles)) for i in range(n_angles)]
    transforms = [
        dict(scale=2),
        dict(scale=1),
        dict(scale=1.5),
        dict(scale=0.8),
        dict(scale=0.6),
        dict(scale=0.5)
    ]

    batch_size = 2#config['batch_size']
    results = []
    dataset_list = (
        (TransformedImageDataset(inner_dataset, **transform_dict), transform_dict) 
        for transform_dict in transforms
    )

    # for special datasets
    #d_kwargs = config["val_data_dir"]["kwargs"]
    #dataset_list = [
    #    (datasets.RotSVHN(**d_kwargs, device=config['device']), "RotSVHN")
    #]

    for dataset, transform in tqdm(dataset_list):
        """ k = list(transform.keys())[0]
        x = dataset[0][0].detach().permute(1, 2, 0).cpu().numpy()
        print(x.min(), x.max())
        x = (x - x.min()) / (x.max() - x.min())
        print("imsize:", x.shape)
        plt.imsave(f"{k}_{transform[k]}.png", x[..., 0]) """
        dataloader = DataLoader(
            dataset, batch_size, shuffle=True, 
            collate_fn=None
        )

        stats = evaluate(model, dataloader, progress_bar=True)

            
        print(f"\nFor transform={transform}:\n acc={stats['acc']}, loss={stats['loss']}")
    
        # write after every transform just in case
        results.append({ "transform": transform,  **stats })
        results_file.seek(0)
        yaml.safe_dump(results, results_file)
        results_file.truncate()