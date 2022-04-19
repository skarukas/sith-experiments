import sys
import yaml
from os.path import join
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CIFAR10_Tensor, FastMNIST, TransformedImageDataset, TransformedMNIST
from evaluate import evaluate
from util import collate_examples_pad, Logger
from models.util import get_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
    else:
        experiment_path = "out/Deep_LP_train/lp_mnist_output_prelinear_ab_1191050/control-med"
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

    """ inner_dataset = CIFAR10_Tensor(
        data_dir, download=True, 
        train=False, device=config['device']
    ) """

    inner_dataset = FastMNIST(
        data_dir, download=True, 
        train=False, device=config['device']
    )
    ## evaluate
    transforms = [
        # scale
        dict(scale=1),
        dict(scale=4),
        dict(scale=3),
        dict(scale=2),
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
        dict(t_x=1, t_y=1),
        dict(t_x=2, t_y=2),
        dict(t_x=4, t_y=4),
        dict(t_x=6, t_y=6),
        dict(t_x=10, t_y=10),
    ]
    batch_size = 16#config['batch_size']
    results = []
    for transform_dict in tqdm(transforms):
        # create stretched version of dataset
        dataset = TransformedImageDataset(inner_dataset, **transform_dict)
        #k = list(transform_dict.keys())[0]
        #x = dataset[0][0].detach().permute(1, 2, 0).cpu().numpy()
        #x = (x - x.min()) / (x.max() - x.min())
        #plt.imsave(f"{k}:{transform_dict[k]}.png", x)
        dataloader = DataLoader(
            dataset, batch_size, shuffle=True, 
            collate_fn=None
        )
        stats = evaluate(model, dataloader, progress_bar=True)
        print(f"\nFor transform={transform_dict}:\n acc={stats['acc']}, loss={stats['loss']}")
        # write after every transform just in case
        results.append({ "transform": transform_dict,  **stats })
        results_file.seek(0)
        yaml.safe_dump(results, results_file)
        results_file.truncate()