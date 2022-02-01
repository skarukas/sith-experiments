from torch.utils.data import Dataset
import torch

from os import path
import glob


class Average:
    """
    Keep running average of a series of observations
    """
    def __init__(self):
        self.n = self.sum = 0
    
    def update(self, num, n=1):
        self.n += n
        self.sum += num

    def get(self):
        return self.sum / self.n if self.n else 0


class FileDataset(Dataset):
    """
    Expects that dir contains a bunch of "torch.saved" files
    """
    def __init__(self, dir, device='cpu'):
        self.device = device
        gb_path = path.join(glob.escape(dir), "**/*")
        print(f"Using glob '{gb_path}'...", end=" ")

        gb = glob.glob(gb_path, recursive=True)
        self.files = [f for f in gb if path.isfile(f)]
        print(f"found {len(self.files)} files")

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            return torch.load(f, map_location=self.device)

    def __len__(self):
        return len(self.files)