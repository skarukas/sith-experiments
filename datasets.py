from torch.utils.data import Dataset
import torch
import numpy as np
import random

from os import path
import glob

from torchvision.datasets import MNIST
from audiotsm import phasevocoder
from audiotsm.io.array import ArrayWriter, ArrayReader
import scipy.io.wavfile as wavfile
from torchaudio.datasets import SPEECHCOMMANDS
from PIL import Image

from morlet import phase_pow_multi
from scipy import signal
from util import constant_q, normalize
import matplotlib.pyplot as plt


MAX_INT16 = 2**15
DEFAULT_SR = 16000


def get_dataset(dataset_params, device):
    if isinstance(dataset_params, str):
        root = dataset_params
        return FileDataset(root, device=device)
    else:
        # create instance of dataset class
        dataset_type = dataset_params['type'].lower()
        class_map = {
            "fastmnist": FastMNIST
        }
        DatasetClass = class_map[dataset_type]
        return DatasetClass(*dataset_params.get('args', []), **dataset_params.get('kwargs', {}), device=device)
        
        


class SCStretch(SPEECHCOMMANDS):
    def __init__(self, subset: str, root_dir: str, speed: float, transform_params: dict, device='cpu'):
        super().__init__(root_dir, download=False)

        def load_list(filename):
            filepath = path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self.files = load_list("validation_list.txt")
        elif subset == "testing":
            self.files = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(map(path.abspath, excludes))
            self.files = [w for w in self._walker if path.abspath(w) not in excludes]
        
        self.device = device
        self.speed = speed
        self.transform_params = transform_params
        print(f"found {len(self.files)} files")
    

    def __getitem__(self, idx):
        fname = self.files[idx]
        #x, sr = torchaudio.load(fname)
        sr, x = wavfile.read(fname)
        maxl = self.transform_params['maxl']
        x = np.pad(x, (int(np.floor((maxl - x.shape[0])/2)),
                int(np.ceil((maxl - x.shape[0])/2))), 'constant')

        stretched = stretch_audio(x, self.speed)

        # extract features
        label, id = fname.split("/")[-2:]
        id = id.rstrip(".wav")
        label_idx = self.transform_params['label_to_idx'][label]
        X = transform(stretched, self.transform_params, sr)
        X = X.to(self.device)

        return (X, label, label_idx, id)


    def __len__(self):
        return len(self.files)

SHUFFLE_SEED = 11111


class StretchedAudioMNIST(Dataset):
    """
    Expects that root_dir contains a bunch of wav files that have the
         class as the first char of the filename
    Has the same output format as StretchSC
    """
    def __init__(self, subset: str, root_dir: str, speed: float, transform_params: dict, device='cpu', split=(0.7, 0.15, 0.15)):
        self.device = device
        gb_path = path.join(glob.escape(root_dir), "**/*")
        print(f"Using glob '{gb_path}'...", end=" ")
        gb = glob.glob(gb_path, recursive=True)
        
        allfiles = [f for f in gb if path.isfile(f) and f.endswith(".wav")]
        # apply fixed random tr/te/val split
        random.seed(SHUFFLE_SEED)
        random.shuffle(allfiles)
        nfiles = len(allfiles)
        s0 = int(split[0] * nfiles)
        s1 = int((split[0]+split[1]) * nfiles)
        splitfiles = allfiles[:s0], allfiles[s0:s1], allfiles[s1:]
        print([*map(len, splitfiles)])
        idx = ["training", "validation", "testing"].index(subset)

        self.files = splitfiles[idx]
        self.transform_params = transform_params
        self.speed = speed
        print(f"found {len(self.files)} files")


    def __getitem__(self, idx):
        fname = self.files[idx]
        sr, x = wavfile.read(fname)
        maxl = self.transform_params['maxl']
        x = np.pad(x, (int(np.floor((maxl - x.shape[0])/2)),
                int(np.ceil((maxl - x.shape[0])/2))), 'constant')

        stretched = stretch_audio(x, self.speed)

        # extract features
        short_name = fname.split("/")[-1]
        label = short_name.split("_")[0]
        label_idx = self.transform_params['label_to_idx'][label]

        id = short_name.rstrip(".wav")
        
        X = transform(stretched, self.transform_params, sr)
        X = X.to(self.device)

        return (X, label, label_idx, id)

    def __len__(self):
        return len(self.files)


def stretch_audio(x, speed):
    factor = 1 / speed
    reader = ArrayReader(x[np.newaxis])
    writer = ArrayWriter(1)
    tsm = phasevocoder(1, speed=speed)
    tsm.run(reader, writer)
    stretched = writer.data[0].astype(np.int16)*factor
    return stretched


def transform(x, transform_params, sr=DEFAULT_SR):
    """
    Code adapted from AudioMNIST notebook
    """

    # pad audio data (not strictly necessary, 
    #   but helps short files not run into issues during morlet transform)

    if transform_params['method'] == "morlet":
        # compute morlet transform
        X = phase_pow_multi(
            transform_params['morlet_freqs'], x, samplerates=sr, widths=5,
            to_return='power', time_axis=-1,
            conv_dtype=np.complex64, freq_name='freqs'
        )
    else:
        # constant-Q transform
        X = constant_q(
            x, sr=sr, 
            fmin=transform_params['fmin'], fmax=transform_params['fmax'], 
            bins=transform_params['nbins'], hop_length=transform_params['hop_length']
        )[0]
    # resample 2D features
    resample_factor = transform_params['resample_factor']
    if resample_factor is not None:
        X = signal.resample(X, X.shape[1]//resample_factor, axis=1)
    
    X_norm = normalize(X, transform_params['norm_method'])
    X_norm = torch.tensor(X_norm)
    X_norm[~X_norm.isfinite()] = 0

    if len(X_norm.shape) == 2:
        X_norm = X_norm[np.newaxis]
    return X_norm


class FileDataset(Dataset):
    """
    Expects that dir contains a bunch of "torch.saved" files 
        in subdirectories.
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


class FastMNIST(MNIST):
    def __init__(self, root, device='cpu', download=True, *args, **kwargs):
        super().__init__(root, download=download, *args, **kwargs)
        
        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on device in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        return img, target


class TransformedMNIST(MNIST):
    def __init__(self, root, device='cpu', download=True, max_translate=0, 
                max_angle_deg=0, min_scale=1, max_scale=1, out_size=None,
                *args, **kwargs):
        super().__init__(root, download=download, *args, **kwargs)
        self.max_translate = max_translate
        self.max_angle_deg = max_angle_deg
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.device = device
        self.targets = self.targets.to(device)
        size = max(int(max_scale*(28 + 2*max_translate)), 28)
        self.imsize = (size, size) if out_size is None else out_size


    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        angle = random.uniform(-self.max_angle_deg, self.max_angle_deg)
        t_x = random.randint(-self.max_translate, self.max_translate)
        t_y = random.randint(-self.max_translate, self.max_translate)
        scale = random.uniform(self.min_scale, self.max_scale)

        inner_image = Image.fromarray(image.numpy()) 
        image = Image.new("L", size=self.imsize)
        offset = ((image.width - inner_image.width) // 2, (image.height - inner_image.height) // 2)
        image.paste(inner_image, offset)

        mat = (
            1, 0, t_x,
            0, 1, t_y
        )
        image = image.rotate(angle, fillcolor=0)
        image = image.transform(image.size, Image.AFFINE, mat, fillcolor=0)
        image = rescale_centered(image, scale)
        
        image = torch.tensor(np.array(image), dtype=float, device=self.device).unsqueeze(0)
        image = image.div(255)
        image = (image - 0.1307) / 0.3081
        return image, target


def rescale_centered(image, scale):
    width, height = image.width, image.height
    image = image.resize((int(width*scale), int(height*scale)))
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    right = left + width
    bottom = top + height

    return image.crop((left, top, right, bottom))


if __name__ == "__main__":
    dataset = TransformedMNIST("data", max_translate=20, max_angle_deg=15, min_scale=0.4, max_scale=1)
    for i in range(10):
        image, target = dataset[i]
        plt.imshow(image.numpy()[0])
        plt.show()
