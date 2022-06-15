# Deep Log-Polar Networks

This repo contains code+results for a project in which we train and evaluate image classification networks that are rotation and scale-invariant out-of-the-box.

## Concepts

At the moment, there is a small [powerpoint](etc/Deep%20Log-polar%20networks.pdf) that describes the general architecture of the deep log-polar network. (I can't say whether that might be more or less helpful than the README).

### LPConv layer

The building block of most of the networks is an invariant log-polar space convolution layer ("`LPConv`" layer). This takes in a `H x W x C_in` tensor, performs a bunch of log-polar transforms, one at every pixel in the image, and produces a `H x W x C_in x N_tau x N_theta` tensor (order of dims may be incorrect). Then a 2D convolution and pooling is performed over the last 2 dims. This extracts invariant features, and outputs as `H x W x C_in`. For example, a high activation at `(2, 1)` in the first channel might mean there's an edge at *some* angle and *some* scale.

### Models

The [deep log-polar network](models/DeepLogPolarClassifier.py) is a series of invariant log-polar space convolution layers, with depthwise linear layers (== 1x1 convolutions) in between them.

The [LP-ResNet network](models/LPResNet.py) is just a ResNet, except every convolutional layer is replaced by an invariant log-polar space convolution layer.

There's also the [single log-polar transform network](models/SingleLPClassifier.py), where instead of just one conv-pooling operation in log-polar space, the whole network operates in log-polar space.

In short:
- DeepLP:  `[LP Transform -> Conv+pool] -> [LP Transform -> Conv+pool] -> ...`
- SingleLP: `LP Transform -> Deep Network -> Pool` 

## File Organization

### Overall

- `models/`: Python model code. Containing lots of weird custom layers that might not be useful.
- `out/`: script-generated logs + checkpoints + config files + model weights. `running_jobs` contains the stderr and stdout files for the currently running jobs, which are useful to model to see training progress and model summary (Note: these aren't auto-deleted, unfortunately).
- `param-files/`: YAML files that fully specify a training run. After creating one here, you can run it, or a few at a time, by editing `train-batch.sh` then running `sbatch train_batch.sh`.

Both `out` and `param-files` are organized like `model_name/dataset_name`.


### Training

- A training run is specified by a YAML file, for example [this one for MNIST](param-files/Deep-LP/MNIST/rotation_test/standard.yaml).

- These YAML files are loaded using `train_batch.sh`, which creates the models and trains them
  - You'll have to specify the parent directory in `train_batch.sh` by editing `ExperimentRelativePath`, then put the filenames (without .yaml) in ParamFiles like this: `ParamFiles=(resnet20 resnet32 resnet44 resnet56)`. You'll have to change `#SBATCH --array=0-3` to select from that list (for example, if you have 3 param files you only need `--array=0-2`).

### Evaluation

- After training, you can evaluate a trained model by running `python3 evaluate_2d.py` with the model's experiment directory, for example `out/Deep_LP_train/mnist_r/bilinear_ntau20_new_427144/30deg_rotations_tk12_lg_1`. Or you can use `evaluate_batch.sh` to run a few at a time.
- You'll have to edit `evaluate_2d.py` to generate whatever datasets you want to test the models on (it's a bit of a mess, sorry).
- After running that script, an `evaluate_results.yaml` file will appear in the experiment's directory. If you're using `evaluate_batch.sh`, you can also monitor the progress by looking in the `out/.../running_jobs` directory.