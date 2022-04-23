#! /bin/sh

module unload deeplearning # otherwise we get some gcc version issues

#srun -p gpu --gpus-per-node=1 --time=01:00:00 --pty bash
srun -p debug --time=01:00:00 --pty bash