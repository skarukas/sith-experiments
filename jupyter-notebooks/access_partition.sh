#! /bin/sh

module unload deeplearning # otherwise we get some gcc version issues

srun -p dl --gpus-per-node 1 --time=04:00:00 --pty bash
#srun -p dl --time=03:00:00 --pty bash