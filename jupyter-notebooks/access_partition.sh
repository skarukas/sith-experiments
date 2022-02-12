#! /bin/sh

module unload deeplearning

srun -p dl --gpus-per-node p100:1 --time=04:00:00 --pty bash
#srun -p dl --time=03:00:00 --pty bash
