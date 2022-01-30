#! /bin/sh

module unload deeplearning

srun -p dl --gpus-per-node v100:1 --time=02:00:00 --pty bash
