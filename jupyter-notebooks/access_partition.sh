#! /bin/sh

module unload deeplearning # otherwise we get some gcc version issues

srun -p dl --gpus-per-node p100:1 --time=09:00:00 --pty bash
#srun -p dl --time=03:00:00 --pty bash