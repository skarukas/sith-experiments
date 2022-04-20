#! /bin/sh

module unload deeplearning # otherwise we get some gcc version issues

srun -p general --time=01:00:00 --pty bash
#srun -p dl --time=03:00:00 --pty bash