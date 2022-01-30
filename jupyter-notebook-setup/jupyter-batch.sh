#!/bin/bash

## ignore these lines, ssh requires authentication so this must be interactive
#SBATCH -J jupyter_notebook
#SBATCH -p general
#SBATCH -o jupyter_output_%j.txt
#SBATCH -e jupyter_error_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skarukas@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

#Load any modules that your program needs
module load deeplearning

# port forward requests on port 1133 on carbonate to port 1133 here
ssh -fN -R 1133:localhost:1133 skarukas@h2.carbonate.uits.iu.edu

# run notebook on port 1133
jupyter notebook --no-browser --port=1133

# make sure local port forwarding is being used from h2.carbonate too
