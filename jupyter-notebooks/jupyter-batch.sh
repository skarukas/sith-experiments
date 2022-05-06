#!/bin/bash

#Load any modules that your program needs
module load deeplearning

PORT=1133
USERNAME=skarukas

# run notebook on port 1133
cd ..
#jupyter notebook \
#  --no-browser \
#  --port=$PORT \
#  --NotebookApp.allow_origin='https://colab.research.google.com' \
#  --NotebookApp.port_retries=0

# port forward requests on port 1133 on carbonate to port 1133 here
ssh -fN -R $PORT:localhost:$PORT ${USERNAME}@h2.carbonate.uits.iu.edu
jupyter notebook --no-browser --port=$PORT

# make sure local port forwarding is being used from h2.carbonate too
