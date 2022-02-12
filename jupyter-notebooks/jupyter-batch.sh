#!/bin/bash

#Load any modules that your program needs
module load deeplearning

PORT=1133

# port forward requests on port 1133 on carbonate to port 1133 here
ssh -fN -R $PORT:localhost:$PORT skarukas@h2.carbonate.uits.iu.edu

# run notebook on port 1133
cd ..
jupyter notebook --no-browser --port=$PORT

# make sure local port forwarding is being used from h2.carbonate too
