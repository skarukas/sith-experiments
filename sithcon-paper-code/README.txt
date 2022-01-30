<h2 align="center">
A deep convolutional neural network that is invariant to time rescaling
</h2>

# Overview

Here is the code needed to replicate the results found in the 2022 AAAI submission for "A deep convolutional neural network that is invariant to time rescaling. 

In "SITHCon", you will find 2 files, "isith.py" and "sithcon.py". The implementation of the Scale Invariant Temporal History is located within the first, and the implementation for the pytorch network, SITHCon, is located in the second. 

In "experiments" you will find a separate folder for each experiment containing jupyter notebooks that outline how we ran the experiments on both a TCN model and on a SITHCon model. These experiments include AudioMNIST, Morse Decoder, and Morse Addition. 

In each experiment folder, there will be files pertaining to the SITHCon network and files pertaining to the TCN network. There will also be a *visulaization* file where all of the figures from the paper are made. 
