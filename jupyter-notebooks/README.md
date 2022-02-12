## How to run Jupyter Notebooks on Carbonate DL nodes
Because the DL node and a local machine are blind to each other, we can't directly port-forward between them. This sets up two hops of remote forwarding so that a port on an accessible server (h2.carbonate) acts as a proxy.

### Forwarding of Requests
`localhost:5500 -> h2.carbonate.uits.iu.edu:1133 -> deeplearningnode:1133`

- The Jupyter server is started on port `1133` on the DL node
- The DL node asks `h2:1133` to forward it any requests it receives. 
- Then the local machine asks if it can forward its requests to `h2:1133`. 
- In the end, a GET request for `localhost:5500` on the laptop (hopefully...) is forwarded to the notebook server running on the DL node.

> Note: use a port other than 1133.

### Instructions 
0. ssh to carbonate.uits.iu.edu and `cd` to this directory

    a. Assign your username and desired (arbitrary) port through the variables in `jupyter-batch.sh`

    b. (Optional) Modify `access_partition.sh` to change the type of partition or duration

1. Run `. ./access_partition.sh` on one of Carbonate's login nodes (h1/h2) to get access to a partition

2. Run `. ./jupyter-batch.sh` on that partition to start the server and do some port forwarding; you may have to authenticate with h2.carbonate

3. On your local machine, run `ssh -L 5500:localhost:1133 myusername@h2.carbonate.uits.iu.edu`, assuming `1133` is the port you chose in step 0a.

Now you should be able to see it in `localhost:5500` on your local machine (or sub whatever other port you want here). The access token for the notebook should be printed in the output of step 2.