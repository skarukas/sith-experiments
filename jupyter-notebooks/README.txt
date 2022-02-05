0. ssh to carbonate.uits.iu.edu
1. Run . ./access-partition.sh on h2.carbonate to get access to a partition
2. Run . ./jupyter-batch.sh on that partition to start the server and do a ton of port forwarding
3. Run ssh -L 5500:localhost:1133 skarukas@h2.carbonate.uits.iu.edu on your local machine

Now you should be able to see it in localhost:5500 on your local machine
