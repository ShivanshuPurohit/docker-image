#!/bin/bash

# Start SSH service
service ssh start

# Fuse the kernels
# python /home/root/gpt-neox/megatron/fused_kernels/setup.py install

# For EFA
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> .deepspeed_env

# Loop to prevent pod shutdown
echo '------------------Ready-------------------'
while true; do 
  sleep 30; 
done;
