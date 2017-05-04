#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda -y
  apt-get install linux-headers-$(uname -r) -y
fi
## Pre-loads kernel modules
nvidia-smi
