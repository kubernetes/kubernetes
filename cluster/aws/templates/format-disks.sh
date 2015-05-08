#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Discover all the ephemeral disks

block_devices=()

ephemeral_devices=$(curl --silent http://169.254.169.254/2014-11-05/meta-data/block-device-mapping/ | grep ephemeral)
for ephemeral_device in $ephemeral_devices; do
  echo "Checking ephemeral device: ${ephemeral_device}"
  aws_device=$(curl --silent http://169.254.169.254/2014-11-05/meta-data/block-device-mapping/${ephemeral_device})

  device_path=""
  if [ -b /dev/$aws_device ]; then
    device_path="/dev/$aws_device"
  else
    # Check for the xvd-style name
    xvd_style=$(echo $aws_device | sed "s/sd/xvd/")
    if [ -b /dev/$xvd_style ]; then
      device_path="/dev/$xvd_style"
    fi
  fi

  if [[ -z ${device_path} ]]; then
    echo "  Could not find disk: ${ephemeral_device}@${aws_device}"
  else
    echo "  Detected ephemeral disk: ${ephemeral_device}@${device_path}"
    block_devices+=(${device_path})
  fi
done

# Format the ephemeral disks
if [[ ${#block_devices[@]} == 0 ]]; then
  echo "No ephemeral block devices found"
else
  echo "Block devices: ${block_devices}"

  apt-get install --yes btrfs-tools

  if [[ ${#block_devices[@]} == 1 ]]; then
    echo "One ephemeral block device found; formatting with btrfs"
    mkfs.btrfs -f ${block_devices[0]}
  else
    echo "Found multiple ephemeral block devices, formatting with btrfs as RAID-0"
    mkfs.btrfs -f --data raid0 ${block_devices[@]}
  fi
  mount -t btrfs ${block_devices[0]} /mnt

  # Move docker to /mnt if we have it
  if [[ -d /var/lib/docker ]]; then
    mv /var/lib/docker /mnt/
  fi
  mkdir -p /mnt/docker
  ln -s /mnt/docker /var/lib/docker
  DOCKER_ROOT="/mnt/docker"
  DOCKER_OPTS="${DOCKER_OPTS} -g /mnt/docker"

  # Move /var/lib/kubelet to /mnt if we have it
  # (the backing for empty-dir volumes can use a lot of space!)
  if [[ -d /var/lib/kubelet ]]; then
    mv /var/lib/kubelet /mnt/
  fi
  mkdir -p /mnt/kubelet
  ln -s /mnt/kubelet /var/lib/kubelet
fi

