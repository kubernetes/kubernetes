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

# Format and mount the disk, create directories on it for all of the master's
# persistent data, and link them to where they're used.

echo "Waiting for master pd to be attached"
attempt=0
while true; do
  echo Attempt "$(($attempt+1))" to check for /dev/xvdb
  if [[ -e /dev/xvdb ]]; then
    echo "Found /dev/xvdb"
    break
  fi
  attempt=$(($attempt+1))
  sleep 1
done

# Mount Master Persistent Disk
echo "Mounting master-pd"
mkdir -p /mnt/master-pd
mkfs -t ext4 /dev/xvdb
echo "/dev/xvdb  /mnt/master-pd  ext4  noatime  0 0" >> /etc/fstab
mount /mnt/master-pd

# Contains all the data stored in etcd
mkdir -m 700 -p /mnt/master-pd/var/etcd
# Contains the dynamically generated apiserver auth certs and keys
mkdir -p /mnt/master-pd/srv/kubernetes
# Contains the cluster's initial config parameters and auth tokens
mkdir -p /mnt/master-pd/srv/salt-overlay
# Directory for kube-apiserver to store SSH key (if necessary)
mkdir -p /mnt/master-pd/srv/sshproxy

ln -s -f /mnt/master-pd/var/etcd /var/etcd
ln -s -f /mnt/master-pd/srv/kubernetes /srv/kubernetes
ln -s -f /mnt/master-pd/srv/sshproxy /srv/sshproxy
ln -s -f /mnt/master-pd/srv/salt-overlay /srv/salt-overlay

# This is a bit of a hack to get around the fact that salt has to run after the
# PD and mounted directory are already set up. We can't give ownership of the
# directory to etcd until the etcd user and group exist, but they don't exist
# until salt runs if we don't create them here. We could alternatively make the
# permissions on the directory more permissive, but this seems less bad.
if ! id etcd &>/dev/null; then
  useradd -s /sbin/nologin -d /var/etcd etcd
fi
chown -R etcd /mnt/master-pd/var/etcd
chgrp -R etcd /mnt/master-pd/var/etcd
