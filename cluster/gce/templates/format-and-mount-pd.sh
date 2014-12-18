#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# Formats and mounts a persistent disk to store the persistent data on the
# master -- etcd's data and the security certs/keys.

device_info=$(ls -l /dev/disk/by-id/google-master-pd)
relative_path=${device_info##* }
device_path="/dev/disk/by-id/${relative_path}"

# Format and mount the disk to the directory used by etcd.
mkdir -p /mnt/master-pd
/usr/share/google/safe_format_and_mount -m "mkfs.ext4 -F" "${device_path}" /mnt/master-pd
mkdir -m 700 -p /mnt/master-pd/var/etcd
mkdir -p /mnt/master-pd/srv/kubernetes
ln -s /mnt/master-pd/var/etcd /var/etcd
ln -s /mnt/master-pd/srv/kubernetes /srv/kubernetes

# This is a bit of a hack to get around the fact that salt has to run after the
# PD and mounted directory are already set up. We can't give ownership of the
# directory to etcd until the etcd user and group exist, but they don't exist
# until salt runs if we don't create them here. We could alternatively make the
# permissions on the directory more permissive, but this seems less bad.
useradd -s /sbin/nologin -d /var/etcd etcd
chown etcd /mnt/master-pd/var/etcd
chgrp etcd /mnt/master-pd/var/etcd
