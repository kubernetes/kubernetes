#!/usr/bin/env bash

# Copyright 2015 The Kubernetes Authors.
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

#
# Bootstraps a CEPH server.
# It creates two OSDs on local machine, creates RBD pool there
# and imports 'block' device there.
#
# We must create fresh OSDs and filesystem here, because shipping it
# in a container would increase the image by ~300MB.
#


# Create /etc/ceph/ceph.conf
sh ./ceph.conf.sh "$(hostname -i)"

# Configure and start ceph-mon
sh ./mon.sh "$(hostname -i)"

# Configure and start 2x ceph-osd
mkdir -p /var/lib/ceph/osd/ceph-0 /var/lib/ceph/osd/ceph-1
sh ./osd.sh 0
sh ./osd.sh 1

# Configure and start cephfs metadata server
sh ./mds.sh

# Prepare a RBD volume "foo" (only with layering feature, the others may
# require newer clients).
# NOTE: we need Ceph kernel modules on the host that runs the client!
rbd import --image-feature layering block foo

# Prepare a cephfs volume
ceph osd pool create cephfs_data 4
ceph osd pool create cephfs_metadata 4
ceph fs new cephfs cephfs_metadata cephfs_data
# Put index.html into the volume
# It takes a while until the volume created above is mountable,
# 1 second is usually enough, but try indefinetily.
sleep 1
while ! ceph-fuse -m "$(hostname -i):6789" /mnt; do
    echo "Waiting for cephfs to be up"
    sleep 1
done
echo "Hello Ceph!" > /mnt/index.html
chmod 644 /mnt/index.html
umount /mnt

echo "Ceph is ready"

# Wait forever
while true; do
    sleep 10
done
