#!/bin/bash

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
sh ./ceph.conf.sh `hostname -i`

# Configure and start ceph-mon
sh ./mon.sh `hostname -i`

# Configure and start 2x ceph-osd
mkdir -p /var/lib/ceph/osd/ceph-0 /var/lib/ceph/osd/ceph-1
sh ./osd.sh 0
sh ./osd.sh 1

# Prepare a RBD volume
# NOTE: we need Ceph kernel modules on the host!
rbd import block foo

echo "Ceph is ready"

# Wait forever
while true; do
    sleep 10
done
