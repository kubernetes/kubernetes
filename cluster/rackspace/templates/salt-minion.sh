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

# Prepopulate the name of the Master
mkdir -p /etc/salt/minion.d
echo "master: $MASTER_NAME" > /etc/salt/minion.d/master.conf

# Turn on debugging for salt-minion
# echo "DAEMON_ARGS=\"\$DAEMON_ARGS --log-file-level=debug\"" > /etc/default/salt-minion

MINION_IP=$(ip -f inet a sh dev eth2 | grep -i inet | awk '{print $2}' | cut -d / -f 1)
# Our minions will have a pool role to distinguish them from the master.
cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  roles:
    - kubernetes-pool
  cbr-cidr: $MINION_IP_RANGE
  minion_ip: $MINION_IP
EOF

#Move all of this to salt
apt-get update
apt-get install bridge-utils -y
brctl addbr cbr0
ip link set dev cbr0 up
#for loop to add routes of other minions
for (( i=1; i<=${NUM_MINIONS[@]}; i++)); do
 ip r a 10.240.$i.0/24 dev cbr0
done
ip link add vxlan42 type vxlan id 42 group 239.0.0.42 dev eth2
brctl addif cbr0 vxlan42
# Install Salt
#
# We specify -X to avoid a race condition that can cause minion failure to
# install.  See https://github.com/saltstack/salt-bootstrap/issues/270
curl -L http://bootstrap.saltstack.com | sh -s -- -X
ip link set vxlan42 up
