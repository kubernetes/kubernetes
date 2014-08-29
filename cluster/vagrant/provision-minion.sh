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

# exit on any error
set -e
source $(dirname $0)/provision-config.sh

MINION_IP=$4

# make sure each minion has an entry in hosts file for master
if [ ! "$(cat /etc/hosts | grep $MASTER_NAME)" ]; then
  echo "Adding host entry for $MASTER_NAME"
  echo "$MASTER_IP $MASTER_NAME" >> /etc/hosts
fi

# Let the minion know who its master is
mkdir -p /etc/salt/minion.d
echo "master: $MASTER_NAME" > /etc/salt/minion.d/master.conf

# Our minions will have a pool role to distinguish them from the master.
cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  minion_ip: $MINION_IP
  etcd_servers: $MASTER_IP
  roles:
    - kubernetes-pool
  cbr-cidr: $MINION_IP_RANGE
EOF

# we will run provision to update code each time we test, so we do not want to do salt install each time
if ! which salt-minion >/dev/null 2>&1; then
  # Install Salt
  curl -sS -L --connect-timeout 20 --retry 6 --retry-delay 10 https://bootstrap.saltstack.com | sh -s
fi

# run the networking setup
$(dirname $0)/provision-network.sh $@
