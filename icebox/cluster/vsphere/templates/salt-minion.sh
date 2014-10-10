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

# Use other Debian mirror
sed -i -e "s/http.us.debian.org/mirrors.kernel.org/" /etc/apt/sources.list

# Resolve hostname of master
if ! grep -q $MASTER_NAME /etc/hosts; then
  echo "Adding host entry for $MASTER_NAME"
  echo "$MASTER_IP $MASTER_NAME" >> /etc/hosts
fi

# Prepopulate the name of the Master
mkdir -p /etc/salt/minion.d
echo "master: $MASTER_NAME" > /etc/salt/minion.d/master.conf

# Turn on debugging for salt-minion
# echo "DAEMON_ARGS=\"\$DAEMON_ARGS --log-file-level=debug\"" > /etc/default/salt-minion

# Our minions will have a pool role to distinguish them from the master.
#
# Setting the "minion_ip" here causes the kubelet to use its IP for
# identification instead of its hostname.
#
cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  minion_ip: $(ip route get 1.1.1.1 | awk '{print $7}')
  roles:
    - kubernetes-pool
    - kubernetes-pool-vsphere
  cbr-cidr: $MINION_IP_RANGE
EOF

# Install Salt
#
# We specify -X to avoid a race condition that can cause minion failure to
# install.  See https://github.com/saltstack/salt-bootstrap/issues/270
if [ ! -x /etc/init.d/salt-minion ]; then
  wget -q -O - https://bootstrap.saltstack.com | sh -s -- -X
else
  /etc/init.d/salt-minion restart
fi
