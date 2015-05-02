#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# Setup hosts file to support ping by hostname to master
if [ ! "$(cat /etc/hosts | grep $MASTER_NAME)" ]; then
  echo "Adding $MASTER_NAME to hosts file"
  echo "$MASTER_IP $MASTER_NAME" >> /etc/hosts
fi

# Setup hosts file to support ping by hostname to each minion in the cluster
for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
  minion=${MINION_NAMES[$i]}
  ip=${MINION_IPS[$i]}
  if [ ! "$(cat /etc/hosts | grep $minion)" ]; then
    echo "Adding $minion to hosts file"
    echo "$ip $minion" >> /etc/hosts
  fi
done

# Let the minion know who its master is
# Recover the salt-minion if the salt-master network changes
## auth_timeout - how long we want to wait for a time out
## auth_tries - how many times we will retry before restarting salt-minion
## auth_safemode - if our cert is rejected, we will restart salt minion
## ping_interval - restart the minion if we cannot ping the master after 1 minute
## random_reauth_delay - wait 0-3 seconds when reauthenticating
## recon_default - how long to wait before reconnecting
## recon_max - how long you will wait upper bound
## state_aggregrate - try to do a single yum command to install all referenced packages where possible at once, should improve startup times
##
mkdir -p /etc/salt/minion.d
cat <<EOF >/etc/salt/minion.d/master.conf
master: '$(echo "$MASTER_NAME" | sed -e "s/'/''/g")'
auth_timeout: 10
auth_tries: 2
auth_safemode: True
ping_interval: 1
random_reauth_delay: 3
state_aggregrate:
  - pkg
EOF

cat <<EOF >/etc/salt/minion.d/log-level-debug.conf
log_level: debug
log_level_logfile: debug
EOF

# Our minions will have a pool role to distinguish them from the master.
cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  cloud: vagrant
  network_mode: openvswitch
  node_ip: '$(echo "$MINION_IP" | sed -e "s/'/''/g")'
  api_servers: '$(echo "$MASTER_IP" | sed -e "s/'/''/g")'
  networkInterfaceName: eth1
  roles:
    - kubernetes-pool
  cbr-cidr: '$(echo "$CONTAINER_SUBNET" | sed -e "s/'/''/g")'
  hostname_override: '$(echo "$MINION_IP" | sed -e "s/'/''/g")'
EOF

# we will run provision to update code each time we test, so we do not want to do salt install each time
if ! which salt-minion >/dev/null 2>&1; then
  # Install Salt
  curl -sS -L --connect-timeout 20 --retry 6 --retry-delay 10 https://bootstrap.saltstack.com | sh -s
else
  # Sometimes the minion gets wedged when it comes up along with the master.
  # Restarting it here un-wedges it.
  systemctl restart salt-minion.service
fi
