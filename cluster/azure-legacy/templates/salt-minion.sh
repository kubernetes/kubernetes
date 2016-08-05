#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

mkdir -p /etc/openvpn
umask=$(umask)
umask 0066
echo "$CA_CRT" > /etc/openvpn/ca.crt
echo "$CLIENT_CRT" > /etc/openvpn/client.crt
echo "$CLIENT_KEY" > /etc/openvpn/client.key
umask $umask

# Prepopulate the name of the Master
mkdir -p /etc/salt/minion.d
echo "master: $MASTER_NAME" > /etc/salt/minion.d/master.conf

cat <<EOF >/etc/salt/minion.d/log-level-debug.conf
log_level: debug
log_level_logfile: debug
EOF

hostnamef=$(uname -n)
apt-get install -y ipcalc
netmask=$(ipcalc $MINION_IP_RANGE | grep Netmask | awk '{ print $2 }')
network=$(ipcalc $MINION_IP_RANGE | grep Address | awk '{ print $2 }')
cbrstring="$network $netmask"

# Our minions will have a pool role to distinguish them from the master.
cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  roles:
    - kubernetes-pool
  cbr-cidr: $MINION_IP_RANGE
  cloud: azure-legacy
  hostnamef: $hostnamef
  cbr-string: $cbrstring
EOF

if [[ -n "${DOCKER_OPTS}" ]]; then
  cat <<EOF >>/etc/salt/minion.d/grains.conf
  docker_opts: '$(echo "$DOCKER_OPTS" | sed -e "s/'/''/g")'
EOF
fi

if [[ -n "${DOCKER_ROOT}" ]]; then
  cat <<EOF >>/etc/salt/minion.d/grains.conf
  docker_root: '$(echo "$DOCKER_ROOT" | sed -e "s/'/''/g")'
EOF
fi

if [[ -n "${KUBELET_ROOT}" ]]; then
  cat <<EOF >>/etc/salt/minion.d/grains.conf
  kubelet_root: '$(echo "$KUBELET_ROOT" | sed -e "s/'/''/g")'
EOF
fi

install-salt

# Wait a few minutes and trigger another Salt run to better recover from
# any transient errors.
echo "Sleeping 180"
sleep 180
salt-call state.highstate || true
