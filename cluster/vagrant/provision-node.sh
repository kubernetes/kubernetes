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

set -o errexit
set -o nounset
set -o pipefail

# Set the host name explicitly
# See: https://github.com/mitchellh/vagrant/issues/2430
hostnamectl set-hostname ${NODE_NAME}
if_to_edit=""

if [[ "$(grep 'VERSION_ID' /etc/os-release)" =~ ^VERSION_ID=23 ]]; then
  # Disable network interface being managed by Network Manager (needed for Fedora 21+)
  NETWORK_CONF_PATH=/etc/sysconfig/network-scripts/
  if_to_edit=$( find ${NETWORK_CONF_PATH}ifcfg-* | xargs grep -l VAGRANT-BEGIN )
  for if_conf in ${if_to_edit}; do
    grep -q ^NM_CONTROLLED= ${if_conf} || echo 'NM_CONTROLLED=no' >> ${if_conf}
    sed -i 's/#^NM_CONTROLLED=.*/NM_CONTROLLED=no/' ${if_conf}
  done;
  systemctl restart network
fi

# needed for vsphere support
# handle the case when no 'VAGRANT-BEGIN' comment was defined in network-scripts
# set the NETWORK_IF_NAME to have a default value in such case
NETWORK_IF_NAME=`echo ${if_to_edit} | awk -F- '{ print $3 }'`
if [[ -z "$NETWORK_IF_NAME" ]]; then
  NETWORK_IF_NAME=${DEFAULT_NETWORK_IF_NAME}
fi

# Setup hosts file to support ping by hostname to master
if [ ! "$(cat /etc/hosts | grep $MASTER_NAME)" ]; then
  echo "Adding $MASTER_NAME to hosts file"
  echo "$MASTER_IP $MASTER_NAME" >> /etc/hosts
fi
echo "$NODE_IP $NODE_NAME" >> /etc/hosts

# Setup hosts file to support ping by hostname to each node in the cluster
for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
  node=${NODE_NAMES[$i]}
  ip=${NODE_IPS[$i]}
  if [ ! "$(cat /etc/hosts | grep $node)" ]; then
    echo "Adding $node to hosts file"
    echo "$ip $node" >> /etc/hosts
  fi
done

prepare-package-manager

# Configure network
if [ "${NETWORK_PROVIDER}" != "kubenet" ]; then
  provision-network-node
fi

write-salt-config kubernetes-pool

# Generate kubelet and kube-proxy auth file(kubeconfig) if there is not an existing one
known_kubeconfig_file="/srv/salt-overlay/salt/kubelet/kubeconfig"
if [[ ! -f "${known_kubeconfig_file}" ]]; then
  create-salt-kubelet-auth
  create-salt-kubeproxy-auth
else
  # stop kubelet, let salt start it later
  systemctl stop kubelet
fi

install-salt

run-salt

