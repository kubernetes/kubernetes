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

# provision-network-minion configures flannel on the minion
function provision-network-minion {

  echo "Provisioning network on minion"

  FLANNEL_ETCD_URL="http://${MASTER_IP}:4379"

  # Install flannel for overlay
  if ! which flanneld >/dev/null 2>&1; then

    yum install -y flannel

    # Configure local daemon to speak to master
    NETWORK_CONF_PATH=/etc/sysconfig/network-scripts/
    if_to_edit=$( find ${NETWORK_CONF_PATH}ifcfg-* | xargs grep -l VAGRANT-BEGIN )
    NETWORK_IF_NAME=`echo ${if_to_edit} | awk -F- '{ print $3 }'`
    cat <<EOF >/etc/sysconfig/flanneld
FLANNEL_ETCD="${FLANNEL_ETCD_URL}"
FLANNEL_ETCD_KEY="/coreos.com/network"
FLANNEL_OPTIONS="-iface=${NETWORK_IF_NAME}"
EOF

    # Start flannel
    systemctl enable flanneld
    systemctl start flanneld
  fi

  echo "Network configuration verified"
}
