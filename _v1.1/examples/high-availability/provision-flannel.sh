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

set -o errexit
set -o nounset
set -o pipefail

function setup_flannel {
    yum install -y flannel

    ### Write this k/v to etcd.  Flannel will grab it to setup its networking.
    curl --silent -s -L http://kube0.ha:2379/v2/keys/coreos.com/network/config -XPUT -d value='{"Network": "172.31.255.0/24", "SubnetLen": 27, "Backend": {"Type": "vxlan"}}'

### Write flannel etcd file
cat >> /etc/sysconfig/flanneld << EOF
FLANNEL_ETCD="http://kube0.ha:2379"
FLANNEL_ETCD_KEY="/coreos.com/network"
FLANNEL_OPTIONS="--iface=eth1"
EOF
}

echo "now setting up flannel.  Assuming etcd is online!"
setup_flannel
sudo service flanneld restart
sudo ip link delete docker0
sudo service docker restart

### This should restart etcd and all the others
### The pods will now have a default ip for docker containers which
### runs inside of the kube network.
sudo systemctl restart kubelet
