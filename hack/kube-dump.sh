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

# This script dumps debugging state of currently running Kubernetes cluster.

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

echo "kube-dump.sh: Getting version..."
"${KUBE_ROOT}/cluster/kubectl.sh" version
echo

echo "kube-dump.sh: Getting resources..."
"${KUBE_ROOT}/cluster/kubectl.sh" get nodes,pods,rc,services,events -o json
echo

source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/${KUBERNETES_PROVIDER}/util.sh"

detect-project &> /dev/null

echo "kube-dump.sh: Getting docker statuses on all nodes..."
ALL_NODES=(${MINION_NAMES[*]} ${MASTER_NAME})
for NODE in ${ALL_NODES[*]}; do 
  echo "kube-dump.sh: Node $NODE:"
  ssh-to-node "${NODE}" '
    sudo docker ps -a
    sudo docker images
  '
done
echo

echo "kube-dump.sh: Getting boundpods from etcd..."
ssh-to-node "${MASTER_NAME}" '
  ETCD_SERVER=$(hostname -i):4001 
  for DIR in $(etcdctl -C $ETCD_SERVER ls /registry/nodes); do 
    echo "kube-dump.sh: Dir $DIR:"
    etcdctl -C $ETCD_SERVER get $DIR/boundpods
  done 
'
