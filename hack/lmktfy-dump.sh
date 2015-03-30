#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

# This script dumps debugging state of currently running LMKTFY cluster.

LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/..

echo "lmktfy-dump.sh: Getting version..."
"${LMKTFY_ROOT}/cluster/lmktfyctl.sh" version
echo

echo "lmktfy-dump.sh: Getting resources..."
"${LMKTFY_ROOT}/cluster/lmktfyctl.sh" get nodes,pods,rc,services,events -o json
echo

source "${LMKTFY_ROOT}/cluster/lmktfy-env.sh"
source "${LMKTFY_ROOT}/cluster/${LMKTFYRNETES_PROVIDER}/util.sh"

detect-project &> /dev/null

echo "lmktfy-dump.sh: Getting docker statuses on all nodes..."
ALL_NODES=(${MINION_NAMES[*]} ${MASTER_NAME})
for NODE in ${ALL_NODES[*]}; do 
  echo "lmktfy-dump.sh: Node $NODE:"
  ssh-to-node "${NODE}" '
    sudo docker ps -a
    sudo docker images
  '
done
echo

echo "lmktfy-dump.sh: Getting boundpods from etcd..."
ssh-to-node "${MASTER_NAME}" '
  ETCD_SERVER=$(hostname -i):4001 
  for DIR in $(etcdctl -C $ETCD_SERVER ls /registry/nodes); do 
    echo "lmktfy-dump.sh: Dir $DIR:"
    etcdctl -C $ETCD_SERVER get $DIR/boundpods
  done 
'
