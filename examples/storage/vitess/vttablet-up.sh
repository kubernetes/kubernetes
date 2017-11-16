#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

# This is an example script that creates a vttablet deployment.

set -e

script_root=`dirname "${BASH_SOURCE}"`
source $script_root/env.sh

# Create the pods for shard-0
CELLS=${CELLS:-'test'}
keyspace='test_keyspace'
SHARDS=${SHARDS:-'0'}
TABLETS_PER_SHARD=${TABLETS_PER_SHARD:-5}
port=15002
grpc_port=16002
UID_BASE=${UID_BASE:-100}
VTTABLET_TEMPLATE=${VTTABLET_TEMPLATE:-'vttablet-pod-template.yaml'}
RDONLY_COUNT=${RDONLY_COUNT:-2}

uid_base=$UID_BASE
for shard in $(echo $SHARDS | tr "," " "); do
  cell_index=0
  for cell in `echo $CELLS | tr ',' ' '`; do
    echo "Creating $keyspace.shard-$shard pods in cell $CELL..."
    for uid_index in `seq 0 $(($TABLETS_PER_SHARD-1))`; do
      uid=$[$uid_base + $uid_index + $cell_index]
      printf -v alias '%s-%010d' $cell $uid
      printf -v tablet_subdir 'vt_%010d' $uid

      echo "Creating pod for tablet $alias..."

      # Add xx to beginning or end if there is a dash.  K8s does not allow for
      # leading or trailing dashes for labels
      shard_label=`echo $shard | sed s'/[-]$/-xx/' | sed s'/^-/xx-/'`

      tablet_type=replica
      if [ $uid_index -gt $(($TABLETS_PER_SHARD-$RDONLY_COUNT-1)) ]; then
        tablet_type=rdonly
      fi

      # Expand template variables
      sed_script=""
      for var in alias cell uid keyspace shard shard_label port grpc_port tablet_subdir tablet_type backup_flags; do
        sed_script+="s,{{$var}},${!var},g;"
      done

      # Instantiate template and send to kubectl.
      cat $VTTABLET_TEMPLATE | sed -e "$sed_script" | $KUBECTL create -f -
    done
    let cell_index=cell_index+100000000
  done
  let uid_base=uid_base+100
done
