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

# This is an example script that tears down the vttablet pods started by
# vttablet-up.sh.

set -e

script_root=`dirname "${BASH_SOURCE}"`
source $script_root/env.sh

server=$(get_vtctld_addr)

# Delete the pods for all shards
CELLS=${CELLS:-'test'}
keyspace='test_keyspace'
SHARDS=${SHARDS:-'0'}
TABLETS_PER_SHARD=${TABLETS_PER_SHARD:-5}
UID_BASE=${UID_BASE:-100}

num_shards=`echo $SHARDS | tr "," " " | wc -w`
uid_base=$UID_BASE

for shard in `seq 1 $num_shards`; do
  cell_index=0
  for cell in `echo $CELLS | tr "," " "`; do
    for uid_index in `seq 0 $(($TABLETS_PER_SHARD-1))`; do
      uid=$[$uid_base + $uid_index + $cell_index]
      printf -v alias '%s-%010d' $cell $uid

      echo "Deleting pod for tablet $alias..."
      $KUBECTL delete pod vttablet-$uid
    done
    let cell_index=cell_index+100000000
  done
  let uid_base=uid_base+100
done

