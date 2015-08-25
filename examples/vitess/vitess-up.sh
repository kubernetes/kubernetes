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

# This is an example script that creates a fully functional vitess cluster.
# It performs the following steps:
#   - Create etcd clusters
#   - Create vtctld pod
#   - Create vttablet pods
#   - Perform vtctl initialization:
#       SetKeyspaceShardingInfo, Rebuild Keyspace, Reparent Shard, Apply Schema
#   - Create vtgate pods

script_root=`dirname "${BASH_SOURCE}"`
source $script_root/env.sh

cells=`echo $CELLS | tr ',' ' '`
num_cells=`echo $cells | wc -w`

function update_spinner_value () {
  spinner='-\|/'
  cur_spinner=${spinner:$(($1%${#spinner})):1}
}

function wait_for_running_tasks () {
  # This function waits for pods to be in the "Running" state
  # 1. task_name: Name that the desired task begins with
  # 2. num_tasks: Number of tasks to wait for
  # Returns:
  #   0 if successful, -1 if timed out
  task_name=$1
  num_tasks=$2
  counter=0

  echo "Waiting for ${num_tasks}x $task_name to enter state Running"

  while [ $counter -lt $MAX_TASK_WAIT_RETRIES ]; do
    # Get status column of pods with name starting with $task_name,
    # count how many are in state Running
    num_running=`$KUBECTL get pods | grep ^$task_name | grep Running | wc -l`

    echo -en "\r$task_name: $num_running out of $num_tasks in state Running..."
    if [ $num_running -eq $num_tasks ]
    then
      echo Complete
      return 0
    fi
    update_spinner_value $counter
    echo -n $cur_spinner
    let counter=counter+1
    sleep 1
  done
  echo Timed out
  return -1
}

if [ -z "$GOPATH" ]; then
  echo "ERROR: GOPATH undefined, can't obtain vtctlclient"
  exit -1
fi

export KUBECTL='kubectl'

echo "Downloading and installing vtctlclient..."
go get github.com/youtube/vitess/go/cmd/vtctlclient
num_shards=`echo $SHARDS | tr "," " " | wc -w`
total_tablet_count=$(($num_shards*$TABLETS_PER_SHARD*$num_cells))
vtgate_count=$VTGATE_COUNT
if [ $vtgate_count -eq 0 ]; then
  vtgate_count=$(($total_tablet_count/4>3?$total_tablet_count/4:3))
fi

echo "****************************"
echo "*Creating vitess cluster:"
echo "*  Shards: $SHARDS"
echo "*  Tablets per shard: $TABLETS_PER_SHARD"
echo "*  Rdonly per shard: $RDONLY_COUNT"
echo "*  VTGate count: $vtgate_count"
echo "*  Cells: $cells"
echo "****************************"

echo 'Running etcd-up.sh' && CELLS=$CELLS ./etcd-up.sh
wait_for_running_tasks etcd-global 3
for cell in $cells; do
  wait_for_running_tasks etcd-$cell 3
done

echo 'Running vtctld-up.sh' && ./vtctld-up.sh
echo 'Running vttablet-up.sh' && CELLS=$CELLS ./vttablet-up.sh
echo 'Running vtgate-up.sh' && ./vtgate-up.sh

wait_for_running_tasks vtctld 1
wait_for_running_tasks vttablet $total_tablet_count
wait_for_running_tasks vtgate $vtgate_count

vtctld_port=30000
vtctld_ip=`kubectl get -o yaml nodes | grep 'type: ExternalIP' -B 1 | head -1 | awk '{print $NF}'`
vtctl_server="$vtctld_ip:$vtctld_port"
kvtctl="$GOPATH/bin/vtctlclient -server $vtctl_server"

echo Waiting for tablets to be visible in the topology
counter=0
while [ $counter -lt $MAX_VTTABLET_TOPO_WAIT_RETRIES ]; do
  num_tablets=0
  for cell in $cells; do
    num_tablets=$(($num_tablets+`$kvtctl ListAllTablets $cell | wc -l`))
  done
  echo -en "\r$num_tablets out of $total_tablet_count in topology..."
  if [ $num_tablets -eq $total_tablet_count ]
  then
    echo Complete
    break
  fi
  update_spinner_value $counter
  echo -n $cur_spinner
  let counter=counter+1
  sleep 1
  if [ $counter -eq $MAX_VTTABLET_TOPO_WAIT_RETRIES ]
  then
    echo Timed out
  fi
done

# split_shard_count = num_shards for sharded keyspace, 0 for unsharded
split_shard_count=$num_shards
if [ $split_shard_count -eq 1 ]; then
  split_shard_count=0
fi

echo -n Setting Keyspace Sharding Info...
$kvtctl SetKeyspaceShardingInfo -force -split_shard_count $split_shard_count test_keyspace keyspace_id uint64
echo Done
echo -n Rebuilding Keyspace Graph...
$kvtctl RebuildKeyspaceGraph test_keyspace
echo Done
echo -n Reparenting...
shard_num=1
for shard in $(echo $SHARDS | tr "," " "); do
  $kvtctl InitShardMaster -force test_keyspace/$shard `echo $cells | awk '{print $1}'`-0000000${shard_num}00
  let shard_num=shard_num+1
done
echo Done
echo -n Applying Schema...
$kvtctl ApplySchema -sql "$(cat create_test_table.sql)" test_keyspace
echo Done

echo "****************************"
echo "* Complete!"
echo "* Use the following line to make an alias to kvtctl:"
echo "* alias kvtctl='\$GOPATH/bin/vtctlclient -server $vtctl_server'"
echo "* See the vtctld UI at: http://${vtctl_server}"
echo "****************************"
