#!/bin/bash

# Copyright 2017 Google Inc.
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
# 1. Create etcd clusters
# 2. Create vtctld clusters
# 3. Forward vtctld port
# 4. Create vttablet clusters
# 5. Perform vtctl initialization:
#      SetKeyspaceShardingInfo, Rebuild Keyspace, Reparent Shard, Apply Schema
# 6. Create vtgate clusters
# 7. Forward vtgate port

# Customizable parameters
VITESS_NAME=${VITESS_NAME:-'vitess'}
SHARDS=${SHARDS:-'-80,80-'}
TABLETS_PER_SHARD=${TABLETS_PER_SHARD:-3}
RDONLY_COUNT=${RDONLY_COUNT:-0}
MAX_TASK_WAIT_RETRIES=${MAX_TASK_WAIT_RETRIES:-300}
MAX_VTTABLET_TOPO_WAIT_RETRIES=${MAX_VTTABLET_TOPO_WAIT_RETRIES:-180}
VTTABLET_TEMPLATE=${VTTABLET_TEMPLATE:-'vttablet-pod-benchmarking-template.yaml'}
VTGATE_TEMPLATE=${VTGATE_TEMPLATE:-'vtgate-controller-benchmarking-template.yaml'}
VTGATE_COUNT=${VTGATE_COUNT:-0}
VTDATAROOT_VOLUME=${VTDATAROOT_VOLUME:-''}
CELLS=${CELLS:-'test'}
KEYSPACE=${KEYSPACE:-'test_keyspace'}
TEST_MODE=${TEST_MODE:-'0'}

cells=`echo $CELLS | tr ',' ' '`
num_cells=`echo $cells | wc -w`

num_shards=`echo $SHARDS | tr "," " " | wc -w`
total_tablet_count=$(($num_shards*$TABLETS_PER_SHARD*$num_cells))
vtgate_count=$VTGATE_COUNT
if [ $vtgate_count -eq 0 ]; then
  vtgate_count=$(($total_tablet_count/$num_cells/4>3?$total_tablet_count/$num_cells/4:3))
fi

VTCTLD_SERVICE_TYPE=`[[ $TEST_MODE -gt 0 ]] && echo 'LoadBalancer' || echo 'ClusterIP'`

# export for other scripts
export SHARDS=$SHARDS
export TABLETS_PER_SHARD=$TABLETS_PER_SHARD
export RDONLY_COUNT=$RDONLY_COUNT
export VTDATAROOT_VOLUME=$VTDATAROOT_VOLUME
export VTGATE_TEMPLATE=$VTGATE_TEMPLATE
export VTTABLET_TEMPLATE=$VTTABLET_TEMPLATE
export VTGATE_REPLICAS=$vtgate_count
export VTCTLD_SERVICE_TYPE=$VTCTLD_SERVICE_TYPE
export VITESS_NAME=$VITESS_NAME

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
    num_running=`$KUBECTL get pods --namespace=$VITESS_NAME | grep ^$task_name | grep Running | wc -l`

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
go get github.com/youtube/vitess/go/cmd/vtctlclient

echo "****************************"
echo "*Creating vitess cluster: $VITESS_NAME"
echo "*  Shards: $SHARDS"
echo "*  Tablets per shard: $TABLETS_PER_SHARD"
echo "*  Rdonly per shard: $RDONLY_COUNT"
echo "*  VTGate count: $vtgate_count"
echo "*  Cells: $cells"
echo "****************************"

echo 'Running namespace-up.sh' && ./namespace-up.sh

echo 'Running etcd-up.sh' && CELLS=$CELLS ./etcd-up.sh
wait_for_running_tasks etcd-global 3
for cell in $cells; do
  wait_for_running_tasks etcd-$cell 3
done

echo 'Running vtctld-up.sh' && TEST_MODE=$TEST_MODE ./vtctld-up.sh

wait_for_running_tasks vtctld 1
kvtctl="./kvtctl.sh"

if [ $num_shards -gt 0 ]
then
  echo Calling CreateKeyspace and SetKeyspaceShardingInfo
  $kvtctl CreateKeyspace -force $KEYSPACE
  $kvtctl SetKeyspaceShardingInfo -force $KEYSPACE keyspace_id uint64
fi

echo 'Running vttablet-up.sh' && CELLS=$CELLS ./vttablet-up.sh
echo 'Running vtgate-up.sh' && ./vtgate-up.sh
wait_for_running_tasks vttablet $total_tablet_count
wait_for_running_tasks vtgate $(($vtgate_count*$num_cells))


echo Waiting for tablets to be visible in the topology
counter=0
while [ $counter -lt $MAX_VTTABLET_TOPO_WAIT_RETRIES ]; do
  num_tablets=0
  for cell in $cells; do
    num_tablets=$(($num_tablets+`$kvtctl ListAllTablets $cell | grep $KEYSPACE | wc -l`))
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

echo -n Setting Keyspace Sharding Info...
$kvtctl SetKeyspaceShardingInfo -force $KEYSPACE keyspace_id uint64
echo Done
echo -n Rebuilding Keyspace Graph...
$kvtctl RebuildKeyspaceGraph $KEYSPACE
echo Done
echo -n Reparenting...
shard_num=1
master_cell=`echo $cells | awk '{print $1}'`
for shard in $(echo $SHARDS | tr "," " "); do
  [[ $num_cells -gt 1 ]] && cell_id=01 || cell_id=00
  printf -v master_tablet_id '%s-%02d0000%02d00' $master_cell $cell_id $shard_num
  $kvtctl InitShardMaster -force $KEYSPACE/$shard $master_tablet_id
  let shard_num=shard_num+1
done
echo Done
echo -n Applying Schema...
$kvtctl ApplySchema -sql "$(cat create_test_table.sql)" $KEYSPACE
echo Done

if [ $TEST_MODE -gt 0 ]; then
  echo Creating firewall rule for vtctld
  vtctld_port=15000
  gcloud compute firewall-rules create ${VITESS_NAME}-vtctld --allow tcp:$vtctld_port
  vtctld_ip=''
  until [ $vtctld_ip ]; do
    vtctld_ip=`$KUBECTL get -o template --template '{{if ge (len .status.loadBalancer) 1}}{{index (index .status.loadBalancer.ingress 0) "ip"}}{{end}}' service vtctld --namespace=$VITESS_NAME`
    sleep 1
  done
  vtctld_server="$vtctld_ip:$vtctld_port"
fi

vtgate_servers=''
for cell in $cells; do
  echo Creating firewall rule for vtgate in cell $cell
  vtgate_port=15001
  gcloud compute firewall-rules create ${VITESS_NAME}-vtgate-$cell --allow tcp:$vtgate_port
  vtgate_ip=''
  until [ $vtgate_ip ]; do
    vtgate_ip=`$KUBECTL get -o template --template '{{if ge (len .status.loadBalancer) 1}}{{index (index .status.loadBalancer.ingress 0) "ip"}}{{end}}' service vtgate-$cell --namespace=$VITESS_NAME`
    sleep 1
  done
  vtgate_servers+="vtgate-$cell: $vtgate_ip:$vtgate_port,"
done

if [ -n "$NEWRELIC_LICENSE_KEY" ]; then
  echo Setting up Newrelic monitoring
  i=1
  for nodename in `$KUBECTL get nodes --no-headers --namespace=$VITESS_NAME | awk '{print $1}'`; do
    gcloud compute copy-files newrelic.sh $nodename:~/
    gcloud compute copy-files newrelic_start_agent.sh $nodename:~/
    gcloud compute copy-files newrelic_start_mysql_plugin.sh $nodename:~/
    gcloud compute ssh $nodename --command "bash -c '~/newrelic.sh ${NEWRELIC_LICENSE_KEY} ${VTDATAROOT}'"
    let i=i+1
  done
fi

echo "****************************"
echo "* Complete!"
if [ $TEST_MODE -gt 0 ]; then
  echo "* vtctld: $vtctld_server"
else
  echo "* Access the vtctld web UI by performing the following steps:"
  echo "*   $ kubectl proxy --port=8001"
  echo "*   Visit http://localhost:8001/api/v1/proxy/namespaces/default/services/vtctld:web/"
fi
echo $vtgate_servers | awk -F',' '{for(i=1;i<NF;i++) print "* " $i}'
echo "****************************"
