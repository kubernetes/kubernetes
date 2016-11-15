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

# This is an include file used by the other scripts in this directory.

# Most clusters will just be accessed with 'kubectl' on $PATH.
# However, some might require a different command. For example, GKE required
# KUBECTL='gcloud beta container kubectl' for a while. Now that most of our
# use cases just need KUBECTL=kubectl, we'll make that the default.
KUBECTL=${KUBECTL:-kubectl}

# This should match the nodePort in vtctld-service.yaml
VTCTLD_PORT=${VTCTLD_PORT:-30001}

# Customizable parameters
SHARDS=${SHARDS:-'-80,80-'}
TABLETS_PER_SHARD=${TABLETS_PER_SHARD:-2}
RDONLY_COUNT=${RDONLY_COUNT:-0}
MAX_TASK_WAIT_RETRIES=${MAX_TASK_WAIT_RETRIES:-300}
MAX_VTTABLET_TOPO_WAIT_RETRIES=${MAX_VTTABLET_TOPO_WAIT_RETRIES:-180}
VTTABLET_TEMPLATE=${VTTABLET_TEMPLATE:-'vttablet-pod-template.yaml'}
VTGATE_TEMPLATE=${VTGATE_TEMPLATE:-'vtgate-controller-template.yaml'}
VTGATE_COUNT=${VTGATE_COUNT:-1}
CELLS=${CELLS:-'test'}
ETCD_REPLICAS=3

VTGATE_REPLICAS=$VTGATE_COUNT

# Get the ExternalIP of any node.
get_node_ip() {
  $KUBECTL get -o template -t '{{range (index .items 0).status.addresses}}{{if eq .type "ExternalIP"}}{{.address}}{{end}}{{end}}' nodes
}

# Try to find vtctld address if not provided.
get_vtctld_addr() {
  if [ -z "$VTCTLD_ADDR" ]; then
    node_ip=$(get_node_ip)
    if [ -n "$node_ip" ]; then
      VTCTLD_ADDR="$node_ip:$VTCTLD_PORT"
    fi
  fi
  echo "$VTCTLD_ADDR"
}

config_file=`dirname "${BASH_SOURCE}"`/config.sh
if [ ! -f $config_file ]; then
  echo "Please run ./configure.sh first to generate config.sh file."
  exit 1
fi

source $config_file

