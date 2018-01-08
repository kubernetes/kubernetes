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

# This is an include file used by the other scripts in this directory.

# Most clusters will just be accessed with 'kubectl' on $PATH.
# However, some might require a different command. For example, GKE required
# KUBECTL='gcloud container kubectl' for a while. Now that most of our
# use cases just need KUBECTL=kubectl, we'll make that the default.
KUBECTL=${KUBECTL:-kubectl}

# Kubernetes API address for $KUBECTL. When the Kubernetes API server is not
# local, We can easily access the API by editing KUBERNETES_API_SERVER's value
KUBERNETES_API_SERVER=${KUBERNETES_API_SERVER:-""}

# Kubernetes namespace for Vitess and components.
VITESS_NAME=${VITESS_NAME:-'default'}

# Kubernetes options config
KUBECTL_OPTIONS="--namespace=$VITESS_NAME"
if [[ -n "$KUBERNETES_API_SERVER" ]]; then
  KUBECTL_OPTIONS+=" --server=$KUBERNETES_API_SERVER"
fi

# CELLS should be a comma separated list of cells
# the first cell listed will become local to vtctld.
CELLS=${CELLS:-'test'}

# This should match the nodePort in vtctld-service.yaml
VTCTLD_PORT=${VTCTLD_PORT:-30001}

# Get the ExternalIP of any node.
get_node_ip() {
  $KUBECTL $KUBECTL_OPTIONS get -o template --template '{{range (index .items 0).status.addresses}}{{if eq .type "ExternalIP" "LegacyHostIP"}}{{.address}}{{end}}{{end}}' nodes
}

# Try to find vtctld address if not provided.
get_vtctld_addr() {
  if [ -z "$VTCTLD_ADDR" ]; then
    VTCTLD_HOST=$(get_node_ip)
    if [ -n "$VTCTLD_HOST" ]; then
      VTCTLD_ADDR="$VTCTLD_HOST:$VTCTLD_PORT"
    fi
  fi
}

# Find the name of a vtctld pod.
get_vtctld_pod() {
  $KUBECTL $KUBECTL_OPTIONS get -o template --template "{{if ge (len .items) 1 }}{{(index .items 0).metadata.name}}{{end}}" -l 'app=vitess,component=vtctld' pods
}

start_vtctld_forward() {
  pod=`get_vtctld_pod`
  if [ -z "$pod" ]; then
    >&2 echo "ERROR: Can't get vtctld pod name. Is vtctld running?"
    return 1
  fi

  tmpfile=`mktemp`
  $KUBECTL $KUBECTL_OPTIONS port-forward $pod 0:15999 &> $tmpfile &
  vtctld_forward_pid=$!

  until [[ `cat $tmpfile` =~ :([0-9]+)\ -\> ]]; do :; done
  vtctld_forward_port=${BASH_REMATCH[1]}
  rm $tmpfile
}

stop_vtctld_forward() {
  kill $vtctld_forward_pid
}

config_file=`dirname "${BASH_SOURCE}"`/config.sh
if [ ! -f $config_file ]; then
  echo "Please run ./configure.sh first to generate config.sh file."
  exit 1
fi

source $config_file

# Fill in defaults for new variables, so old config.sh files still work.
vitess_image=${vitess_image:-vitess/lite}

