#!/usr/bin/env bash

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

# This script grabs profiles from running components.
# Usage: `hack/grab-profiles.sh`.

set -o errexit
set -o nounset
set -o pipefail

function grab_profiles_from_component {
  local requested_profiles=$1
  local mem_pprof_flags=$2
  local binary=$3
  local tunnel_port=$4
  local path=$5
  local output_prefix=$6
  local timestamp=$7

  echo "binary: $binary"

  for profile in ${requested_profiles}; do
    case ${profile} in
      cpu)
        go tool pprof "-pdf" "${binary}" "http://localhost:${tunnel_port}${path}/debug/pprof/profile" > "${output_prefix}-${profile}-profile-${timestamp}.pdf"
        ;;
      mem)
        # There are different kinds of memory profiles that are available that
        # had to be grabbed separately: --inuse-space, --inuse-objects,
        # --alloc-space, --alloc-objects. We need to iterate over all requested
        # kinds.
        for flag in ${mem_pprof_flags}; do
          go tool pprof "-${flag}" "-pdf" "${binary}" "http://localhost:${tunnel_port}${path}/debug/pprof/heap" > "${output_prefix}-${profile}-${flag}-profile-${timestamp}.pdf"
        done
        ;;
    esac
  done
}

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

server_addr=""
kubelet_addresses=""
kubelet_binary=""
master_binary=""
scheduler_binary=""
scheduler_port="10251"
controller_manager_port="10252"
controller_manager_binary=""
requested_profiles=""
mem_pprof_flags=""
profile_components=""
output_dir="."
tunnel_port="${tunnel_port:-1234}"

if ! args=$(getopt -o s:mho:k:c -l server:,master,heapster,output:,kubelet:,scheduler,controller-manager,help,inuse-space,inuse-objects,alloc-space,alloc-objects,cpu,kubelet-binary:,master-binary:,scheduler-binary:,controller-manager-binary:,scheduler-port:,controller-manager-port: -- "$@"); then
  >&2 echo "Error in getopt"
  exit 1
fi

HEAPSTER_VERSION="v0.18.2"
MASTER_PPROF_PATH=""
HEAPSTER_PPROF_PATH="/api/v1/namespaces/kube-system/services/monitoring-heapster/proxy"
KUBELET_PPROF_PATH_PREFIX="/api/v1/proxy/nodes"
SCHEDULER_PPROF_PATH_PREFIX="/api/v1/namespaces/kube-system/pods/kube-scheduler/proxy"
CONTROLLER_MANAGER_PPROF_PATH_PREFIX="/api/v1/namespaces/kube-system/pods/kube-controller-manager/proxy"

eval set -- "${args}"

while true; do
  case $1 in
    -s|--server)
      shift
      if [ -z "$1" ]; then
        >&2 echo "empty argument to --server flag"
        exit 1
      fi
      server_addr=$1
      shift
      ;;
    -m|--master)
      shift
      profile_components="master ${profile_components}"
      ;;
    --master-binary)
      shift
      if [ -z "$1" ]; then
        >&2 echo "empty argument to --master-binary flag"
        exit 1
      fi
      master_binary=$1
      shift
      ;;
    -h|--heapster)
      shift
      profile_components="heapster ${profile_components}"
      ;;
    -k|--kubelet)
      shift
      profile_components="kubelet ${profile_components}"
      if [ -z "$1" ]; then
        >&2 echo "empty argument to --kubelet flag"
        exit 1
      fi
      kubelet_addresses="$1 $kubelet_addresses"
      shift
      ;;
    --kubelet-binary)
      shift
      if [ -z "$1" ]; then
        >&2 echo "empty argument to --kubelet-binary flag"
        exit 1
      fi
      kubelet_binary=$1
      shift
      ;;
    --scheduler)
      shift
      profile_components="scheduler ${profile_components}"
      ;;
    --scheduler-binary)
      shift
      if [ -z "$1" ]; then
        >&2 echo "empty argument to --scheduler-binary flag"
        exit 1
      fi
      scheduler_binary=$1
      shift
      ;;
    --scheduler-port)
      shift
      if [ -z "$1" ]; then
        >&2 echo "empty argument to --scheduler-port flag"
        exit 1
      fi
      scheduler_port=$1
      shift
      ;;
    -c|--controller-manager)
      shift
      profile_components="controller-manager ${profile_components}"
      ;;
    --controller-manager-binary)
      shift
      if [ -z "$1" ]; then
        >&2 echo "empty argument to --controller-manager-binary flag"
        exit 1
      fi
      controller_manager_binary=$1
      shift
      ;;
    --controller-manager-port)
      shift
      if [ -z "$1" ]; then
        >&2 echo "empty argument to --controller-manager-port flag"
        exit 1
      fi
      controller_manager_port=$1
      shift
      ;;
    -o|--output)
      shift
      if [ -z "$1" ]; then
        >&2 echo "empty argument to --output flag"
        exit 1
      fi
      output_dir=$1
      shift
      ;;
    --inuse-space)
      shift
      requested_profiles="mem ${requested_profiles}"
      mem_pprof_flags="inuse_space ${mem_pprof_flags}"
      ;;
    --inuse-objects)
      shift
      requested_profiles="mem ${requested_profiles}"
      mem_pprof_flags="inuse_objects ${mem_pprof_flags}"
      ;;
    --alloc-space)
      shift
      requested_profiles="mem ${requested_profiles}"
      mem_pprof_flags="alloc_space ${mem_pprof_flags}"
      ;;
    --alloc-objects)
      shift
      requested_profiles="mem ${requested_profiles}"
      mem_pprof_flags="alloc_objects ${mem_pprof_flags}"
      ;;
    --cpu)
      shift
      requested_profiles="cpu ${requested_profiles}"
      ;;
    --help)
      shift
      echo "Recognized options:
        -o/--output,
        -s/--server,
        -m/--master,
        -h/--heapster,
        --inuse-space,
        --inuse-objects,
        --alloc-space,
        --alloc-objects,
        --cpu,
        --help"
      exit 0
      ;;
    --)
      shift
      break;
      ;;
  esac
done

if [[ -z "${server_addr}" ]]; then
  >&2 echo "Server flag is required"
  exit 1
fi

if [[ -z "${profile_components}" ]]; then
  >&2 echo "Choose at least one component to profile"
  exit 1
fi

if [[ -z "${requested_profiles}" ]]; then
  >&2 echo "Choose at least one profiling option"
  exit 1
fi

gcloud compute ssh "${server_addr}" --ssh-flag=-nN --ssh-flag=-L"${tunnel_port}":localhost:8080 &

echo "Waiting for tunnel to be created..."
kube::util::wait_for_url http://localhost:"${tunnel_port}"/healthz

SSH_PID=$(pgrep -f "/usr/bin/ssh.*${tunnel_port}:localhost:8080")
kube::util::trap_add "kill $SSH_PID" EXIT
kube::util::trap_add "kill $SSH_PID" SIGTERM

requested_profiles=$(echo "${requested_profiles}" | xargs -n1 | LC_ALL=C sort -u | xargs)
profile_components=$(echo "${profile_components}" | xargs -n1 | LC_ALL=C sort -u | xargs)
kubelet_addresses=$(echo "${kubelet_addresses}" | xargs -n1 | LC_ALL=C sort -u | xargs)
echo "requested profiles: ${requested_profiles}"
echo "flags for heap profile: ${mem_pprof_flags}"

timestamp=$(date +%Y%m%d%H%M%S)
binary=""

for component in ${profile_components}; do
  case ${component} in
    master)
      path=${MASTER_PPROF_PATH}
      binary=${master_binary}
      ;;
    controller-manager)
      path="${CONTROLLER_MANAGER_PPROF_PATH_PREFIX}-${server_addr}:${controller_manager_port}"
      binary=${controller_manager_binary}
      ;;
    scheduler)
      path="${SCHEDULER_PPROF_PATH_PREFIX}-${server_addr}:${scheduler_port}"
      binary=${scheduler_binary}
      ;;
    heapster)
      rm heapster
      wget https://github.com/kubernetes/heapster/releases/download/${HEAPSTER_VERSION}/heapster
      kube::util::trap_add 'rm -f heapster' EXIT
      kube::util::trap_add 'rm -f heapster' SIGTERM
      binary=heapster
      path=${HEAPSTER_PPROF_PATH}
      ;;
    kubelet)
      path="${KUBELET_PPROF_PATH_PREFIX}"
      if [[ -z "${kubelet_binary}" ]]; then
        binary="${KUBE_ROOT}/_output/local/bin/linux/amd64/kubelet"
      else
        binary=${kubelet_binary}
      fi
      ;;
  esac

  if [[ "${component}" == "kubelet" ]]; then
    for node in ${kubelet_addresses//[,;]/' '}; do
      grab_profiles_from_component "${requested_profiles}" "${mem_pprof_flags}" "${binary}" "${tunnel_port}" "${path}/${node}/proxy" "${output_dir}/${component}" "${timestamp}"
    done
  else
    grab_profiles_from_component "${requested_profiles}" "${mem_pprof_flags}" "${binary}" "${tunnel_port}" "${path}" "${output_dir}/${component}" "${timestamp}"
  fi
done
