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
        go tool pprof "-pdf" "${binary}" "http://localhost:${tunnel_port}/${path}/profile" > "${output_prefix}-${profile}-profile-${timestamp}.pdf"
        ;;
      mem)
        # There are different kinds of memory profiles that are available that
        # had to be grabbed separately: --inuse-space, --inuse-objects,
        # --alloc-space, --alloc-objects. We need to iterate over all requested
        # kinds.
        for flag in ${mem_pprof_flags}; do
          go tool pprof "-${flag}" "-pdf" "${binary}" "http://localhost:${tunnel_port}/${path}/heap" > "${output_prefix}-${profile}-${flag}-profile-${timestamp}.pdf"
        done
        ;;
    esac
  done
}

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

server_addr=""
kubelet_addreses=""
kubelet_binary=""
requested_profiles=""
mem_pprof_flags=""
profile_components=""
output_dir="."
tunnel_port="${tunnel_port:-1234}"

args=$(getopt -o s:mho:k: -l server:,master,heapster,output:,kubelet:,help,inuse-space,inuse-objects,alloc-space,alloc-objects,cpu,kubelet-binary: -- "$@")
if [[ $? -ne 0 ]]; then
  >&2 echo "Error in getopt"
  exit 1
fi

HEAPSTER_VERSION="v0.18.2"
MASTER_PPROF_PATH="debug/pprof"
HEAPSTER_PPROF_PATH="api/v1/proxy/namespaces/kube-system/services/monitoring-heapster/debug/pprof"
KUBELET_PPROF_PATH_PREFIX="api/v1/proxy/nodes"

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
    -h|--heapster)
      shift
      profile_components="heapster ${profile_components}"
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
    -k|--kubelet)
      shift
      profile_components="kubelet ${profile_components}"
      if [ -z "$1" ]; then
        >&2 echo "empty argumet to --kubelet flag"
        exit 1
      fi
      kubelet_addreses="$1 $kubelet_addreses"
      shift
      ;;
    --kubelet-binary)
      shift
      if [ -z "$1" ]; then
        >&2 echo "empty argumet to --kubelet-binary flag"
        exit 1
      fi
      kubelet_binary=$1
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

gcloud compute ssh "${server_addr}" --ssh-flag=-nN --ssh-flag=-L${tunnel_port}:localhost:8080 &

echo "Waiting for tunnel to be created..."
kube::util::wait_for_url http://localhost:${tunnel_port}/healthz

SSH_PID=$(pgrep -f "/usr/bin/ssh.*${tunnel_port}:localhost:8080")
kube::util::trap_add 'kill $SSH_PID' EXIT
kube::util::trap_add 'kill $SSH_PID' SIGTERM

requested_profiles=$(echo ${requested_profiles} | xargs -n1 | sort -u | xargs)
profile_components=$(echo ${profile_components} | xargs -n1 | sort -u | xargs)
kubelet_addreses=$(echo ${kubelet_addreses} | xargs -n1 | sort -u | xargs)
echo "requested profiles: ${requested_profiles}"
echo "flags for heap profile: ${mem_pprof_flags}"

timestamp=$(date +%Y%m%d%H%M%S)
binary=""

for component in ${profile_components}; do
  case ${component} in
    master)
      path=${MASTER_PPROF_PATH}
      binary=""
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
    for node in $(echo ${kubelet_addreses} | sed 's/[,;]/\n/g'); do
      grab_profiles_from_component "${requested_profiles}" "${mem_pprof_flags}" "${binary}" "${tunnel_port}" "${path}/${node}/debug/pprof" "${output_dir}/${component}" "${timestamp}"
    done    
  else 
    grab_profiles_from_component "${requested_profiles}" "${mem_pprof_flags}" "${binary}" "${tunnel_port}" "${path}" "${output_dir}/${component}" "${timestamp}"
  fi
done
