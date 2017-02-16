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

# This script will download latest version of kubectl command line tool and will
# bring up a local Kubernetes cluster with a single node.
set -o errexit
set -o nounset
set -o pipefail

KUBE_HOST=${KUBE_HOST:-localhost}

declare -r RED="\033[0;31m"
declare -r GREEN="\033[0;32m"
declare -r YELLOW="\033[0;33m"

function echo_green {
  echo -e "${GREEN}$1"; tput sgr0
}

function echo_red {
  echo -e "${RED}$1"; tput sgr0
}

function echo_yellow {
  echo -e "${YELLOW}$1"; tput sgr0
}

function run {
  # For a moment we need to change bash options to capture message if a command fails.
  set +o errexit
  output=$($1 2>&1)
  exit_code=$?
  set -o errexit
  if [ $exit_code -eq 0 ]; then
    echo_green "SUCCESS"
  else
    echo_red "FAILED"
    echo $output >&2
    exit 1
  fi
}

function create_cluster {
  echo "Creating a local cluster:"
  echo -e -n "\tStarting kubelet..."
  run "docker run \
  --volume=/:/rootfs:ro \
  --volume=/sys:/sys:ro \
  --volume=/var/lib/docker/:/var/lib/docker:rw \
  --volume=/var/lib/kubelet/:/var/lib/kubelet:rw \
  --volume=/var/run:/var/run:rw \
  --net=host \
  --pid=host \
  --privileged=true \
  -d \
  gcr.io/google_containers/hyperkube-${arch}:${release} \
    /hyperkube kubelet \
      --containerized \
      --hostname-override="127.0.0.1" \
      --address="0.0.0.0" \
      --api-servers=http://localhost:8080 \
      --pod-manifest-path=/etc/kubernetes/manifests \
      --allow-privileged=true \
      --cluster-dns=10.0.0.10 \
      --cluster-domain=cluster.local \
      --v=2"

  echo -e -n "\tWaiting for master components to start..."
  while true; do
    local running_count=$(kubectl -s=http://${KUBE_HOST}:8080 get pods --no-headers 2>/dev/null | grep "Running" | wc -l)
    # We expect to have 3 running pods - etcd, master and kube-proxy.
    if [ "$running_count" -ge 3 ]; then
      break
    fi
    echo -n "."
    sleep 1
  done
  echo_green "SUCCESS"
  echo_green "Cluster created!"
  echo ""
  kubectl -s http://${KUBE_HOST}:8080 clusterinfo
}

function get_latest_version_number {
  local -r latest_url="https://storage.googleapis.com/kubernetes-release/release/stable.txt"
  if [[ $(which wget) ]]; then
    wget -qO- ${latest_url}
  elif [[ $(which curl) ]]; then
    curl -Ss ${latest_url}
  else
    echo_red "Couldn't find curl or wget.  Bailing out."
    exit 4
  fi
}

latest_release=$(get_latest_version_number)
release=${KUBE_VERSION:-${latest_release}}

uname=$(uname)
if [[ "${uname}" == "Darwin" ]]; then
  platform="darwin"
elif [[ "${uname}" == "Linux" ]]; then
  platform="linux"
else
  echo_red "Unknown, unsupported platform: (${uname})."
  echo_red "Supported platforms: Linux, Darwin."
  echo_red "Bailing out."
  exit 2
fi

machine=$(uname -m)
if [[ "${machine}" == "x86_64" ]]; then
  arch="amd64"
elif [[ "${machine}" == "i686" ]]; then
  arch="386"
elif [[ "${machine}" == "arm*" ]]; then
  arch="arm"
elif [[ "${machine}" == "s390x*" ]]; then
  arch="s390x"
else
  echo_red "Unknown, unsupported architecture (${machine})."
  echo_red "Supported architectures x86_64, i686, arm, s390x."
  echo_red "Bailing out."
  exit 3
fi

kubectl_url="https://storage.googleapis.com/kubernetes-release/release/${release}/bin/${platform}/${arch}/kubectl"

if [[ $(ls . | grep ^kubectl$ | wc -l) -lt 1 ]]; then
  echo -n "Downloading kubectl binary..."
  if [[ $(which wget) ]]; then
    run "wget ${kubectl_url}"
  elif [[ $(which curl) ]]; then
    run "curl -OL ${kubectl_url}"
  else
    echo_red "Couldn't find curl or wget.  Bailing out."
    exit 1
  fi
  chmod a+x kubectl
  echo ""
else
  # TODO: We should detect version of kubectl binary if it too old
  # download newer version.
  echo "Detected existing kubectl binary. Skipping download."
fi

create_cluster

echo ""
echo ""
echo "To list the nodes in your cluster run"
echo_yellow "\tkubectl -s=http://${KUBE_HOST}:8080 get nodes"
echo ""
echo "To run your first pod run"
echo_yellow "\tkubectl -s http://${KUBE_HOST}:8080 run nginx --image=nginx --port=80"
