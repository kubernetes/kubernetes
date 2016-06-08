#!/bin/bash

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

## MAIN

kube::main::start(){
  KUBE_VERSION=${KUBE_VERSION:-$(kube::util::get_latest_version)}
  KUBE_PLATFORM=$(kube::util::host_platform)
  KUBE_ARCH=${KUBE_PLATFORM##*/}

  kube::util::install_kubectl

  KUBECTL=${KUBECTL:-$(which kubectl)}

  kube::main::docker

  cat <<'EOF'

Now your brand-new Kubernetes cluster should be up and running!
Data is stored in /var/lib/kubelet, and all Kubernetes components are inside containers
The apiserver is running at localhost:8080, which means it is not visible to other computers on the network.

To get started with your new one-node cluster, use `kubectl`:

  kubectl get nodes
  kubectl get pods
  kubectl get services
  kubectl version

To run a sample pod to test it's working, run this (use an image other than `nginx` for other architectures):

  kubectl run nginx --image=nginx --port=80 --expose
  kubectl get pods
  kubectl get services

You'll notice that a new service is created at an internal ip, let's curl it

  curl $(kubectl get svc nginx --template={{.spec.clusterIP}})

To shut everything (all containers) down, run this (you might have to use `sudo`):

  docker rm $(docker ps -aq)
  umount $(cat /proc/mounts | grep /var/lib/kubelet | awk '{print $2}') 
  rm -rf /var/lib/kubelet

For more information, visit kubernetes.io
EOF
}

kube::main::docker() {
  kube::log::normal "Creating a local cluster:"
  kube::log::indented "Starting kubelet..."
  kube::util::run "docker run -d \
  --volume=/:/rootfs:ro \
  --volume=/sys:/sys:ro \
  --volume=/var/lib/docker/:/var/lib/docker:rw \
  --volume=/var/lib/kubelet/:/var/lib/kubelet:rw \
  --volume=/var/run:/var/run:rw \
  --net=host \
  --pid=host \
  --privileged=true \
  gcr.io/google_containers/hyperkube-${KUBE_ARCH}:${KUBE_VERSION} \
    /hyperkube kubelet \
      --containerized \
      --hostname-override="127.0.0.1" \
      --api-servers=http://localhost:8080 \
      --config=/etc/kubernetes/manifests \
      --allow-privileged=true \
      --cluster-dns=10.0.0.10 \
      --cluster-domain=cluster.local \
      --v=2"

  kube::log::indented "Waiting for master components to start..."

  # We expect to have at least 3 running pods - etcd, master and kube-proxy.
  while (($(${KUBECTL} get pods --no-headers 2>/dev/null | grep "Running" | wc -l) < 3)); do
    echo -n "."
    sleep 2
  done

  kube::log::green "SUCCESS"
  kube::log::green "Cluster created!"
}

## UTILS

# Installs kubectl for the specific os/arch
kube::util::install_kubectl(){

  kubectl_url="https://storage.googleapis.com/kubernetes-release/release/${KUBE_VERSION}/bin/${KUBE_PLATFORM}/kubectl"

  if [[ ! -f $(which kubectl 2>&1) ]]; then
    echo -n "Downloading kubectl binary to /usr/local/bin/kubectl..."

    # Download kubectl to /usr/local/bin and 
    kube::util::curl ${kubectl_url} > /usr/local/bin/kubectl
    chmod a+x /usr/local/bin/kubectl
  else

    # TODO: We should detect version of kubectl binary if it's too old and download newer version.
    kube::log::normal "Detected existing kubectl binary. Skipping download."
  fi
}

kube::util::get_latest_version() {

  # Curl for the latest stable version
  kube::util::curl "https://storage.googleapis.com/kubernetes-release/release/stable.txt"
}

# This figures out the host platform without relying on golang. We need this as
# we don't want a golang install to be a prerequisite to building yet we need
# this info to figure out where the final binaries are placed.
kube::util::host_platform() {
  local host_os
  local host_arch
  case "$(uname -s)" in
    Linux)
      host_os=linux;;
    *)
      kube::log::red "Unsupported host OS. Must be linux."
      exit 1;;
  esac

  case "$(uname -m)" in
    x86_64*)
      host_arch=amd64;;
    i?86_64*)
      host_arch=amd64;;
    amd64*)
      host_arch=amd64;;
    aarch64*)
      host_arch=arm64;;
    arm64*)
      host_arch=arm64;;
    arm*)
      host_arch=arm;;
    ppc64le*)
      host_arch=ppc64le;;
    *)  
      kube::log::red "Unsupported host arch. Must be x86_64, arm, arm64 or ppc64le."
      exit 1;;
  esac
  echo "${host_os}/${host_arch}"
}

kube::util::run() {

  # For a moment we need to change bash options to capture message if a command fails.
  set +o errexit
  output=$($1 2>&1)
  exit_code=$?
  set -o errexit

  if [[ ${exit_code} == 0 ]]; then
    kube::log::green "SUCCESS"
  else
    kube::log::red "FAILED"
    echo ${output} >&2
    exit 1
  fi
}

# Wraps curl or wget in a helper function.
# Output is redirected to stdout
kube::util::curl(){
  if [[ $(which curl 2>&1) ]]; then
    curl -sSL $1
  elif [[ $(which wget 2>&1) ]]; then
    wget -qO- $1
  else
    kube::log::red "Couldn't find curl or wget."
    kube::log::red "Bailing out."
    exit 4
  fi
}

## LOGGING

declare -r RED="\033[0;31m"
declare -r GREEN="\033[0;32m"
declare -r YELLOW="\033[0;33m"

kube::log::green(){
  echo -e "${GREEN}$1"; tput sgr0
}

kube::log::yellow(){
  echo -e "${YELLOW}$1"; tput sgr0
}

kube::log::red(){
  echo -e "${RED}$1"; tput sgr0
}

kube::log::normal(){
  echo $1
}
kube::log::indented(){
  echo -e -n "\t${1}"
}


# Execute the script
kube::main::start
