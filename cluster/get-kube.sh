#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# Bring up a Kubernetes cluster.
# Usage:
#   wget -q -O - https://get.k8s.io | bash
# or
#   curl -sS https://get.k8s.io | bash
#
# Advanced options
#  Set KUBERNETES_PROVIDER to choose between different providers:
#  Google Compute Engine [default]
#   * export KUBERNETES_PROVIDER=gce; wget -q -O - https://get.k8s.io | bash
#  Google Container Engine
#   * export KUBERNETES_PROVIDER=gke; wget -q -O - https://get.k8s.io | bash
#  Amazon EC2
#   * export KUBERNETES_PROVIDER=aws; wget -q -O - https://get.k8s.io | bash
#  Vagrant (local virtual machines)
#   * export KUBERNETES_PROVIDER=vagrant; wget -q -O - https://get.k8s.io | bash
#  VMWare VSphere
#   * export KUBERNETES_PROVIDER=vsphere; wget -q -O - https://get.k8s.io | bash
#  Rackspace
#   * export KUBERNETES_PROVIDER=rackspace; wget -q -O - https://get.k8s.io | bash
#
#  Set KUBERNETES_SKIP_DOWNLOAD to non-empty to skip downloading a release.
#  Set KUBERNETES_SKIP_CONFIRM to skip the installation confirmation prompt.
set -o errexit
set -o nounset
set -o pipefail

function create_cluster {
  echo "Creating a kubernetes on ${KUBERNETES_PROVIDER:-gce}..."
  (
    cd kubernetes
    ./cluster/kube-up.sh
    echo "Kubernetes binaries at ${PWD}/cluster/"
    if [[ ":$PATH:" != *":${PWD}/cluster:"* ]]; then
      echo "You may want to add this directory to your PATH in \$HOME/.profile"
    fi

    echo "Installation successful!"
  )
}

if [[ "${KUBERNETES_SKIP_DOWNLOAD-}" ]]; then
  create_cluster
  exit 0
fi

function get_latest_version_number {
  local -r latest_url="https://storage.googleapis.com/kubernetes-release/release/stable.txt"
  if [[ $(which wget) ]]; then
    wget -qO- ${latest_url}
  elif [[ $(which curl) ]]; then
    curl -Ss ${latest_url}
  else
    echo "Couldn't find curl or wget.  Bailing out."
    exit 4
  fi
}

release=$(get_latest_version_number)
release_url=https://storage.googleapis.com/kubernetes-release/release/${release}/kubernetes.tar.gz

uname=$(uname)
if [[ "${uname}" == "Darwin" ]]; then
  platform="darwin"
elif [[ "${uname}" == "Linux" ]]; then
  platform="linux"
else
  echo "Unknown, unsupported platform: (${uname})."
  echo "Supported platforms: Linux, Darwin."
  echo "Bailing out."
  exit 2
fi

machine=$(uname -m)
if [[ "${machine}" == "x86_64" ]]; then
  arch="amd64"
elif [[ "${machine}" == "i686" ]]; then
  arch="386"
elif [[ "${machine}" == "arm*" ]]; then
  arch="arm"
else
  echo "Unknown, unsupported architecture (${machine})."
  echo "Supported architectures x86_64, i686, arm*"
  echo "Bailing out."
  exit 3
fi

file=kubernetes.tar.gz

echo "Downloading kubernetes release ${release} to ${PWD}/kubernetes.tar.gz"
if [[ -n "${KUBERNETES_SKIP_CONFIRM-}" ]]; then
  echo "Is this ok? [Y]/n"
  read confirm
  if [[ "$confirm" == "n" ]]; then
    echo "Aborting."
    exit 0
  fi
fi

if [[ $(which wget) ]]; then
  wget -O ${file} ${release_url}
elif [[ $(which curl) ]]; then
  curl -L -o ${file} ${release_url}
else
  echo "Couldn't find curl or wget.  Bailing out."
  exit 1
fi

echo "Unpacking kubernetes release ${release}"
tar -xzf ${file}
rm ${file}

create_cluster
