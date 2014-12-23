#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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
#   wget -q -O - https://get.k8s.io | sh
# or
#   curl -sS https://get.k8s.io | sh
#
# Advanced options
#  Set KUBERNETES_PROVIDER to choose between different providers:
#  Google Compute Engine [default]
#   * export KUBERNETES_PROVIDER=gce; wget -q -O - https://get.k8s.io | sh
#  Amazon EC2
#   * export KUBERNETES_PROVIDER=aws; wget -q -O - https://get.k8s.io | sh
#  Microsoft Azure
#   * export KUBERNETES_PROVIDER=azure; wget -q -O - https://get.k8s.io | sh
#  Vagrant (local virtual machines)
#   * export KUBERNETES_PROVIDER=vagrant; wget -q -O - https://get.k8s.io | sh
#  VMWare VSphere
#   * export KUBERNETES_PROVIDER=vsphere; wget -q -O - https://get.k8s.io | sh
#  Rackspace
#   * export KUBERNETES_PROVIDER=rackspace; wget -q -O - https://get.k8s.io | sh

set -o errexit
set -o nounset
set -o pipefail

release=v0.7.0
release_url=https://storage.googleapis.com/kubernetes-release/release/${release}/kubernetes.tar.gz

uname=$(uname)
if [[ "${uname}" == "Darwin" ]]; then
  platform="darwin"
elif [[ "${uname}" == "Linux" ]]; then
  platform="linux"
else
  echo "Unknown, unsupported platform: (${uname}).  Bailing out."
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
  echo "Unknown, unsupported architecture (${machine}).  Bailing out."
  exit 3
fi

file=kubernetes.tar.gz

echo "Downloading kubernetes release ${release}"
if [[ $(which wget) ]]; then
  wget -O ${file} ${release_url}
elif [[ $(which curl) ]]; then
  curl -o ${file} ${release_url}
else
  echo "Couldn't find curl or wget.  Bailing out."
  exit 1
fi

echo "Unpacking kubernetes release ${release}"
tar -xzf ${file}
rm ${file}

echo "Installing kubernetes..."
(
  cd kubernetes
  ./cluster/kube-up.sh
)

echo "Kubernetes binaries at ${PWD}/kubernetes/platforms/${platform}/${arch}/PATH"
echo "You may want to add this directory to your PATH in \$HOME/.profile"

echo "Installation successful!"

