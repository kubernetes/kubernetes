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

# Download the etcd, flannel, and K8s binaries automatically and stored in binaries directory
# Run as root only

# author @resouer @WIZARD-CXY
set -e

function cleanup {
  # cleanup work
  rm -rf flannel* kubernetes* etcd* binaries out
}
trap cleanup SIGHUP SIGINT SIGTERM

pushd $(dirname $0)
mkdir -p binaries/master
mkdir -p binaries/minion
mkdir -p out

# flannel
FLANNEL_VERSION=${FLANNEL_VERSION:-"0.5.5"}
echo "Prepare flannel ${FLANNEL_VERSION} release ..."
grep -q "^${FLANNEL_VERSION}\$" binaries/.flannel 2>/dev/null || {
  ( curl --fail -L https://github.com/coreos/flannel/releases/download/v${FLANNEL_VERSION}/flannel-${FLANNEL_VERSION}-linux-amd64.tar.gz -o flannel.tar.gz &&
    tar xzf flannel.tar.gz flannel-${FLANNEL_VERSION}/flanneld -O > out/flanneld
  ) ||
  ( curl --fail -L https://github.com/coreos/flannel/releases/download/v${FLANNEL_VERSION}/flannel-v${FLANNEL_VERSION}-linux-amd64.tar.gz -o flannel.tar.gz &&
    tar xzf flannel.tar.gz flanneld -O > out/flanneld
  )
  chmod 0755 out/flanneld
  cp out/flanneld binaries/master
  cp out/flanneld binaries/minion
  echo ${FLANNEL_VERSION} > binaries/.flannel
}

# ectd
ETCD_VERSION=${ETCD_VERSION:-"2.3.1"}
ETCD="etcd-v${ETCD_VERSION}-linux-amd64"
echo "Prepare etcd ${ETCD_VERSION} release ..."
grep -q "^${ETCD_VERSION}\$" binaries/.etcd 2>/dev/null || {
  curl -L https://github.com/coreos/etcd/releases/download/v${ETCD_VERSION}/${ETCD}.tar.gz -o etcd.tar.gz
  tar xzf etcd.tar.gz
  cp ${ETCD}/etcd ${ETCD}/etcdctl binaries/master
  echo ${ETCD_VERSION} > binaries/.etcd
}

function get_latest_version_number {
  # TODO(#33726): switch to dl.k8s.io
  local -r latest_url="https://storage.googleapis.com/kubernetes-release/release/stable.txt"
  if [[ $(which wget) ]]; then
    wget -qO- ${latest_url}
  elif [[ $(which curl) ]]; then
    curl -Ss ${latest_url}
  else
    echo "Couldn't find curl or wget.  Bailing out." >&2
    exit 4
  fi
}

if [ -z "$KUBE_VERSION" ]; then
  KUBE_VERSION=$(get_latest_version_number | sed 's/^v//')
fi

# k8s
echo "Prepare kubernetes ${KUBE_VERSION} release ..."
grep -q "^${KUBE_VERSION}\$" binaries/.kubernetes 2>/dev/null || {
  # TODO(#33726): switch to dl.k8s.io
  curl -L https://storage.googleapis.com/kubernetes-release/release/v${KUBE_VERSION}/kubernetes-client-linux-amd64.tar.gz -o kubernetes-client-linux-amd64.tar.gz
  curl -L https://storage.googleapis.com/kubernetes-release/release/v${KUBE_VERSION}/kubernetes-server-linux-amd64.tar.gz -o kubernetes-server-linux-amd64.tar.gz
  tar xzf kubernetes-client-linux-amd64.tar.gz
  tar xzf kubernetes-server-linux-amd64.tar.gz
  cp kubernetes/client/bin/kubectl binaries/
  cp kubernetes/server/bin/kube-apiserver \
     kubernetes/server/bin/kube-controller-manager \
     kubernetes/server/bin/kube-scheduler binaries/master
  cp kubernetes/server/bin/kubelet \
     kubernetes/server/bin/kube-proxy binaries/minion
  echo ${KUBE_VERSION} > binaries/.kubernetes
}

rm -rf flannel* kubernetes* etcd* out

echo "Done! All your binaries locate in kubernetes/cluster/ubuntu/binaries directory"
popd
