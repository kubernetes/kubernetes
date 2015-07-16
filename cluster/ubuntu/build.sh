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

# Download the etcd, flannel, and K8s binaries automatically and stored in binaries directory
# Run as root only

# author @resouer @WIZARD-CXY
set -e

function cleanup {
  # cleanup work
  rm -rf flannel* kubernetes* etcd* binaries
}
trap cleanup SIGHUP SIGINT SIGTERM

mkdir -p binaries/master
mkdir -p binaries/minion

# flannel
echo "Download flannel release ..."
FLANNEL_VERSION=${FLANNEL_VERSION:-"0.4.0"}
echo "Flannel version is $FLANNEL_VERSION"
if [ ! -f flannel.tar.gz ] ; then
  curl -L  https://github.com/coreos/flannel/releases/download/v${FLANNEL_VERSION}/flannel-${FLANNEL_VERSION}-linux-amd64.tar.gz -o flannel.tar.gz
  tar xzf flannel.tar.gz
fi
cp flannel-${FLANNEL_VERSION}/flanneld binaries/master
cp flannel-${FLANNEL_VERSION}/flanneld binaries/minion

# ectd
echo "Download etcd release ..."

ETCD_VERSION=${ETCD_VERSION:-"2.0.12"}
ETCD="etcd-v${ETCD_VERSION}-linux-amd64"

if [ ! -f etcd.tar.gz ] ; then
  curl -L https://github.com/coreos/etcd/releases/download/v${ETCD_VERSION}/${ETCD}.tar.gz -o etcd.tar.gz
  tar xzf etcd.tar.gz
fi
cp $ETCD/etcd $ETCD/etcdctl binaries/master
cp $ETCD/etcd $ETCD/etcdctl binaries/minion

# k8s
echo "Download kubernetes release ..."
K8S_VERSION=${K8S_VERSION:-"0.19.3"}

if [ ! -f kubernetes.tar.gz ] ; then
  curl -L https://github.com/GoogleCloudPlatform/kubernetes/releases/download/v${K8S_VERSION}/kubernetes.tar.gz -o kubernetes.tar.gz
  tar xzf kubernetes.tar.gz
fi
pushd kubernetes/server
tar xzf kubernetes-server-linux-amd64.tar.gz
popd
cp kubernetes/server/kubernetes/server/bin/kube-apiserver \
   kubernetes/server/kubernetes/server/bin/kube-controller-manager \
   kubernetes/server/kubernetes/server/bin/kube-scheduler binaries/master

cp kubernetes/server/kubernetes/server/bin/kubelet \
   kubernetes/server/kubernetes/server/bin/kube-proxy binaries/minion

cp kubernetes/server/kubernetes/server/bin/kubectl binaries/

rm -rf flannel* kubernetes* etcd*

echo "Done! All your commands locate in ./binaries dir"
