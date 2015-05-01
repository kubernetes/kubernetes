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

# Download the etcd, flannel, and K8s binaries automatically
# Run as root only

# author @resouer @WIZARD-CXY
set -e

function cleanup {
    # cleanup work
    rm -rf flannel kubernetes* etcd* binaries
}
trap cleanup SIGHUP SIGINT SIGTERM

# check root
if [ "$(id -u)" != "0" ]; then
    echo >&2 "Please run as root"
    exit 1
fi

mkdir -p binaries

# flannel
echo "Download & build flanneld ..."
apt-get install linux-libc-dev
if [ ! -d flannel ] ; then
    echo "flannel does not exsit, cloning ..."
    git clone https://github.com/coreos/flannel.git
fi

pushd flannel
docker run -v `pwd`:/opt/flannel -i -t google/golang /bin/bash -c "cd /opt/flannel && ./build"
popd
cp flannel/bin/flanneld binaries/

# ectd
echo "Download etcd release ..."
ETCD_V="v2.0.0"
ETCD="etcd-${ETCD_V}-linux-amd64"
if [ ! -f etcd.tar.gz ] ; then
    curl -L  https://github.com/coreos/etcd/releases/download/$ETCD_V/$ETCD.tar.gz -o etcd.tar.gz
    tar xzf etcd.tar.gz
fi
cp $ETCD/etcd $ETCD/etcdctl binaries

# k8s
echo "Download kubernetes release ..."

K8S_V="v0.12.0"
if [ ! -f kubernetes.tar.gz ] ; then
    curl -L https://github.com/GoogleCloudPlatform/kubernetes/releases/download/$K8S_V/kubernetes.tar.gz -o kubernetes.tar.gz
    tar xzf kubernetes.tar.gz
fi
pushd kubernetes/server
tar xzf kubernetes-server-linux-amd64.tar.gz
popd
cp kubernetes/server/kubernetes/server/bin/* binaries/

rm -rf flannel kubernetes* etcd*
echo "Done! All your commands locate in ./binaries dir"
