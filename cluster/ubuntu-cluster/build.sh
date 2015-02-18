#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

# simple use the sed to replace some ip settings on user's demand
# Run as root only

# author @resouer
set -e

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

# kuber
echo "Download kubernetes release ..."
if [ ! -f kubernetes.tar.gz ] ; then
    curl -L https://github.com/GoogleCloudPlatform/kubernetes/releases/download/v0.10.1/kubernetes.tar.gz -o kubernetes.tar.gz
    tar xzf kubernetes.tar.gz
fi
pushd kubernetes/server
tar xzf kubernetes-server-linux-amd64.tar.gz
popd
cp kubernetes/server/kubernetes/server/bin/* binaries/

rm -rf flannel kubernetes* etcd*
echo "Done! All your commands locate in ./binaries dir"
