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

set -o errexit
set -o nounset
set -o pipefail

. /etc/sysconfig/heat-params

FLANNEL_ETCD_URL="http://${MASTER_IP}:4379"

# Install flannel for overlay
if ! which flanneld >/dev/null 2>&1; then
  yum install -y flannel
fi

cat <<EOF >/etc/sysconfig/flanneld
FLANNEL_ETCD="${FLANNEL_ETCD_URL}"
FLANNEL_ETCD_KEY="/coreos.com/network"
FLANNEL_OPTIONS="-iface=eth0 --ip-masq"
EOF

systemctl enable flanneld
systemctl restart flanneld

# Kubernetes node shoud be able to resolve its hostname.
# In some cloud providers, myhostname is not enabled by default.
grep '^hosts:.*myhostname' /etc/nsswitch.conf || (
  sed -e 's/^hosts:\(.*\)/hosts:\1 myhostname/' -i /etc/nsswitch.conf
)
