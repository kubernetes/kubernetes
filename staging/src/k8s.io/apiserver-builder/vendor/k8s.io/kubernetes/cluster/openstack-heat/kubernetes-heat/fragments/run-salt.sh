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

#Reads in profile, need to relax restrictions for some OSes.
set +o nounset
. /etc/profile
set -o nounset

rm -rf /kube-install
mkdir -p /kube-install
cd /kube-install

curl "${KUBERNETES_SERVER_URL}" -o kubernetes-server.tar.gz
curl "${KUBERNETES_SALT_URL}" -o kubernetes-salt.tar.gz

tar xzf kubernetes-salt.tar.gz
./kubernetes/saltbase/install.sh kubernetes-server.tar.gz

if ! which salt-call >/dev/null 2>&1; then
  echo "+++ Install salt binaries from https://bootstrap.saltstack.com"
  # Install salt binaries but do not start daemon after installation
  curl -sS -L --connect-timeout 20 --retry 6 --retry-delay 10 https://bootstrap.saltstack.com | sh -s -- "-X"
fi

# Salt server runs at locahost
echo "127.0.0.1 salt" >> /etc/hosts

echo "+++ run salt-call and finalize installation"
# Run salt-call
# salt-call wants to start docker daemon but is unable to.
# See <https://github.com/projectatomic/docker-storage-setup/issues/77>.
# Run salt-call in background and make cloud-final finished.
# Salt-call might be unstable in some environments, execute it twice.
salt-call --local state.highstate && salt-call --local state.highstate && $$wc_notify --data-binary '{"status": "SUCCESS"}' || $$wc_notify --data-binary '{"status": "FAILURE"}' &
