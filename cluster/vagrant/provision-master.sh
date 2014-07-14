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

# exit on any error
set -e
source $(dirname $0)/provision-config.sh

# we will run provision to update code each time we test, so we do not want to do salt install each time
if [ ! -f "/var/kube-vagrant-setup" ]; then
  mkdir -p /etc/salt/minion.d
  echo "master: $MASTER_NAME" > /etc/salt/minion.d/master.conf

  cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  master_ip: $MASTER_IP
  etcd_servers: $MASTER_IP
  minion_ips: $MINION_IPS
  roles:
    - kubernetes-master
EOF

  # Configure the salt-master
  # Auto accept all keys from minions that try to join
  mkdir -p /etc/salt/master.d
  cat <<EOF >/etc/salt/master.d/auto-accept.conf
open_mode: True
auto_accept: True
EOF

  cat <<EOF >/etc/salt/master.d/reactor.conf
# React to new minions starting by running highstate on them.
reactor:
  - 'salt/minion/*/start':
    - /srv/reactor/start.sls
EOF

  # Install Salt
  #
  # We specify -X to avoid a race condition that can cause minion failure to
  # install.  See https://github.com/saltstack/salt-bootstrap/issues/270
  #
  # -M installs the master
  curl -L http://bootstrap.saltstack.com | sh -s -- -M

  mkdir -p /srv/salt/nginx
  echo $MASTER_HTPASSWD > /srv/salt/nginx/htpasswd

  # a file we touch to state that base-setup is done
  echo "Salt configured" > /var/kube-vagrant-setup
fi

# Build release
echo "Building release"
pushd /vagrant
  ./release/build-release.sh kubernetes
popd

echo "Running release install script"
pushd /vagrant/output/release/master-release/src/scripts
  ./master-release-install.sh
popd

echo "Executing configuration"
salt '*' mine.update
salt --force-color '*' state.highstate