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

# Prepopulate the name of the Master
mkdir -p /etc/salt/minion.d
echo "master: $SALT_MASTER" > /etc/salt/minion.d/master.conf

cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  roles:
    - kubernetes-master
  cloud: aws
  cbr-cidr: "${MASTER_IP_RANGE}"
EOF

if [[ -n "${DOCKER_OPTS}" ]]; then
  cat <<EOF >>/etc/salt/minion.d/grains.conf
  docker_opts: '$(echo "$DOCKER_OPTS" | sed -e "s/'/''/g")'
EOF
fi

if [[ -n "${DOCKER_ROOT}" ]]; then
  cat <<EOF >>/etc/salt/minion.d/grains.conf
  docker_root: '$(echo "$DOCKER_ROOT" | sed -e "s/'/''/g")'
EOF
fi

# Auto accept all keys from minions that try to join
mkdir -p /etc/salt/master.d
cat <<EOF >/etc/salt/master.d/auto-accept.conf
auto_accept: True
EOF

cat <<EOF >/etc/salt/master.d/reactor.conf
# React to new minions starting by running highstate on them.
reactor:
  - 'salt/minion/*/start':
    - /srv/reactor/highstate-new.sls
EOF

# Install Salt
#
# We specify -X to avoid a race condition that can cause minion failure to
# install.  See https://github.com/saltstack/salt-bootstrap/issues/270
#
# -M installs the master
set +x
curl -L --connect-timeout 20 --retry 6 --retry-delay 10 https://bootstrap.saltstack.com | sh -s -- -M -X
set -x
