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

# Turn on debugging for salt-minion
# echo "DAEMON_ARGS=\"\$DAEMON_ARGS --log-file-level=debug\"" > /etc/default/salt-minion

# Our minions will have a pool role to distinguish them from the master.
cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  roles:
    - kubernetes-pool
  cbr-cidr: $MINION_IP_RANGE
  cloud: aws
EOF

# We set the hostname_override to the full EC2 private dns name
# we'd like to use EC2 instance-id, but currently the kubelet health-check assumes the name
# is resolvable, although that check should be going away entirely (#7092)
if [[ -z "${HOSTNAME_OVERRIDE}" ]]; then
  HOSTNAME_OVERRIDE=`curl --silent curl http://169.254.169.254/2007-01-19/meta-data/local-hostname`
fi

if [[ -n "${HOSTNAME_OVERRIDE}" ]]; then
  cat <<EOF >>/etc/salt/minion.d/grains.conf
  hostname_override: "${HOSTNAME_OVERRIDE}"
EOF
fi

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

install-salt

service salt-minion start
