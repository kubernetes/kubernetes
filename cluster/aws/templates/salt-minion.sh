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
  cbr-cidr: 10.123.45.0/30
  cloud: aws
EOF

# We set the hostname_override to the full EC2 private dns name
# we'd like to use EC2 instance-id, but currently the kubelet health-check assumes the name
# is resolvable, although that check should be going away entirely (#7092)
if [[ -z "${HOSTNAME_OVERRIDE}" ]]; then
  HOSTNAME_OVERRIDE=`curl --silent curl http://169.254.169.254/2007-01-19/meta-data/local-hostname`
fi

# Helper that sets a salt grain in grains.conf, if the upper-cased key is a non-empty env
function env_to_salt {
  local key=$1
  local env_key=`echo $key | tr '[:lower:]' '[:upper:]'`
  local value=${!env_key}
  if [[ -n "${value}" ]]; then
    # Note this is yaml, so indentation matters
    cat <<EOF >>/etc/salt/minion.d/grains.conf
  ${key}: '$(echo "${value}" | sed -e "s/'/''/g")'
EOF
  fi
}

env_to_salt hostname_override
env_to_salt docker_opts
env_to_salt docker_root
env_to_salt kubelet_root
env_to_salt non_masquerade_cidr

install-salt

# Cleanup to free some disk space.
apt-get autoclean

service salt-minion start
