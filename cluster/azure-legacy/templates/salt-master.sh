#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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
echo "master: $MASTER_NAME" > /etc/salt/minion.d/master.conf

cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  roles:
    - kubernetes-master
  cloud: azure-legacy
EOF


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

env_to_salt docker_opts
env_to_salt docker_root
env_to_salt kubelet_root
env_to_salt master_extra_sans
env_to_salt runtime_config


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

mkdir -p /srv/salt/nginx
echo $MASTER_HTPASSWD > /srv/salt/nginx/htpasswd

mkdir -p /etc/openvpn
umask=$(umask)
umask 0066
echo "$CA_CRT" > /etc/openvpn/ca.crt
echo "$SERVER_CRT" > /etc/openvpn/server.crt
echo "$SERVER_KEY" > /etc/openvpn/server.key
umask $umask

cat <<EOF >/etc/salt/minion.d/log-level-debug.conf
log_level: debug
log_level_logfile: debug
EOF

cat <<EOF >/etc/salt/master.d/log-level-debug.conf
log_level: debug
log_level_logfile: debug
EOF

echo "Sleep 150 to wait minion to be up"
sleep 150

install-salt --master

# Wait a few minutes and trigger another Salt run to better recover from
# any transient errors.
echo "Sleeping 180"
sleep 180
salt-call state.highstate || true
