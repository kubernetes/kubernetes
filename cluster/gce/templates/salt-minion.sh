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

# The repositories are really slow and there are GCE mirrors
sed -i -e "\|^deb.*http://http.debian.net/debian| s/^/#/" /etc/apt/sources.list
sed -i -e "\|^deb.*http://ftp.debian.org/debian| s/^/#/" /etc/apt/sources.list.d/backports.list

# Prepopulate the name of the Master
mkdir -p /etc/salt/minion.d
cat <<EOF >/etc/salt/minion.d/master.conf
master: '$(echo "$MASTER_NAME" | sed -e "s/'/''/g")'
EOF

cat <<EOF >/etc/salt/minion.d/log-level-debug.conf
log_level: debug
log_level_logfile: debug
EOF

# Our minions will have a pool role to distinguish them from the master.
cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  roles:
    - kubernetes-pool
  cbr-cidr: '$(echo "$MINION_IP_RANGE" | sed -e "s/'/''/g")'
  cloud: gce
EOF

DOCKER_OPTS=""

if [[ -n "${EXTRA_DOCKER_OPTS-}" ]]; then
  DOCKER_OPTS="${EXTRA_DOCKER_OPTS}"
fi

# Decide if enable the cache
if [[ "${ENABLE_DOCKER_REGISTRY_CACHE}" == "true" ]]; then
    REGION=$(echo "${ZONE}" | cut -f 1,2 -d -)
    echo "Enable docker registry cache at region: " $REGION
    DOCKER_OPTS="${DOCKER_OPTS} --registry-mirror='https://${REGION}.docker-cache.clustermaster.net'"
fi

if [[ -n "{DOCKER_OPTS}" ]]; then
cat <<EOF >>/etc/salt/minion.d/grains.conf
  docker_opts: '$(echo "$DOCKER_OPTS" | sed -e "s/'/''/g")'
EOF
fi

install-salt

# Wait a few minutes and trigger another Salt run to better recover from
# any transient errors.
echo "Sleeping 180"
sleep 180
salt-call state.highstate || true
