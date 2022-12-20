#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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

# This script will set up and run rsyncd to allow data to move into and out of
# our dockerized build system.  This is used for syncing sources and changes of
# sources into the docker-build-container.  It is also used to transfer built binaries
# and generated files back out.
#
# When run as root (rare) it'll preserve the file ids as sent from the client.
# Usually it'll be run as non-dockerized UID/GID and end up translating all file
# ownership to that.


set -o errexit
set -o nounset
set -o pipefail

# The directory that gets sync'd
VOLUME=${HOME}

# Assume that this is running in Docker on a bridge.  Allow connections from
# anything on the local subnet.
ALLOW=$(ip route | awk  '/^default via/ { reg = "^[0-9./]+ dev "$5 } ; $0 ~ reg { print $1 }')

CONFDIR="/tmp/rsync.k8s"
PIDFILE="${CONFDIR}/rsyncd.pid"
CONFFILE="${CONFDIR}/rsyncd.conf"
SECRETS="${CONFDIR}/rsyncd.secrets"

mkdir -p "${CONFDIR}"

if [[ -f "${PIDFILE}" ]]; then
  PID=$(cat "${PIDFILE}")
  echo "Cleaning up old PID file: ${PIDFILE}"
  kill "${PID}" &> /dev/null || true
  rm "${PIDFILE}"
fi

PASSWORD=$(</rsyncd.password)

cat <<EOF >"${SECRETS}"
k8s:${PASSWORD}
EOF
chmod go= "${SECRETS}"

USER_CONFIG=
if [[ "$(id -u)" == "0" ]]; then
  USER_CONFIG="  uid = 0"$'\n'"  gid = 0"
fi

cat <<EOF >"${CONFFILE}"
pid file = ${PIDFILE}
use chroot = no
log file = /dev/stdout
reverse lookup = no
munge symlinks = no
port = 8730
[k8s]
  numeric ids = true
  $USER_CONFIG
  hosts deny = *
  hosts allow = ${ALLOW} ${ALLOW_HOST-}
  auth users = k8s
  secrets file = ${SECRETS}
  read only = false
  path = ${VOLUME}
  filter = - /_tmp/
EOF

exec /usr/bin/rsync --no-detach --daemon --config="${CONFFILE}" "$@"
