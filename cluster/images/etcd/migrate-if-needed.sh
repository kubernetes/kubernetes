#!/bin/sh

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

# This script performs data migration between etcd2 and etcd3 versions
# if needed.
# Expected usage of it is:
#   ./migrate_if_needed <target-storage> <data-dir>
# It will look into the contents of file <data-dir>/version.txt to
# determine the current storage version (no file means etcd2).

set -o errexit
set -o nounset

if [ -z "${TARGET_STORAGE:-}" ]; then
  echo "TARGET_USAGE variable unset - skipping migration"
  exit 0
fi

if [ -z "${DATA_DIRECTORY:-}" ]; then
  echo "DATA_DIRECTORY variable unset - skipping migration"
  exit 0
fi

ETCD="${ETCD:-/usr/local/bin/etcd}"
ETCDCTL="${ETCDCTL:-/usr/local/bin/etcdctl}"
ATTACHLEASE="${ATTACHLEASE:-/usr/local/bin/attachlease}"
VERSION_FILE="version.txt"
CURRENT_STORAGE='etcd2'
if [ -e "${DATA_DIRECTORY}/${VERSION_FILE}" ]; then
  CURRENT_STORAGE="$(cat ${DATA_DIRECTORY}/${VERSION_FILE})"
fi

start_etcd() {
  ETCD_PORT=18629
  ETCD_PEER_PORT=18630
  ${ETCD} --data-dir=${DATA_DIRECTORY} \
    --listen-client-urls http://127.0.0.1:${ETCD_PORT} \
    --advertise-client-urls http://127.0.0.1:${ETCD_PORT} \
    --listen-peer-urls http://127.0.0.1:${ETCD_PEER_PORT} \
    --initial-advertise-peer-urls http://127.0.01:${ETCD_PEER_PORT} \
    1>>/dev/null 2>&1 &
  ETCD_PID=$!
  # Wait until etcd is up.
  for i in $(seq 30); do
    local out
    if out=$(wget -q --timeout=1 http://127.0.0.1:${ETCD_PORT}/v2/machines 2> /dev/null); then
      echo "Etcd on port ${ETCD_PORT} is up."
      return 0
    fi
    sleep 0.5
  done
  echo "Timeout while waiting for etcd on port ${ETCD_PORT}"
  return 1
}

stop_etcd() {
  kill "${ETCD_PID-}" >/dev/null 2>&1 || :
  wait "${ETCD_PID-}" >/dev/null 2>&1 || :
}

if [ "${CURRENT_STORAGE}" = "etcd2" -a "${TARGET_STORAGE}" = "etcd3" ]; then
  # If directory doesn't exist or is empty, this means that there aren't any
  # data for migration, which means we can skip this step.
  if [ -d "${DATA_DIRECTORY}" ]; then
    if [ "$(ls -A ${DATA_DIRECTORY})" ]; then
      echo "Performing etcd2 -> etcd3 migration"
      ETCDCTL_API=3 ${ETCDCTL} migrate --data-dir=${DATA_DIRECTORY}
      echo "Attaching leases to TTL entries"
      # Now attach lease to all keys.
      # To do it, we temporarily start etcd on a random port (so that
      # apiserver actually cannot access it).
      start_etcd
      # Create a lease and attach all keys to it.
      ${ATTACHLEASE} \
        --etcd-address http://127.0.0.1:${ETCD_PORT} \
        --ttl-keys-prefix "${TTL_KEYS_DIRECTORY:-/registry/events}" \
        --lease-duration 1h
      # Kill etcd and wait until this is down.
      stop_etcd
    fi
  fi
fi

if [ "${CURRENT_STORAGE}" = "etcd3" -a "${TARGET_STORAGE}" = "etcd2" ]; then
  echo "Performing etcd3 -> etcd2 migration"
  # TODO: Implement rollback once this will be supported.
  echo "etcd3 -> etcd2 downgrade is NOT supported."
  # FIXME: On rollback, we will not support TTLs - we will simply clear
  # all events.
fi

# Write current storage version to avoid future migrations.
# If directory doesn't exist, we need to create it first.
mkdir -p "${DATA_DIRECTORY}"
echo "${TARGET_STORAGE}" > "${DATA_DIRECTORY}/${VERSION_FILE}"
