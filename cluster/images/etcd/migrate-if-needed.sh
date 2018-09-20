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

# NOTES
# This script performs etcd upgrade based on the following environmental
# variables:
# TARGET_STORAGE - API of etcd to be used (supported: 'etcd2', 'etcd3')
# TARGET_VERSION - etcd release to be used (supported: '2.2.1', '2.3.7', '3.0.17', '3.1.12', '3.2.24')
# DATA_DIRECTORY - directory with etcd data
#
# The current etcd version and storage format is detected based on the
# contents of "${DATA_DIRECTORY}/version.txt" file (if the file doesn't
# exist, we default it to "2.2.1/etcd2".
#
# The update workflow support the following upgrade steps:
# - 2.2.1/etcd2 -> 2.3.7/etcd2
# - 2.3.7/etcd2 -> 3.0.17/etcd2
# - 3.0.17/etcd3 -> 3.1.12/etcd3
# - 3.1.12/etcd3 -> 3.2.24/etcd3
#
# NOTE: The releases supported in this script has to match release binaries
# present in the etcd image (to make this script work correctly).
#
# Based on the current etcd version and storage format we detect what
# upgrade step from this should be done to get reach target configuration

set -o errexit
set -o nounset

# NOTE: BUNDLED_VERSION has to match release binaries present in the
# etcd image (to make this script work correctly).
BUNDLED_VERSIONS="2.2.1, 2.3.7, 3.0.17, 3.1.12, 3.2.24"

ETCD_NAME="${ETCD_NAME:-etcd-$(hostname)}"
if [ -z "${DATA_DIRECTORY:-}" ]; then
  echo "DATA_DIRECTORY variable unset - unexpected failure"
  exit 1
fi

case "${DATA_DIRECTORY}" in
  *event*)
    ETCD_PEER_PORT=2381
    ETCD_CLIENT_PORT=18631
    ;;
  *)
    ETCD_PEER_PORT=2380
    ETCD_CLIENT_PORT=18629
    ;;
esac

if [ -z "${INITIAL_CLUSTER:-}" ]; then
  echo "Warn: INITIAL_CLUSTER variable unset - defaulting to ${ETCD_NAME}=http://localhost:${ETCD_PEER_PORT}"
  INITIAL_CLUSTER="${ETCD_NAME}=http://localhost:${ETCD_PEER_PORT}"
fi
if [ -z "${LISTEN_PEER_URLS:-}" ]; then
  echo "Warn: LISTEN_PEER_URLS variable unset - defaulting to http://localhost:${ETCD_PEER_PORT}"
  LISTEN_PEER_URLS="http://localhost:${ETCD_PEER_PORT}"
fi
if [ -z "${INITIAL_ADVERTISE_PEER_URLS:-}" ]; then
  echo "Warn: INITIAL_ADVERTISE_PEER_URLS variable unset - defaulting to http://localhost:${ETCD_PEER_PORT}"
  INITIAL_ADVERTISE_PEER_URLS="http://localhost:${ETCD_PEER_PORT}"
fi
if [ -z "${TARGET_VERSION:-}" ]; then
  echo "TARGET_VERSION variable unset - unexpected failure"
  exit 1
fi
if [ -z "${TARGET_STORAGE:-}" ]; then
  echo "TARGET_STORAGE variable unset - unexpected failure"
  exit 1
fi
ETCD_DATA_PREFIX="${ETCD_DATA_PREFIX:-/registry}"
ETCD_CREDS="${ETCD_CREDS:-}"

# Correctly support upgrade and rollback to non-default version.
if [ "${DO_NOT_MOVE_BINARIES:-}" != "true" ]; then
  cp "/usr/local/bin/etcd-${TARGET_VERSION}" "/usr/local/bin/etcd"
  cp "/usr/local/bin/etcdctl-${TARGET_VERSION}" "/usr/local/bin/etcdctl"
fi

/usr/local/bin/migrate \
  --name "${ETCD_NAME}" \
  --port "${ETCD_CLIENT_PORT}" \
  --listen-peer-urls "${LISTEN_PEER_URLS}" \
  --initial-advertise-peer-urls "${INITIAL_ADVERTISE_PEER_URLS}" \
  --data-dir "${DATA_DIRECTORY}" \
  --bundled-versions "${BUNDLED_VERSIONS}" \
  --initial-cluster "${INITIAL_CLUSTER}" \
  --target-version "${TARGET_VERSION}" \
  --target-storage "${TARGET_STORAGE}" \
  --etcd-data-prefix "${ETCD_DATA_PREFIX}" \
  --ttl-keys-directory "${TTL_KEYS_DIRECTORY:-${ETCD_DATA_PREFIX}/events}" \
  --etcd-server-extra-args "${ETCD_CREDS}"
