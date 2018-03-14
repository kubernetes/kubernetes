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

# Starts 'etcd' version ${START_VERSION} and writes to it:
# 'etcd_version' -> "${START_VERSION}"
# Successful write confirms that etcd is up and running.
# Sets ETCD_PID at the end.
# Returns 0 if etcd was successfully started, non-0 otherwise.
start_etcd() {
  # Use random ports, so that apiserver cannot connect to etcd.
  ETCD_PORT=18629
  ETCD_PEER_PORT=2380
  # Avoid collisions between etcd and event-etcd.
  case "${DATA_DIRECTORY}" in
    *event*)
      ETCD_PORT=18631
      ETCD_PEER_PORT=2381
      ;;
  esac
  local ETCD_CMD="${ETCD:-/usr/local/bin/etcd-${START_VERSION}}"
  local ETCDCTL_CMD="${ETCDCTL:-/usr/local/bin/etcdctl-${START_VERSION}}"
  local API_VERSION="$(echo ${START_STORAGE} | cut -c5-5)"
  if [ "${API_VERSION}" = "2" ]; then
    ETCDCTL_CMD="${ETCDCTL_CMD} --debug --endpoint=http://127.0.0.1:${ETCD_PORT} set"
  else
    ETCDCTL_CMD="${ETCDCTL_CMD} --endpoints=http://127.0.0.1:${ETCD_PORT} put"
  fi
  ${ETCD_CMD} \
    --name="etcd-$(hostname)" \
    --initial-cluster="etcd-$(hostname)=http://127.0.0.1:${ETCD_PEER_PORT}" \
    --debug \
    --data-dir=${DATA_DIRECTORY} \
    --listen-client-urls http://127.0.0.1:${ETCD_PORT} \
    --advertise-client-urls http://127.0.0.1:${ETCD_PORT} \
    --listen-peer-urls http://127.0.0.1:${ETCD_PEER_PORT} \
    --initial-advertise-peer-urls http://127.0.0.1:${ETCD_PEER_PORT} &
  ETCD_PID=$!
  # Wait until we can write to etcd.
  for i in $(seq 240); do
    sleep 0.5
    ETCDCTL_API="${API_VERSION}" ${ETCDCTL_CMD} 'etcd_version' ${START_VERSION}
    if [ "$?" -eq "0" ]; then
      echo "Etcd on port ${ETCD_PORT} is up."
      return 0
    fi
  done
  echo "Timeout while waiting for etcd on port ${ETCD_PORT}"
  return 1
}

# Stops etcd with ${ETCD_PID} pid.
stop_etcd() {
  kill "${ETCD_PID-}" >/dev/null 2>&1 || :
  wait "${ETCD_PID-}" >/dev/null 2>&1 || :
}
