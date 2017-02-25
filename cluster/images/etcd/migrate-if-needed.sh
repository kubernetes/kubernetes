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
# TARGET_VERSION - etcd release to be used (supported: '2.2.1', '2.3.7', '3.0.17')
# DATA_DIRECTORY - directory with etcd data
#
# The current etcd version and storage format is detected based on the
# contents of "${DATA_DIRECTORY}/version.txt" file (if the file doesn't
# exist, we default it to "2.2.1/etcd2".
#
# The update workflow support the following upgrade steps:
# - 2.2.1/etcd2 -> 2.3.7/etcd2
# - 2.3.7/etcd2 -> 3.0.17/etcd2
# - 3.0.17/etcd2 -> 3.0.17/etcd3
#
# NOTE: The releases supported in this script has to match release binaries
# present in the etcd image (to make this script work correctly).
#
# Based on the current etcd version and storage format we detect what
# upgrade step from this should be done to get reach target configuration

set -o errexit
set -o nounset

if [ -z "${TARGET_STORAGE:-}" ]; then
  echo "TARGET_STORAGE variable unset - unexpected failure"
  exit 1
fi
if [ -z "${TARGET_VERSION:-}" ]; then
  echo "TARGET_VERSION variable unset - unexpected failure"
  exit 1
fi
if [ -z "${DATA_DIRECTORY:-}" ]; then
  echo "DATA_DIRECTORY variable unset - unexpected failure"
  exit 1
fi

echo "$(date +'%Y-%m-%d %H:%M:%S') Detecting if migration is needed"

if [ "${TARGET_STORAGE}" != "etcd2" -a "${TARGET_STORAGE}" != "etcd3" ]; then
  echo "Not supported version of storage: ${TARGET_STORAGE}"
  exit 1
fi

# Correctly support upgrade and rollback to non-default version.
if [ "${DO_NOT_MOVE_BINARIES:-}" != "true" ]; then
  cp "/usr/local/bin/etcd-${TARGET_VERSION}" "/usr/local/bin/etcd"
  cp "/usr/local/bin/etcdctl-${TARGET_VERSION}" "/usr/local/bin/etcdctl"
fi

# NOTE: SUPPORTED_VERSION has to match release binaries present in the
# etcd image (to make this script work correctly).
# We cannot use array since sh doesn't support it.
SUPPORTED_VERSIONS_STRING="2.2.1 2.3.7 3.0.17"
SUPPORTED_VERSIONS=$(echo "${SUPPORTED_VERSIONS_STRING}" | tr " " "\n")

VERSION_FILE="version.txt"
CURRENT_STORAGE="etcd2"
CURRENT_VERSION="2.2.1"
if [ -e "${DATA_DIRECTORY}/${VERSION_FILE}" ]; then
  VERSION_CONTENTS="$(cat ${DATA_DIRECTORY}/${VERSION_FILE})"
  # Example usage: if contents of VERSION_FILE is 2.3.7/etcd2, then
  # - CURRENT_VERSION would be '2.3.7'
  # - CURRENT_STORAGE would be 'etcd2'
  CURRENT_VERSION="$(echo $VERSION_CONTENTS | cut -d '/' -f 1)"
  CURRENT_STORAGE="$(echo $VERSION_CONTENTS | cut -d '/' -f 2)"
fi

# If there is no data in DATA_DIRECTORY, this means that we are
# starting etcd from scratch. In that case, we don't need to do
# any migration.
if [ ! -d "${DATA_DIRECTORY}" ]; then
  mkdir -p "${DATA_DIRECTORY}"
fi
if [ -z "$(ls -A ${DATA_DIRECTORY})" ]; then
  echo "${DATA_DIRECTORY} is empty - skipping migration"
  echo "${TARGET_VERSION}/${TARGET_STORAGE}" > "${DATA_DIRECTORY}/${VERSION_FILE}"
  exit 0
fi

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
    --debug \
    --force-new-cluster \
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

ATTACHLEASE="${ATTACHLEASE:-/usr/local/bin/attachlease}"
ROLLBACK="${ROLLBACK:-/usr/local/bin/rollback}"

# If we are upgrading from 2.2.1 and this is the first try for upgrade,
# do the backup to allow restoring from it in case of failed upgrade.
BACKUP_DIR="${DATA_DIRECTORY}/migration-backup"
if [ "${CURRENT_VERSION}" = "2.2.1" -a ! "${CURRENT_VERSION}" != "${TARGET_VERSION}" -a -d "${BACKUP_DIR}" ]; then
  echo "Backup etcd before starting migration"
  mkdir ${BACKUP_DIR}
  ETCDCTL_CMD="/usr/local/bin/etcdctl-2.2.1"
  ETCDCTL_API=2 ${ETCDCTL_CMD} --debug backup --data-dir=${DATA_DIRECTORY} \
    --backup-dir=${BACKUP_DIR}
  echo "Backup done in ${BACKUP_DIR}"
fi

# Do the roll-forward migration if needed.
# The migration goes as following:
# 1. for all versions starting one after the current version of etcd
#    we do "start, wait until healthy and stop etcd". This is the
#    procedure that etcd documentation suggests for upgrading binaries.
# 2. For the first 3.0.x version that we encounter, if we are still in
#    v2 API, we do upgrade to v3 API using the "etcdct migrate" and
#    attachlease commands.
SKIP_STEP=true
for step in ${SUPPORTED_VERSIONS}; do
  if [ "${step}" = "${CURRENT_VERSION}" ]; then
    SKIP_STEP=false
  elif [ "${SKIP_STEP}" != "true" ]; then
    # Do the migration step, by just starting etcd in this version.
    START_VERSION="${step}"
    START_STORAGE="${CURRENT_STORAGE}"
    if ! start_etcd; then
      # Migration failed.
      echo "Starting etcd ${step} failed"
      exit 1
    fi
    # Kill etcd and wait until this is down.
    stop_etcd
    CURRENT_VERSION=${step}
    echo "${CURRENT_VERSION}/${CURRENT_STORAGE}" > "${DATA_DIRECTORY}/${VERSION_FILE}"
  fi
  if [ "$(echo ${CURRENT_VERSION} | cut -c1-2)" = "3." -a "${CURRENT_VERSION}" = "${step}" -a "${CURRENT_STORAGE}" = "etcd2" -a "${TARGET_STORAGE}" = "etcd3" ]; then
    # If it is the first 3.x release in the list and we are migrating
    # also from 'etcd2' to 'etcd3', do the migration now.
    echo "Performing etcd2 -> etcd3 migration"
    START_VERSION="${step}"
    START_STORAGE="etcd3"
    ETCDCTL_CMD="${ETCDCTL:-/usr/local/bin/etcdctl-${START_VERSION}}"
    ETCDCTL_API=3 ${ETCDCTL_CMD} migrate --data-dir=${DATA_DIRECTORY}
    echo "Attaching leases to TTL entries"
    # Now attach lease to all keys.
    # To do it, we temporarily start etcd on a random port (so that
    # apiserver actually cannot access it).
    if ! start_etcd; then
      echo "Starting etcd ${step} in v3 mode failed"
      exit 1
    fi
    # Create a lease and attach all keys to it.
    ${ATTACHLEASE} \
      --etcd-address http://127.0.0.1:${ETCD_PORT} \
      --ttl-keys-prefix "${TTL_KEYS_DIRECTORY:-/registry/events}" \
      --lease-duration 1h
    # Kill etcd and wait until this is down.
    stop_etcd
    CURRENT_STORAGE="etcd3"
    echo "${CURRENT_VERSION}/${CURRENT_STORAGE}" > "${DATA_DIRECTORY}/${VERSION_FILE}"
  fi
  if [ "${CURRENT_VERSION}" = "${TARGET_VERSION}" -a "${CURRENT_STORAGE}" = "${TARGET_STORAGE}" ]; then
    break
  fi
done

# Do the rollback of needed.
# NOTE: Rollback is only supported from "3.0.x" version in 'etcd3' mode to
# "2.2.1" version in 'etcd2' mode.
if [ "${CURRENT_STORAGE}" = "etcd3" -a "${TARGET_STORAGE}" = "etcd2" ]; then
  if [ "$(echo ${CURRENT_VERSION} | cut -c1-4)" != "3.0." -o "${TARGET_VERSION}" != "2.2.1" ]; then
    echo "etcd3 -> etcd2 downgrade is supported only between 3.0.x and 2.2.1"
    return 0
  fi
  echo "Backup and remove all existing v2 data"
  ROLLBACK_BACKUP_DIR="${DATA_DIRECTORY}.bak"
  rm -rf "${ROLLBACK_BACKUP_DIR}"
  mkdir -p "${ROLLBACK_BACKUP_DIR}"
  cp -r "${DATA_DIRECTORY}" "${ROLLBACK_BACKUP_DIR}"
  echo "Performing etcd3 -> etcd2 rollback"
  ${ROLLBACK} --data-dir "${DATA_DIRECTORY}"
  if [ "$?" -ne "0" ]; then
    echo "Rollback to etcd2 failed"
    exit 1
  fi
  CURRENT_STORAGE="etcd2"
  CURRENT_VERSION="2.2.1"
  echo "${CURRENT_VERSION}/${CURRENT_STORAGE}" > "${DATA_DIRECTORY}/${VERSION_FILE}"
fi

echo "$(date +'%Y-%m-%d %H:%M:%S') Migration finished"
