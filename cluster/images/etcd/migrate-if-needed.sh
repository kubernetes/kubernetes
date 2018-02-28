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
# TARGET_VERSION - etcd release to be used (supported: '2.2.1', '2.3.7', '3.0.17', '3.1.11', '3.2.14')
# DATA_DIRECTORY - directory with etcd data
#
# The current etcd version and storage format is detected based on the
# contents of "${DATA_DIRECTORY}/version.txt" file (if the file doesn't
# exist, we default it to "2.2.1/etcd2".
#
# The update workflow support the following upgrade steps:
# - 2.2.1/etcd2 -> 2.3.7/etcd2
# - 2.3.7/etcd2 -> 3.0.17/etcd2
# - 3.0.17/etcd3 -> 3.1.11/etcd3
# - 3.1.11/etcd3 -> 3.2.14/etcd3
#
# NOTE: The releases supported in this script has to match release binaries
# present in the etcd image (to make this script work correctly).
#
# Based on the current etcd version and storage format we detect what
# upgrade step from this should be done to get reach target configuration

set -o errexit
set -o nounset

source $(dirname "$0")/start-stop-etcd.sh

# Rollback to previous minor version of etcd 3.x, if needed.
#
# Warning: For HA etcd clusters (any cluster with more than one member), all members must be stopped before rolling back, zero
# downtime rollbacks are not supported.
rollback_etcd3_minor_version() {
  if [ ${TARGET_MINOR_VERSION} != $((${CURRENT_MINOR_VERSION}-1)) ]; then
    echo "Rollback from ${CURRENT_VERSION} to ${TARGET_VERSION} not supported, only rollbacks to the previous minor version are supported."
    exit 1
  fi
  echo "Performing etcd ${CURRENT_VERSION} -> ${TARGET_VERSION} rollback"
  ROLLBACK_BACKUP_DIR="${DATA_DIRECTORY}.bak"
  rm -rf "${ROLLBACK_BACKUP_DIR}"
  SNAPSHOT_FILE="${DATA_DIRECTORY}.snapshot.db"
  rm -rf "${SNAPSHOT_FILE}"
  ETCD_CMD="/usr/local/bin/etcd-${CURRENT_VERSION}"
  ETCDCTL_CMD="/usr/local/bin/etcdctl-${CURRENT_VERSION}"

  # Start CURRENT_VERSION of etcd.
  START_VERSION="${CURRENT_VERSION}"
  START_STORAGE="${CURRENT_STORAGE}"
  echo "Starting etcd version ${START_VERSION} to capture rollback snapshot."
  if ! start_etcd; then
    echo "Unable to automatically downgrade etcd: starting etcd version ${START_VERSION} to capture rollback snapshot failed."
    echo "See https://coreos.com/etcd/docs/3.2.13/op-guide/recovery.html for manual downgrade options."
    exit 1
  else
    ETCDCTL_API=3 ${ETCDCTL_CMD} snapshot --endpoints "http://127.0.0.1:${ETCD_PORT}" save "${SNAPSHOT_FILE}"
  fi
  stop_etcd

  # Backup the data before rolling back.
  mv "${DATA_DIRECTORY}" "${ROLLBACK_BACKUP_DIR}"
  ETCDCTL_CMD="/usr/local/bin/etcdctl-${TARGET_VERSION}"
  NAME="etcd-$(hostname)"
  ETCDCTL_API=3 ${ETCDCTL_CMD} snapshot restore "${SNAPSHOT_FILE}" \
      --data-dir "${DATA_DIRECTORY}"  --name "${NAME}" --initial-cluster "${INITIAL_CLUSTER}"

  CURRENT_VERSION="${TARGET_VERSION}"
  echo "${CURRENT_VERSION}/${CURRENT_STORAGE}" > "${DATA_DIRECTORY}/${VERSION_FILE}"
}

# Rollback from "3.0.x" version in 'etcd3' mode to "2.2.1" version in 'etcd2' mode, if needed.
rollback_to_etcd2() {
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
}


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
if [ -z "${INITIAL_CLUSTER:-}" ]; then
  echo "Warn: INITIAL_CLUSTER variable unset - defaulting to etcd-$(hostname)=http://localhost:2380"
  INITIAL_CLUSTER="etcd-$(hostname)=http://localhost:2380"
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
SUPPORTED_VERSIONS_STRING="2.2.1 2.3.7 3.0.17 3.1.11 3.2.14"
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
ETCD_DATA_PREFIX="${ETCD_DATA_PREFIX:-/registry}"

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

ATTACHLEASE="${ATTACHLEASE:-/usr/local/bin/attachlease}"
ROLLBACK="${ROLLBACK:-/usr/local/bin/rollback}"

# If we are upgrading from 2.2.1 and this is the first try for upgrade,
# do the backup to allow restoring from it in case of failed upgrade.
BACKUP_DIR="${DATA_DIRECTORY}/migration-backup"
if [ "${CURRENT_VERSION}" = "2.2.1" -a "${CURRENT_VERSION}" != "${TARGET_VERSION}" -a ! -d "${BACKUP_DIR}" ]; then
  echo "Backup etcd before starting migration"
  mkdir ${BACKUP_DIR}
  ETCDCTL_CMD="/usr/local/bin/etcdctl-2.2.1"
  ETCDCTL_API=2 ${ETCDCTL_CMD} --debug backup --data-dir=${DATA_DIRECTORY} \
    --backup-dir=${BACKUP_DIR}
  echo "Backup done in ${BACKUP_DIR}"
fi

CURRENT_MINOR_VERSION="$(echo ${CURRENT_VERSION} | awk -F'.' '{print $2}')"
TARGET_MINOR_VERSION="$(echo ${TARGET_VERSION} | awk -F'.' '{print $2}')"

# "rollback-if-needed"
case "${CURRENT_STORAGE}-${TARGET_STORAGE}" in
  "etcd3-etcd3")
    [ ${TARGET_MINOR_VERSION} -lt ${CURRENT_MINOR_VERSION} ] && rollback_etcd3_minor_version
    break
    ;;
  "etcd3-etcd2")
    rollback_to_etcd2
    break
    ;;
  *)
    break
    ;;
esac

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
      --ttl-keys-prefix "${TTL_KEYS_DIRECTORY:-${ETCD_DATA_PREFIX}/events}" \
      --lease-duration 1h
    # Kill etcd and wait until this is down.
    stop_etcd
    CURRENT_STORAGE="etcd3"
    echo "${CURRENT_VERSION}/${CURRENT_STORAGE}" > "${DATA_DIRECTORY}/${VERSION_FILE}"
  fi
  if [ "$(echo ${CURRENT_VERSION} | cut -c1-4)" = "3.1." -a "${CURRENT_VERSION}" = "${step}" -a "${CURRENT_STORAGE}" = "etcd3" ]; then
    # If we are upgrading to 3.1.* release, if the cluster was migrated
    # from v2 version, the v2 data may still be around. So now is the
    # time to actually remove them.
    echo "Remove stale v2 data"
    START_VERSION="${step}"
    START_STORAGE="etcd3"
    ETCDCTL_CMD="${ETCDCTL:-/usr/local/bin/etcdctl-${START_VERSION}}"
    if ! start_etcd; then
      echo "Starting etcd ${step} in v3 mode failed"
      exit 1
    fi
    ${ETCDCTL_CMD} --endpoints "http://127.0.0.1:${ETCD_PORT}" rm --recursive "${ETCD_DATA_PREFIX}"
    # Kill etcd and wait until this is down.
    stop_etcd
    echo "Successfully remove v2 data"
    # Also remove backup from v2->v3 migration.
    rm -rf "${BACKUP_DIR}"
  fi
  if [ "${CURRENT_VERSION}" = "${TARGET_VERSION}" -a "${CURRENT_STORAGE}" = "${TARGET_STORAGE}" ]; then
    break
  fi
done

echo "$(date +'%Y-%m-%d %H:%M:%S') Migration finished"
