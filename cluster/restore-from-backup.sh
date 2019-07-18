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

# This script performs disaster recovery of etcd from the backup data.
# Assumptions:
# - backup was done using etcdctl command:
#   a) in case of etcd2
#      $ etcdctl backup --data-dir=<dir>
#      produced .snap and .wal files
#   b) in case of etcd3
#      $ etcdctl --endpoints=<address> snapshot save
#      produced .db file
# - version.txt file is in the current directory (if it isn't it will be
#     defaulted to "3.0.17/etcd3"). Based on this file, the script will
#     decide to which version we are restoring (procedures are different
#     for etcd2 and etcd3).
# - in case of etcd2 - *.snap and *.wal files are in current directory
# - in case of etcd3 - *.db file is in the current directory
# - the script is run as root
# - for event etcd, we only support clearing it - to do it, you need to
#   set RESET_EVENT_ETCD=true env var.

set -o errexit
set -o nounset
set -o pipefail

# Version file contains information about current version in the format:
# <etcd binary version>/<etcd api mode> (e.g. "3.0.12/etcd3").
#
# If the file doesn't exist we assume "3.0.17/etcd3" configuration is
# the current one and create a file with such configuration.
# The restore procedure is chosen based on this information.
VERSION_FILE="version.txt"

# Make it possible to overwrite version file (or default version)
# with VERSION_CONTENTS env var.
if [ -n "${VERSION_CONTENTS:-}" ]; then
  echo "${VERSION_CONTENTS}" > "${VERSION_FILE}"
fi
if [ ! -f "${VERSION_FILE}" ]; then
  echo "3.0.17/etcd3" > "${VERSION_FILE}"
fi
VERSION_CONTENTS="$(cat ${VERSION_FILE})"
ETCD_VERSION="$(echo "$VERSION_CONTENTS" | cut -d '/' -f 1)"
ETCD_API="$(echo "$VERSION_CONTENTS" | cut -d '/' -f 2)"

# Name is used only in case of etcd3 mode, to appropriate set the metadata
# for the etcd data.
# NOTE: NAME HAS TO BE EQUAL TO WHAT WE USE IN --name flag when starting etcd.
NAME="${NAME:-etcd-$(hostname)}"

INITIAL_CLUSTER="${INITIAL_CLUSTER:-${NAME}=http://localhost:2380}"
INITIAL_ADVERTISE_PEER_URLS="${INITIAL_ADVERTISE_PEER_URLS:-http://localhost:2380}"

# Port on which etcd is exposed.
etcd_port=2379
event_etcd_port=4002

# Wait until both etcd instances are up
wait_for_etcd_up() {
  port=$1
  # TODO: As of 3.0.x etcd versions, all 2.* and 3.* versions return
  # {"health": "true"} on /health endpoint in healthy case.
  # However, we should come with a regex for it to avoid future break.
  health_ok="{\"health\": \"true\"}"
  for _ in $(seq 120); do
    # TODO: Is it enough to look into /health endpoint?
    health=$(curl --silent "http://127.0.0.1:${port}/health")
    if [ "${health}" == "${health_ok}" ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Wait until apiserver is up.
wait_for_cluster_healthy() {
  for _ in $(seq 120); do
    cs_status=$(kubectl get componentstatuses -o template --template='{{range .items}}{{with index .conditions 0}}{{.type}}:{{.status}}{{end}}{{"\n"}}{{end}}') || true
    componentstatuses=$(echo "${cs_status}" | grep -c 'Healthy:') || true
    healthy=$(echo "${cs_status}" | grep -c 'Healthy:True') || true
    if [ "${componentstatuses}" -eq "${healthy}" ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Wait until etcd and apiserver pods are down.
wait_for_etcd_and_apiserver_down() {
  for _ in $(seq 120); do
    etcd=$(docker ps | grep -c etcd-server)
    apiserver=$(docker ps | grep -c apiserver)
    # TODO: Theoretically it is possible, that apiserver and or etcd
    # are currently down, but Kubelet is now restarting them and they
    # will reappear again. We should avoid it.
    if [ "${etcd}" -eq "0" ] && [ "${apiserver}" -eq "0" ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Move the manifest files to stop etcd and kube-apiserver
# while we swap the data out from under them.
MANIFEST_DIR="/etc/kubernetes/manifests"
MANIFEST_BACKUP_DIR="/etc/kubernetes/manifests-backups"
mkdir -p "${MANIFEST_BACKUP_DIR}"
echo "Moving etcd(s) & apiserver manifest files to ${MANIFEST_BACKUP_DIR}"
# If those files were already moved (e.g. during previous
# try of backup) don't fail on it.
mv "${MANIFEST_DIR}/kube-apiserver.manifest" "${MANIFEST_BACKUP_DIR}" || true
mv "${MANIFEST_DIR}/etcd.manifest" "${MANIFEST_BACKUP_DIR}" || true
mv "${MANIFEST_DIR}/etcd-events.manifest" "${MANIFEST_BACKUP_DIR}" || true

# Wait for the pods to be stopped
echo "Waiting for etcd and kube-apiserver to be down"
if ! wait_for_etcd_and_apiserver_down; then
  # Couldn't kill etcd and apiserver.
  echo "Downing etcd and apiserver failed"
  exit 1
fi

read -rsp $'Press enter when all etcd instances are down...\n'

# Create the sort of directory structure that etcd expects.
# If this directory already exists, remove it.
BACKUP_DIR="/var/tmp/backup"
rm -rf "${BACKUP_DIR}"
if [ "${ETCD_API}" == "etcd2" ]; then
  echo "Preparing etcd backup data for restore"
  # In v2 mode, we simply copy both snap and wal files to a newly created
  # directory. After that, we start etcd with --force-new-cluster option
  # that (according to the etcd documentation) is required to recover from
  # a backup.
  echo "Copying data to ${BACKUP_DIR} and restoring there"
  mkdir -p "${BACKUP_DIR}/member/snap"
  mkdir -p "${BACKUP_DIR}/member/wal"
  # If the cluster is relatively new, there can be no .snap file.
  mv ./*.snap "${BACKUP_DIR}/member/snap/" || true
  mv ./*.wal "${BACKUP_DIR}/member/wal/"

  # TODO(jsz): This won't work with HA setups (e.g. do we need to set --name flag)?
  echo "Starting etcd ${ETCD_VERSION} to restore data"
  if ! image=$(docker run -d -v ${BACKUP_DIR}:/var/etcd/data \
    --net=host -p ${etcd_port}:${etcd_port} \
    "k8s.gcr.io/etcd:${ETCD_VERSION}" /bin/sh -c \
    "/usr/local/bin/etcd --data-dir /var/etcd/data --force-new-cluster"); then
    echo "Docker container didn't started correctly"
    exit 1
  fi
  echo "Container ${image} created, waiting for etcd to report as healthy"

  if ! wait_for_etcd_up "${etcd_port}"; then
    echo "Etcd didn't come back correctly"
    exit 1
  fi

  # Kill that etcd instance.
  echo "Etcd healthy - killing ${image} container"
  docker kill "${image}"
elif [ "${ETCD_API}" == "etcd3" ]; then
  echo "Preparing etcd snapshot for restore"
  mkdir -p "${BACKUP_DIR}"
  echo "Copying data to ${BACKUP_DIR} and restoring there"
  number_files=$(find . -maxdepth 1 -type f -name "*.db" | wc -l)
  if [ "${number_files}" -ne "1" ]; then
    echo "Incorrect number of *.db files - expected 1"
    exit 1
  fi
  mv ./*.db "${BACKUP_DIR}/"
  snapshot="$(ls ${BACKUP_DIR})"

  # Run etcdctl snapshot restore command and wait until it is finished.
  # setting with --name in the etcd manifest file and then it seems to work.
  if ! docker run -v ${BACKUP_DIR}:/var/tmp/backup --env ETCDCTL_API=3 \
    "k8s.gcr.io/etcd:${ETCD_VERSION}" /bin/sh -c \
    "/usr/local/bin/etcdctl snapshot restore ${BACKUP_DIR}/${snapshot} --name ${NAME} --initial-cluster ${INITIAL_CLUSTER} --initial-advertise-peer-urls ${INITIAL_ADVERTISE_PEER_URLS}; mv /${NAME}.etcd/member /var/tmp/backup/"; then
    echo "Docker container didn't started correctly"
    exit 1
  fi

  rm -f "${BACKUP_DIR}/${snapshot}"
fi
# Also copy version.txt file.
cp "${VERSION_FILE}" "${BACKUP_DIR}"

export MNT_DISK="/mnt/disks/master-pd"

# Save the corrupted data (clean directory if it is already non-empty).
rm -rf "${MNT_DISK}/var/etcd-corrupted"
mkdir -p "${MNT_DISK}/var/etcd-corrupted"
echo "Saving corrupted data to ${MNT_DISK}/var/etcd-corrupted"
mv /var/etcd/data "${MNT_DISK}/var/etcd-corrupted"

# Replace the corrupted data dir with the restored data.
echo "Copying restored data to /var/etcd/data"
mv "${BACKUP_DIR}" /var/etcd/data

if [ "${RESET_EVENT_ETCD:-}" == "true" ]; then
  echo "Removing event-etcd corrupted data"
  EVENTS_CORRUPTED_DIR="${MNT_DISK}/var/etcd-events-corrupted"
  # Save the corrupted data (clean directory if it is already non-empty).
  rm -rf "${EVENTS_CORRUPTED_DIR}"
  mkdir -p "${EVENTS_CORRUPTED_DIR}"
  mv /var/etcd/data-events "${EVENTS_CORRUPTED_DIR}"
fi

# Start etcd and kube-apiserver again.
echo "Restarting etcd and apiserver from restored snapshot"
mv "${MANIFEST_BACKUP_DIR}"/* "${MANIFEST_DIR}/"
rm -rf "${MANIFEST_BACKUP_DIR}"

# Verify that etcd is back.
echo "Waiting for etcd to come back"
if ! wait_for_etcd_up "${etcd_port}"; then
  echo "Etcd didn't come back correctly"
  exit 1
fi

# Verify that event etcd is back.
echo "Waiting for event etcd to come back"
if ! wait_for_etcd_up "${event_etcd_port}"; then
  echo "Event etcd didn't come back correctly"
  exit 1
fi

# Verify that kube-apiserver is back and cluster is healthy.
echo "Waiting for apiserver to come back"
if ! wait_for_cluster_healthy; then
  echo "Apiserver didn't come back correctly"
  exit 1
fi

echo "Cluster successfully restored!"
