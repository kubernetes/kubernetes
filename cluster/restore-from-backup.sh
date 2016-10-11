#!/bin/bash

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
# - version.txt file is in the current directy (if it isn't it will be
#     defaulted to "2.2.1/etcd2"). Based on this file, the script will
#     decide to which version we are restoring (procedures are different
#     for etcd2 and etcd3).
# - in case of etcd2 - *.snap and *.wal files are in current directory
# - in case of etcd3 - *.db file is in the current directory
# - the script is run as root
#
# FIXME: The script currently doesn't support restoring of event etcd.


set -o errexit
set -o nounset
set -o pipefail

# Version file contains information about current version in the format:
# <etcd binary version>/<etcd api mode> (e.g. "3.0.12/etcd3").
#
# If the file doesn't exist we assume "2.2.1/etcd2" configuration is
# the current one and create a file with such configuration.
# The restore procedure is choosed based on this information.
VERSION_FILE="version.txt"
if [ ! -f "${VERSION_FILE}" ]; then
  echo "2.2.1/etcd2" > "${VERSION_FILE}"
fi
VERSION_CONTENTS="$(cat ${VERSION_FILE})"
ETCD_VERSION="$(echo $VERSION_CONTENTS | cut -d '/' -f 1)"
ETCD_API="$(echo $VERSION_CONTENTS | cut -d '/' -f 2)"

# Name is used only in case of etcd3 mode, to appropriate set the metadata
# for the etcd data.
# NOTE: NAME HAS TO BE EQUAL TO WHAT WE USE IN --name flag when starting etcd.
NAME="${NAME:-etcd-$(hostname)}"

# Wait until both etcd instances are up
wait_for_etcd_up() {
  health_ok="{\"health\": \"true\"}"
  for i in $(seq 120); do
    # TODO: Is it enough to look into /health endpoint?
    etcd=$(curl --silent http://localhost:2379/health)
    etcd_events=$(curl --silent http://localhost:4002/health)
    if [ "${etcd}" == "${health_ok}" -a "${etcd-events}" == "${health_ok}" ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Wait until apiserver is up.
wait_for_apiserver_up() {
  for i in $(seq 120); do
    kubectl get componentstatuses 1>/dev/null 2>&1
    if [ "$?" -eq "0" ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Wait until etcd and apiserver pods are down.
wait_for_etcd_and_apiserver_down() {
  for i in $(seq 120); do
    etcd=$(docker ps | grep etcd | grep -v etcd-empty-dir | wc -l)
    apiserver=$(docker ps | grep apiserver | wc -l)
    # TODO: Theoretically it is possible, that apiserver and or etcd
    # are currently down, but Kubelet is now restarting them and they
    # will reappear again. We should avoid it.
    if [ "${etcd}" -eq "0" -a "${apiserver}" -eq "0" ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Move the manifest files to stop etcd and kube-apiserver
# while we swap the data out from under them.
mv /etc/kubernetes/manifests/kube-apiserver.manifest ./
mv /etc/kubernetes/manifests/etcd.manifest ./
mv /etc/kubernetes/manifests/etcd-events.manifest ./

# Wait for the pods to be stopped
echo "Waiting for etcd and kube-apiserver to be down"
if ! wait_for_etcd_and_apiserver_down; then
  # Couldn't kill etcd and apiserver.
  echo "Downing etcd and apiserver failed"
  exit 1
fi

# Create the sort of directory structure that etcd expects.
BACKUP_DIR="/var/tmp/backup"
if [ "${ETCD_API}" == "etcd2" ]; then
  echo "Preparing etcd backup data for restore"
  # In v2 mode, we simply copy both snap and wal files to a newly created
  # directory. After that, we start etcd with --force-new-cluster option
  # that (according to the etcd documentation) is required to recover from
  # a backup.
  mkdir -p "${BACKUP_DIR}/member/snap"
  mkdir -p "${BACKUP_DIR}/member/wal"
  # If the cluster is relatively new, there can be no .snap file.
  mv *.snap "${BACKUP_DIR}/member/snap/" || true
  mv *.wal "${BACKUP_DIR}/member/wal/"

  # TODO(jsz): This won't work with HA setups (e.g. do we need to set --name flag)?
  image=$(docker run -d -v ${BACKUP_DIR}:/var/etcd/data \
    "gcr.io/google_containers/etcd:${ETCD_VERSION}" /bin/sh -c \
    "/usr/local/bin/etcd --data-dir /var/etcd/data --force-new-cluster")

  # FIXME: Wait until this etcd is up.
  sleep 10

  # Kill that etcd instance.
  docker kill "${image}"
elif [ "${ETCD_API}" == "etcd3" ]; then
  echo "Preparing etcd snapshot for restore"
  mkdir -p "${BACKUP_DIR}"
  mv *.db "${BACKUP_DIR}/"
  # FIXME: Ensure that there is exactly one file.
  snapshot="$(ls ${BACKUP_DIR})"

  # Run etcdctl snapshot restore command and wait until it is finished.
  # setting with --name in the etcd manifest file and then it seems to work.
  # TODO(jsz): This command may not work in case of HA.
  image=$(docker run -d -v ${BACKUP_DIR}:/var/tmp/backup --env ETCDCTL_API=3 \
    "gcr.io/google_containers/etcd:${ETCD_VERSION}" /bin/sh -c \
    "/usr/local/bin/etcdctl snapshot restore ${BACKUP_DIR}/${snapshot} --name ${NAME} --initial-cluster ${NAME}=http://localhost:2380; mv /${NAME}.etcd/member /var/tmp/backup/")
  echo "Prepare container exit code: $(docker wait ${image})"

  rm -f "${BACKUP_DIR}/${snapshot}"
fi
# Also copy version.txt file
cp "${VERSION_FILE}" "${BACKUP_DIR}"

# Find out if we are running GCI vs CVM
export CVM=$(curl "http://metadata/computeMetadata/v1/instance/attributes/" -H "Metadata-Flavor: Google" |& grep -q gci; echo $?)
if [[ "$CVM" == "1" ]]; then
  export MNT_DISK="/mnt/master-pd"
else
  export MNT_DISK="/mnt/disks/master-pd"
fi

# Save the corrupted data (clean directory if it is already non-empty)
rm -rf "${MNT_DISK}/var/etcd-corrupted"
mkdir -p "${MNT_DISK}/var/etcd-corrupted"
mv /var/etcd/data "${MNT_DISK}/var/etcd-corrupted"

# Replace the corrupted data dir with the resotred data
mv "${BACKUP_DIR}" /var/etcd/data

# Start etcd and kube-apiserver again.
echo "Restarting etcd and apiserver from restored snapshot"
mv ./etcd.manifest /etc/kubernetes/manifests/
mv ./etcd-events.manifest /etc/kubernetes/manifests/
mv ./kube-apiserver.manifest /etc/kubernetes/manifests/

# Verify that etcd is back
echo "Waiting for etcd to come back"
if ! wait_for_etcd_up; then
  echo "Etcd didn't come back correctly"
  exit 1
fi

# Verify that kube-apiserver is back.
echo "Waiting for apiserver to come back"
if ! wait_for_apiserver_up; then
  echo "Apiserver didn't come back correctly"
  exit 1
fi
