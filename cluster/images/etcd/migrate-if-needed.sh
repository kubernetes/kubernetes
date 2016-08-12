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
set -o pipefail

if [ -z "${TARGET_STORAGE:-}" ]; then
  echo "TARGET_USAGE variable unset - skipping migration"
  exit 0
fi

if [ -z "${DATA_DIRECTORY:-}" ]; then
  echo "DATA_DIRECTORY variable unset - skipping migration"
  exit 0
fi

VERSION_FILE="version.txt"
CURRENT_STORAGE='etcd2'
if [ -e "${DATA_DIRECTORY}/${VERSION_FILE}" ]; then
  CURRENT_STORAGE="$(cat ${DATA_DIRECTORY}/${VERSION_FILE})"
fi

if [ "${CURRENT_STORAGE}" = "etcd2" -a "${TARGET_STORAGE}" = "etcd3" ]; then
  # If directory doesn't exist or is empty, this means that there aren't any
  # data for migration, which means we can skip this step.
  if [ -d "${DATA_DIRECTORY}" ]; then
    if [ "$(ls -A ${DATA_DIRECTORY})" ]; then
      echo "Performing etcd2 -> etcd3 migration"
      # TODO: Pass a correct transformer to handle TTLs.
      ETCDCTL_API=3 /usr/local/bin/etcdctl migrate --data-dir=${DATA_DIRECTORY}
    fi
  fi
fi

if [ "${CURRENT_STORAGE}" = "etcd3" -a "${TARGET_STORAGE}" = "etcd2" ]; then
  echo "Performing etcd3 -> etcd2 migration"
  # TODO: Implement rollback once this will be supported.
  echo "etcd3 -> etcd2 downgrade is NOT supported."
fi

# Write current storage version to avoid future migrations.
# If directory doesn't exist, we need to create it first.
mkdir -p "${DATA_DIRECTORY}"
echo "${TARGET_STORAGE}" > "${DATA_DIRECTORY}/${VERSION_FILE}"
