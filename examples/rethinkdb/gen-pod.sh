#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

set -o errexit
set -o nounset
set -o pipefail

: ${VERSION:=1.16.0}

readonly NAME=${1-}
if [[ -z "${NAME}" ]]; then
  echo -e "\033[1;31mName must be specified\033[0m"
  exit 1
fi

ADMIN=""
if [[ ${NAME} == "admin" ]]; then
  ADMIN="role: admin"
fi

NODE=""
if [[ ! -z "${2-}" ]]; then
  NODE="nodeSelector: { name: ${2} }"
fi

cat << EOF
apiVersion: v1beta1
namespace: rethinkdb
kind: Pod
id: rethinkdb-${NAME}-${VERSION}
${NODE}
labels:
  db: rethinkdb
  ${ADMIN}
desiredState:
  manifest:
    version: v1beta1
    id: rethinkdb
    containers:
      - name: rethinkdb
        image: antmanler/rethinkdb:${VERSION}
        ports:
          - name: admin-port
            containerPort: 8080
          - name: driver-port
            containerPort: 28015
          - name: cluster-port
            containerPort: 29015
        volumeMounts:
          - name: rethinkdb-storage
            mountPath: /data/rethinkdb_data
    volumes:
      - name: rethinkdb-storage
        source:
          hostDir:
            path: /data/db/rethinkdb
    restartPolicy:
      always: {}
EOF
