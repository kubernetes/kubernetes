#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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
# One needs to label a node with the same key/value pair, 
# i.e., 'kubectl label nodes <node-name> name=${2}'
if [[ ! -z "${2-}" ]]; then
  NODE="nodeSelector: { name: ${2} }"
fi

cat << EOF
apiVersion: v1
kind: Pod
metadata:
  labels:
    ${ADMIN}
    db: rethinkdb
  name: rethinkdb-${NAME}-${VERSION}
  namespace: rethinkdb
spec:
  containers:
  - image: antmanler/rethinkdb:${VERSION}
    name: rethinkdb
    ports:
    - containerPort: 8080
      name: admin-port
      protocol: TCP
    - containerPort: 28015
      name: driver-port
      protocol: TCP
    - containerPort: 29015
      name: cluster-port
      protocol: TCP
    volumeMounts:
    - mountPath: /data/rethinkdb_data
      name: rethinkdb-storage
  ${NODE}
  restartPolicy: Always
  volumes:
  - hostPath:
      path: /data/db/rethinkdb
    name: rethinkdb-storage
EOF
