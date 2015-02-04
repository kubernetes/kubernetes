#!/bin/bash

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
