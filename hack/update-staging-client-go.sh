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

set -o errexit
set -o nounset
set -o pipefail

V=""
IN_PLACE="0"
COPY_FLAGS="-u"
while getopts ":vid" opt; do
  case $opt in
    v) # increase verbosity
      V="-v"
      ;;
    i)
      IN_PLACE=1
      ;;
    d)
	  COPY_FLAGS+=" -d"
	  ;;
    f)
	  COPY_FLAGS+=" -f"
	  ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done
readonly V IN_PLACE

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
cd ${KUBE_ROOT}

# Create temporary GOPATH to run godep-restore into
if [ "${IN_PLACE}" != 1 ]; then
    GODEP_RESTORE_TMPDIR=$(mktemp -d -t verify-staging-client-go.XXXXX)
    echo "Creating a temporary GOPATH directory for godep-restore: ${GODEP_RESTORE_TMPDIR}"
    cleanup() {
        if [ "${KEEP_TEMP_DIR:-0}" != 1 ]; then
            rm -rf "${GODEP_RESTORE_TMPDIR}"
        fi
    }
    trap cleanup EXIT SIGINT
    mkdir -p "${GODEP_RESTORE_TMPDIR}/src"
    GODEP_RESTORE_GOPATH="${GODEP_RESTORE_TMPDIR}:${GOPATH}"
else
    GODEP_RESTORE_GOPATH="${GOPATH}"
fi

echo "Running godep restore"
GOPATH="${GODEP_RESTORE_GOPATH}" godep restore ${V} 2>&1 | sed 's/^/  /'

echo "Running staging/copy.sh"
eval staging/copy.sh ${COPY_FLAGS} 2>&1 | sed 's/^/  /'
