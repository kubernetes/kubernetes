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
while getopts ":v" opt; do
  case $opt in
    v) # increase verbosity
      V="-v"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done
readonly V

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
cd ${KUBE_ROOT}

# Smoke test client-go examples
go install ./staging/src/k8s.io/client-go/examples/...

# Create a temporary GOPATH for apimachinery and client-go, copy the current HEAD into each and turn them
# into a git repo. Then we can run copy.sh with this additional GOPATH. The Godeps.json will
# have invalid git SHA1s, but it's good enough as a smoke test of copy.sh.
if [ "${USE_TEMP_DIR:-1}" = 1 ]; then
    TEMP_STAGING_GOPATH=$(mktemp -d -t verify-staging-client-go.XXXXX)
    echo "Creating the temporary staging GOPATH directory: ${TEMP_STAGING_GOPATH}"
    cleanup() {
        if [ "${KEEP_TEMP_DIR:-0}" != 1 ]; then
            rm -rf "${TEMP_STAGING_GOPATH}"
        fi
    }
    trap cleanup EXIT SIGINT
    mkdir -p "${TEMP_STAGING_GOPATH}/src/k8s.io"
    ln -s "${PWD}" "${TEMP_STAGING_GOPATH}/src/k8s.io"
else
    TEMP_STAGING_GOPATH="${GOPATH}"
fi
for PACKAGE in apimachinery client-go; do
    PACKAGE_PATH="${TEMP_STAGING_GOPATH}/src/k8s.io/${PACKAGE}"
    echo "Creating a temporary ${PACKAGE} repo with a snapshot of HEAD"
    mkdir -p "${PACKAGE_PATH}"
    rsync -ax --delete staging/src/k8s.io/${PACKAGE}/ "${PACKAGE_PATH}/"
    pushd "${PACKAGE_PATH}" >/dev/null
    git init >/dev/null
    git add *
    git -c user.email="nobody@k8s.io" -c user.name="verify-staging-client-go.sh" commit -q -m "Snapshot"
    popd >/dev/null
done

echo "Running godep restore"
pushd "${TEMP_STAGING_GOPATH}/src/k8s.io/kubernetes" >/dev/null
export GOPATH="${TEMP_STAGING_GOPATH}"
godep restore ${V} 2>&1 | sed 's/^/  /'

echo "Testing staging/copy.sh"
staging/copy.sh -d 2>&1 | sed 's/^/  /'
popd
