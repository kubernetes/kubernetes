#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

# This script checks coding style for go language files in each
# Kubernetes package by golint.
# Usage: `hack/verify-golangci-lint.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::golang::verify_go_version

# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BINPATH}"
PATH="${GOBIN}:${PATH}"

# Explicitly opt into go modules, even though we're inside a GOPATH directory
export GO111MODULE=on

# Install golangci-lint
echo "installing golangci-lint and logcheck plugin from hack/tools into ${GOBIN}"
pushd "${KUBE_ROOT}/hack/tools" >/dev/null
  go install github.com/golangci/golangci-lint/cmd/golangci-lint
  go build -o "${GOBIN}/logcheck.so" -buildmode=plugin sigs.k8s.io/logtools/logcheck/plugin
popd >/dev/null

cd "${KUBE_ROOT}"

# The config is in ${KUBE_ROOT}/.golangci.yaml where it will be found
# even when golangci-lint is invoked in a sub-directory.
#
# The logcheck plugin currently has to be configured via env variables
# (https://github.com/golangci/golangci-lint/issues/1512).
#
# Remember to clean the golangci-lint cache when changing
# the configuration and running this script multiple times,
# otherwise golangci-lint will report stale results:
# _output/local/bin/golangci-lint cache clean
export LOGCHECK_CONFIG="${KUBE_ROOT}/hack/logcheck.conf"
echo 'running golangci-lint ' >&2
res=0
if [[ "$#" -gt 0 ]]; then
    golangci-lint run "$@" >&2 || res=$?
else
    golangci-lint run ./... >&2 || res=$?
    for d in staging/src/k8s.io/*; do
        MODPATH="staging/src/k8s.io/$(basename "${d}")"
        echo "running golangci-lint for ${KUBE_ROOT}/${MODPATH}"
        pushd "${KUBE_ROOT}/${MODPATH}" >/dev/null
            golangci-lint --path-prefix "${MODPATH}" run ./... >&2 || res=$?
        popd >/dev/null
    done
fi

# print a message based on the result
if [ "$res" -eq 0 ]; then
  echo 'Congratulations! All files are passing lint :-)'
else
  {
    echo
    echo 'Please review the above warnings. You can test via "./hack/verify-golangci-lint.sh"'
    echo 'If the above warnings do not make sense, you can exempt this warning with a comment'
    echo ' (if your reviewer is okay with it).'
    echo 'In general please prefer to fix the error, we have already disabled specific lints'
    echo ' that the project chooses to ignore.'
    echo 'See: https://golangci-lint.run/usage/false-positives/'
    echo
  } >&2
  exit 1
fi

# preserve the result
exit "$res"
