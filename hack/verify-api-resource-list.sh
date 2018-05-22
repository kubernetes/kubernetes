#!/bin/bash

# Copyright 2018 The Kubernetes Authors.
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

# Script to verify if we have the correct list of latest api resources
# served by kube apiserver, under
# test/e2e/testing-manifests/apiresource/resources_all.csv

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env
kube::etcd::install
export PATH="${KUBE_ROOT}/third_party/etcd:${PATH}"

make -C "${KUBE_ROOT}" WHAT=cmd/kube-apiserver

apiserver=$(kube::util::find-binary "kube-apiserver")

LISTROOT="${KUBE_ROOT}/test/e2e/testing-manifests/apiresource"
TMP_LISTROOT="${KUBE_ROOT}/_tmp/apiresource"
_tmp="${KUBE_ROOT}/_tmp"

mkdir -p "${_tmp}"
cp -a "${LISTROOT}" "${TMP_LISTROOT}"
kube::util::trap_add "cp -a ${TMP_LISTROOT} ${LISTROOT}/..; rm -rf ${_tmp}" EXIT SIGINT
rm ${LISTROOT}/resources_all.csv ${LISTROOT}/resources.csv

"${KUBE_ROOT}/hack/update-api-resource-list.sh"

# For the propose of easily comparing KUBE_RESOURCE_FILE and
# KUBE_RESOURCE_WHITELIST_FILE in order to remove whitelisted API resource
# lines, we need to sort the files.
#
# The API coverage e2e test requires reading parent resource before reading
# subresource, to correctly construct the resource map. For example we want
#   ,v1,pods,true,VERB
# to be ordered before
#   ,v1,pods/status,true,VERB
sort -t',' -k1,1 -k2,2 -k3,3 -k4 "${LISTROOT}/resources_whitelist.csv" -o "${LISTROOT}/resources_whitelist.csv"
# Remove whitelisted API resource lines from KUBE_RESOURCE_FILE, to generated
# KUBE_RESOURCE_TEST_FILE.
#
# NOTE: GNU comm and BSD comm have different defaulting. We sort the inputs to satisfy
# the defaulting first, then sort the output to more readable csv format
comm -13 <(sort "${LISTROOT}/resources_whitelist.csv") <(sort "${LISTROOT}/resources_all.csv") | sort -t',' -k1,1 -k2,2 -k3,3 -k4 -o "${LISTROOT}/resources.csv"

function assert_equal() {
  echo "Diffing $1 against freshly generated api resource list"
  ret=0
  diff -Naupr $1 $2 || ret=$?
  if [[ $ret -eq 0 ]]
  then
    echo "$1 up to date."
  else
    echo "ERROR: $1 is out of date. Please run hack/update-api-resource-list.sh"
    echo "WARNING: If you are making API change that adds/updates API GROUP/VERSION/KIND, please update test/e2e/testing-manifests/apiresource/yamlfiles/GROUP/VERSION/KIND.yaml to properly pass API coverage e2e test: test/e2e/apimachinery/coverage.go"
    exit 1
  fi
}

assert_equal "${LISTROOT}/resources_all.csv" "${TMP_LISTROOT}/resources_all.csv"
assert_equal "${LISTROOT}/resources_whitelist.csv" "${TMP_LISTROOT}/resources_whitelist.csv"
assert_equal "${LISTROOT}/resources.csv" "${TMP_LISTROOT}/resources.csv"

kube::log::status "SUCCESS"

# ex: ts=2 sw=2 et filetype=sh
