#!/bin/bash

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

# This script runs an individual test using go test -exec ./hack/run-test.sh.
# This allows us to run loops of the test binary with some parameters tweaked,
# namely etcd path and api version. This is really only neccessary because go
# test compilation caching is terrible (probably because of all the weirdness
# we do in our builds) and this allows us to compile each test binary once.

set -o errexit
set -o nounset
set -o pipefail

test_binary="${1}"
shift

# Convert the CSVs to arrays.
IFS=';' read -a apiVersions <<< "${KUBE_TEST_API_VERSIONS}"
IFS=',' read -a etcdPrefixes <<< "${KUBE_TEST_ETCD_PREFIXES}"
apiVersionsCount=${#apiVersions[@]}
etcdPrefixesCount=${#etcdPrefixes[@]}
for (( i=0, j=0; ; )); do
  apiVersion=${apiVersions[i]}
  etcdPrefix=${etcdPrefixes[j]}
  # echo "Running tests for APIVersion: $apiVersion with etcdPrefix: $etcdPrefix"
  # KUBE_TEST_API sets the version of each group to be tested.
  KUBE_TEST_API="${apiVersion}" ETCD_PREFIX="${etcdPrefix}" "${test_binary}" "$@"
  i=${i}+1
  j=${j}+1
  if [[ i -eq ${apiVersionsCount} ]] && [[ j -eq ${etcdPrefixesCount} ]]; then
    # All api versions and etcd prefixes tested.
    break
  fi
  if [[ i -eq ${apiVersionsCount} ]]; then
    # Use the last api version for remaining etcd prefixes.
    i=${i}-1
  fi
   if [[ j -eq ${etcdPrefixesCount} ]]; then
     # Use the last etcd prefix for remaining api versions.
    j=${j}-1
  fi
done
