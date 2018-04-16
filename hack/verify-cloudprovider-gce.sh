#!/usr/bin/env bash

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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
GENERATOR="${KUBE_ROOT}/pkg/cloudprovider/providers/gce/cloud/gen/main.go"

GEN_GO="${KUBE_ROOT}/pkg/cloudprovider/providers/gce/cloud/gen.go"
GEN_TEST_GO="${KUBE_ROOT}/pkg/cloudprovider/providers/gce/cloud/gen_test.go"

kube::golang::setup_env

TMPFILE=$(mktemp verify-cloudprovider-gce-XXXX)
trap "{ rm -f ${TMPFILE}; }" EXIT

go run "${GENERATOR}" > ${TMPFILE}
if ! diff "${TMPFILE}" "${GEN_GO}"; then
  echo "Generated file ${GEN_GO} needs to be updated (run hack/update-cloudprovider-gce.sh)"
  echo
  diff -u "${TMPFILE}" "${GEN_GO}" || true
  exit 1
fi

go run "${GENERATOR}" -mode test > ${TMPFILE}
if ! diff "${TMPFILE}" "${GEN_TEST_GO}"; then
  echo "Generated file ${GEN_TEST_GO} needs to be updated (run hack/update-cloudprovider-gce.sh)"
  echo
  diff -u "${TMPFILE}" "${GEN_TEST_GO}" || true
  exit 1
fi

exit 0
