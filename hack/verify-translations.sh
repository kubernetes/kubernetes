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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

cd "${KUBE_ROOT}"

kube::util::ensure-temp-dir
kube::util::ensure_clean_working_dir

BINDATA_OUTPUT="pkg/generated/bindata.go"
TEST_BINDATA_OUTPUT="test/e2e/generated/bindata.go"

cp "${BINDATA_OUTPUT}" "${KUBE_TEMP}/bindata.go"
cp "${TEST_BINDATA_OUTPUT}" "${KUBE_TEMP}/test_bindata.go"

hack/update-translations.sh

ret=0
diff -Naup -I 'Generated bindata' "${BINDATA_OUTPUT}" "${KUBE_TEMP}/bindata.go" || ret=$?
diff -Naup -I 'Generated test bindata' "${TEST_BINDATA_OUTPUT}" "${KUBE_TEMP}/test_bindata.go" || ret=$?

if [[ $ret -eq 0 ]]
then
  echo "bindata.go is up to date."
else
  echo "bindata.go is out of date. Please run hack/update-translations.sh" >&2
  exit 1
fi
