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

# Verifies that api reference docs are up to date.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

API_REFERENCE_DOCS_ROOT="${KUBE_ROOT}/docs/api-reference"
OUTPUT_DIR="${KUBE_ROOT}/_tmp/api-reference"
mkdir -p ${OUTPUT_DIR}
TMP_ROOT="${KUBE_ROOT}/_tmp"
trap "rm -rf ${TMP_ROOT}" EXIT SIGINT

# Generate API reference docs in tmp.
"./hack/update-api-reference-docs.sh" "${OUTPUT_DIR}"

echo "diffing ${API_REFERENCE_DOCS_ROOT} against freshly generated docs"
ret=0
diff -NauprB -I 'Last update' --exclude=*.md "${API_REFERENCE_DOCS_ROOT}" "${OUTPUT_DIR}" || ret=$?
if [[ $ret -eq 0 ]]
then
  echo "${API_REFERENCE_DOCS_ROOT} up to date."
else
  echo "${API_REFERENCE_DOCS_ROOT} is out of date. Please run hack/update-api-reference-docs.sh"
  exit 1
fi

# ex: ts=2 sw=2 et filetype=sh
