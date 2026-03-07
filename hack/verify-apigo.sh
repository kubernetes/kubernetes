#!/usr/bin/env bash

# Copyright The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
cd "${KUBE_ROOT}"

kube::golang::setup_env

OUT_DIR="_output"
APIGO_CMD="./hack/tools/apigo/main.go"
APIGO_BIN="${OUT_DIR}/apigo"

mkdir -p "${OUT_DIR}"

echo "Building apigo into ${APIGO_BIN}..."
go build -o "${APIGO_BIN}" "${APIGO_CMD}"

# Target only modules in staging/ that have explicitly opted in
# by creating an "apigo" directory at their root.
MODULE_DIRS=()
for dir in "staging/src/k8s.io"/*; do
    if [ -d "${dir}/apigo" ]; then
        MODULE_DIRS+=("${dir}")
    fi
done

if [ ${#MODULE_DIRS[@]} -eq 0 ]; then
    echo "No modules with an 'apigo' tracking directory found. Skipping."
    exit 0
fi

FAILED=0

echo "Verifying cumulative API baselines for targeted modules..."

for module_dir in "${MODULE_DIRS[@]}"; do
    abs_module_dir="${KUBE_ROOT}/${module_dir}"
    
    echo "  -> Verifying ${module_dir}..."
    if ! "${APIGO_BIN}" -verify "${abs_module_dir}"; then
        FAILED=1
    fi
done

echo "--------------------------------------------------------"
if [ $FAILED -ne 0 ]; then
    echo "ERROR: Unauthorized API breakages or unregistered additions detected!"
    echo "Please read hack/tools/apigo/README.md for instructions on how to handle"
    echo "new additions (next.txt) or approved breaking changes (except.txt)."
    exit 1
else
    echo "SUCCESS: All targeted module APIs satisfy their historical contracts."
    exit 0
fi
