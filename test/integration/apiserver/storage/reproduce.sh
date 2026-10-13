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

# Reproduces a historical storage correctness bug by reverting its fix, running
# the correctness suite, and restoring the tree afterwards.
#
# The suite is expected to FAIL while the patch is applied. This script inverts
# the exit code accordingly: it succeeds when the bug was detected and fails
# when it was not, so that losing detection power is itself a failure.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KUBE_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
export PATH="${KUBE_ROOT}/third_party/etcd:${PATH}"

REPRODUCTIONS_DIR="${SCRIPT_DIR}/reproductions"
WHAT="./test/integration/apiserver/storage"
COUNT="${COUNT:-1}"

log_status() {
  echo "+++ [$(date +'%m%d %H:%M:%S')] $*"
}

log_error() {
  echo "!!! [$(date +'%m%d %H:%M:%S')] $*" >&2
}

list_reproductions() {
  local patch
  for patch in "${REPRODUCTIONS_DIR}"/*.patch; do
    [[ -e "${patch}" ]] || continue
    echo "  $(basename "${patch}" .patch)"
  done
}

ISSUE="${ISSUE:-${1:-}}"
if [[ -z "${ISSUE}" ]]; then
  log_error "ISSUE is required, for example: ./test/integration/apiserver/storage/reproduce.sh 58545"
  log_error "Available reproductions:"
  list_reproductions >&2
  exit 1
fi

PATCH_FILE="${REPRODUCTIONS_DIR}/${ISSUE}.patch"
if [[ ! -f "${PATCH_FILE}" ]]; then
  log_error "No reproduction patch for issue ${ISSUE} at ${PATCH_FILE}"
  log_error "Available reproductions:"
  list_reproductions >&2
  exit 1
fi

if ! git -C "${KUBE_ROOT}" apply --check "${PATCH_FILE}" 2>/dev/null; then
  log_error "Patch ${PATCH_FILE} does not apply cleanly."
  log_error "The fix it reverts has likely moved. Refresh the patch and update the"
  log_error "track record in test/integration/apiserver/storage/README.md."
  exit 1
fi

PATCH_APPLIED=
OUTPUT_FILE=
cleanup() {
  if [[ -n "${PATCH_APPLIED}" ]]; then
    log_status "Reverting reproduction patch for issue ${ISSUE}"
    git -C "${KUBE_ROOT}" apply -R "${PATCH_FILE}"
    PATCH_APPLIED=
  fi
  if [[ -n "${OUTPUT_FILE}" && -f "${OUTPUT_FILE}" ]]; then
    rm -f "${OUTPUT_FILE}"
    OUTPUT_FILE=
  fi
}
trap cleanup EXIT

log_status "Applying reproduction patch for issue ${ISSUE}"
git -C "${KUBE_ROOT}" apply "${PATCH_FILE}"
PATCH_APPLIED=1

log_status "Running correctness suite (expected to fail)"
OUTPUT_FILE=$(mktemp)

rc=0
go -C "${KUBE_ROOT}" test -v -run "^TestCorrectness$" -count="${COUNT}" "${WHAT}" >"${OUTPUT_FILE}" 2>&1 || rc=$?
cat "${OUTPUT_FILE}"

reproduced=
if grep -q -- "--- FAIL: TestCorrectness" "${OUTPUT_FILE}" 2>/dev/null; then
  reproduced=1
fi

cleanup

# A non-zero exit code is not sufficient evidence: a missing etcd or a compile
# error also fails the build. Require an actual test failure from the suite.
if [[ -z "${reproduced}" ]]; then
  if [[ "${rc}" -ne 0 ]]; then
    log_error "Inconclusive: the build failed before TestCorrectness reported a result."
    log_error "Fix the environment (etcd on PATH, working build) and retry."
    exit 1
  fi
  log_error "NOT reproduced: the correctness suite passed with the fix for issue ${ISSUE} reverted."
  log_error "Detection of this bug class has been lost."
  exit 1
fi

log_status "Reproduced: issue ${ISSUE} was detected by the correctness suite"
exit 0
