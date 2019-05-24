#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors.
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

kube::golang::setup_env

kube::util::ensure-temp-dir
OUTPUT="${KUBE_TEMP}"/symbols-output
cleanup() {
	rm -rf "${OUTPUT}"
}
trap "cleanup" EXIT SIGINT
mkdir -p "${OUTPUT}"

GOLDFLAGS="-w" make -C "${KUBE_ROOT}" WHAT=cmd/hyperkube

# Add other BADSYMBOLS here.
BADSYMBOLS=(
  "httptest"
  "testify"
  "testing[.]"
  "TestOnlySetFatalOnDecodeError"
  "TrackStorageCleanup"
)

# b/c hyperkube binds everything simply check that for bad symbols
go tool nm "${KUBE_OUTPUT_HOSTBIN}/hyperkube" > "${OUTPUT}/hyperkube-symbols"

if ! grep -q "NewHyperKubeCommand" "${OUTPUT}/hyperkube-symbols"; then
  echo "No symbols found in hyperkube binary."
  exit 1
fi

RESULT=0
for BADSYMBOL in "${BADSYMBOLS[@]}"; do
  if FOUND=$(grep "${BADSYMBOL}" < "${OUTPUT}/hyperkube-symbols"); then
    echo "Found bad symbol '${BADSYMBOL}':"
    echo "$FOUND"
    RESULT=1
  fi
done

exit $RESULT

# ex: ts=2 sw=2 et filetype=sh
