#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
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

# TODO this does not address client-go, since it takes a different approach to vendoring
# TODO client-go should probably be made consistent

set -o errexit
set -o nounset
set -o pipefail


KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"


TARGET_DIR=$(mktemp -d "${TMPDIR:-/tmp/}$(basename 0).XXXXXXXXXXXX")
# Register function to be called on EXIT to remove folder.
function cleanup {
	SKIP_CLEANUP=${SKIP_CLEANUP:-}
	if [ "${SKIP_CLEANUP}" != "true" ]; then
		rm -rf "${TARGET_DIR}"
	fi
}
trap cleanup EXIT

TARGET_DIR=${TARGET_DIR} ${KUBE_ROOT}/hack/update-staging-godeps.sh

# check each staging repo to make sure its Godeps.json is correct
for stagingRepo in $(ls ${KUBE_ROOT}/staging/src/k8s.io); do
	# we have to skip client-go because it does unusual manipulation of its godeps
	if [ "${stagingRepo}" == "client-go" ]; then
		continue
	fi

	diff --ignore-matching-lines='^\s*\"Comment\"' ${KUBE_ROOT}/staging/src/k8s.io/${stagingRepo}/Godeps/Godeps.json ${TARGET_DIR}/src/k8s.io/${stagingRepo}/Godeps/Godeps.json
done
