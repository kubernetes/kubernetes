#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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
set -o pipefail
set -o nounset

if [[ -z "${KUBE_ROOT:-}" ]]; then
	# Relative to test/e2e/generated/
	KUBE_ROOT="../../../"
fi

source "${KUBE_ROOT}/cluster/lib/logging.sh"

if [[ ! -d "${KUBE_ROOT}/examples" ]]; then
	echo "${KUBE_ROOT}/examples not detected.  This script should be run from a location where the source dirs are available."
	exit 1
fi

if ! which go-bindata &>/dev/null ; then
	echo "Cannot find go-bindata. Install with"
	echo "  go get -u github.com/jteeuwen/go-bindata/go-bindata"
	exit 5
fi

BINDATA_OUTPUT="${KUBE_ROOT}/test/e2e/generated/bindata.go"
go-bindata -nometadata -prefix "${KUBE_ROOT}" -o ${BINDATA_OUTPUT} -pkg generated \
	-ignore .jpg -ignore .png -ignore .md \
	"${KUBE_ROOT}/examples/..." \
	"${KUBE_ROOT}/docs/user-guide/..." \
	"${KUBE_ROOT}/test/e2e/testing-manifests/..." \
	"${KUBE_ROOT}/test/images/..."

gofmt -s -w ${BINDATA_OUTPUT}

V=2 kube::log::info "Generated bindata file : $(wc -l ${BINDATA_OUTPUT}) lines of lovely automated artifacts"
