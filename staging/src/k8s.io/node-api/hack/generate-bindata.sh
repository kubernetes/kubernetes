#!/usr/bin/env bash

# Copyright 2019 The Kubernetes Authors.
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

export KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

if [[ ! -d "${KUBE_ROOT}/pkg" ]]; then
	echo "${KUBE_ROOT}/pkg not detected.  This script should be run from a location where the source dirs are available."
	exit 1
fi

if ! which go-bindata &>/dev/null ; then
	echo "Cannot find go-bindata."
	exit 5
fi

# run the generation from the root directory for stable output
pushd "${KUBE_ROOT}" >/dev/null

# These are files for the node-api CRDs
BINDATA_OUTPUT="pkg/apis/node/crd/bindata.go"
go-bindata -nometadata -o "${BINDATA_OUTPUT}.tmp" -pkg crd \
	-ignore .jpg -ignore .png -ignore .md -ignore 'BUILD(\.bazel)?' \
  "pkg/apis/node/crd/runtimeclass_crd.yaml"

gofmt -s -w "${BINDATA_OUTPUT}.tmp"

# Here we compare and overwrite only if different to avoid updating the
# timestamp and triggering a rebuild. The 'cat' redirect trick to preserve file
# permissions of the target file.
if ! cmp -s "${BINDATA_OUTPUT}.tmp" "${BINDATA_OUTPUT}" ; then
	cat "${BINDATA_OUTPUT}.tmp" > "${BINDATA_OUTPUT}"
fi

rm -f "${BINDATA_OUTPUT}.tmp"

popd >/dev/null
