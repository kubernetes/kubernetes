#!/usr/bin/env bash

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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
export KUBE_ROOT
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/logging.sh"

if [[ ! -d "${KUBE_ROOT}/pkg" ]]; then
	echo "${KUBE_ROOT}/pkg not detected.  This script should be run from a location where the source dirs are available."
	exit 1
fi

# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BINPATH}"
PATH="${GOBIN}:${PATH}"

# Install tools we need, but only from vendor/...
if [[ ! "${SKIP_INSTALL_GO_BINDATA:-}" ]]; then
  # Only install if being called directly to allow being called from origin in a vendored path.
  GO111MODULE=off go install ./vendor/github.com/go-bindata/go-bindata/...
fi

# run the generation from the root directory for stable output
pushd "${KUBE_ROOT}" >/dev/null

# These are files for e2e tests.
BINDATA_OUTPUT="test/e2e/generated/bindata.go"
# IMPORTANT: if you make any changes to these arguments, you must also update
# test/e2e/generated/BUILD and/or build/bindata.bzl.
go-bindata -nometadata -o "${BINDATA_OUTPUT}.tmp" -pkg generated \
	-ignore .jpg -ignore .png -ignore .md -ignore 'BUILD(\.bazel)?' \
	"test/conformance/testdata/..." \
	"test/e2e/testing-manifests/..." \
	"test/e2e_node/testing-manifests/..." \
	"test/images/..." \
	"test/fixtures/..."

gofmt -s -w "${BINDATA_OUTPUT}.tmp"

# Here we compare and overwrite only if different to avoid updating the
# timestamp and triggering a rebuild. The 'cat' redirect trick to preserve file
# permissions of the target file.
if ! cmp -s "${BINDATA_OUTPUT}.tmp" "${BINDATA_OUTPUT}" ; then
	cat "${BINDATA_OUTPUT}.tmp" > "${BINDATA_OUTPUT}"
	V=2 kube::log::info "Generated bindata file : ${BINDATA_OUTPUT} has $(wc -l ${BINDATA_OUTPUT}) lines of lovely automated artifacts"
else
	V=2 kube::log::info "No changes in generated bindata file: ${BINDATA_OUTPUT}"
fi

rm -f "${BINDATA_OUTPUT}.tmp"

# These are files for runtime code
BINDATA_OUTPUT="staging/src/k8s.io/kubectl/pkg/generated/bindata.go"
# IMPORTANT: if you make any changes to these arguments, you must also update
# pkg/generated/BUILD and/or build/bindata.bzl.
go-bindata -nometadata -nocompress -o "${BINDATA_OUTPUT}.tmp" -pkg generated \
	-ignore .jpg -ignore .png -ignore .md -ignore 'BUILD(\.bazel)?' \
	"translations/..."

gofmt -s -w "${BINDATA_OUTPUT}.tmp"

# Here we compare and overwrite only if different to avoid updating the
# timestamp and triggering a rebuild. The 'cat' redirect trick to preserve file
# permissions of the target file.
if ! cmp -s "${BINDATA_OUTPUT}.tmp" "${BINDATA_OUTPUT}" ; then
	cat "${BINDATA_OUTPUT}.tmp" > "${BINDATA_OUTPUT}"
	V=2 kube::log::info "Generated bindata file : ${BINDATA_OUTPUT} has $(wc -l ${BINDATA_OUTPUT}) lines of lovely automated artifacts"
else
	V=2 kube::log::info "No changes in generated bindata file: ${BINDATA_OUTPUT}"
fi

rm -f "${BINDATA_OUTPUT}.tmp"

popd >/dev/null
