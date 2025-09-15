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

# This script builds some binaries and then the conformance image.
# REGISTRY and VERSION must be set.
# Example usage:
#   $ export REGISTRY=gcr.io/someone
#   $ export VERSION=v1.4.0-testfix
#   ./hack/dev-push-conformance.sh
# That will build and push gcr.io/someone/conformance-amd64:v1.4.0-testfix

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT="$(dirname "${BASH_SOURCE[0]}")/.."
source "${KUBE_ROOT}/build/common.sh"

if [[ -z "${REGISTRY:-}" ]]; then
	echo "REGISTRY must be set"
	exit 1
fi
if [[ -z "${VERSION:-}" ]]; then
	echo "VERSION must be set"
	exit 1
fi

IMAGE="${REGISTRY}/conformance-amd64:${VERSION}"

kube::build::verify_prereqs
kube::build::build_image
kube::build::run_build_command make WHAT="github.com/onsi/ginkgo/v2/ginkgo test/e2e/e2e.test cmd/kubectl test/conformance/image/go-runner"
kube::build::copy_output

make -C "${KUBE_ROOT}/test/conformance/image" build
docker push "${IMAGE}"
