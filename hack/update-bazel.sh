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
set -o nounset
set -o pipefail

export KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Remove generated files prior to running kazel.
# TODO(spxtr): Remove this line once Bazel is the only way to build.
rm -f "${KUBE_ROOT}/pkg/generated/openapi/zz_generated.openapi.go"

# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BINPATH}"
PATH="${GOBIN}:${PATH}"

# Install tools we need, but only from vendor/...
go install k8s.io/kubernetes/vendor/github.com/bazelbuild/bazel-gazelle/cmd/gazelle
go install k8s.io/kubernetes/vendor/github.com/kubernetes/repo-infra/kazel

touch "${KUBE_ROOT}/vendor/BUILD"
# Ensure that we use the correct importmap for all vendored dependencies.
# Probably not necessary in gazelle 0.13+
# (https://github.com/bazelbuild/bazel-gazelle/pull/207).
if ! grep -q "# gazelle:importmap_prefix" "${KUBE_ROOT}/vendor/BUILD"; then
  echo "# gazelle:importmap_prefix k8s.io/kubernetes/vendor" >> "${KUBE_ROOT}/vendor/BUILD"
fi

gazelle fix \
    -external=vendored \
    -mode=fix \
    -repo_root "${KUBE_ROOT}" \
    "${KUBE_ROOT}"

kazel
