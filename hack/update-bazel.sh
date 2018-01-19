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

kube::util::ensure-gnu-sed

# Remove generated files prior to running kazel.
# TODO(spxtr): Remove this line once Bazel is the only way to build.
rm -f "${KUBE_ROOT}/pkg/generated/openapi/zz_generated.openapi.go"

# The git commit sha1s here should match the values in $KUBE_ROOT/WORKSPACE.
kube::util::go_install_from_commit \
    github.com/kubernetes/repo-infra/kazel \
    ae4e9a3906ace4ba657b7a09242610c6266e832c
kube::util::go_install_from_commit \
    github.com/bazelbuild/bazel-gazelle/cmd/gazelle \
    31ce76e3acc34a22434d1a783bb9b3cae790d108  # 0.8.0

touch "${KUBE_ROOT}/vendor/BUILD"

gazelle fix \
    -build_file_name=BUILD,BUILD.bazel \
    -external=vendored \
    -proto=legacy \
    -mode=fix
# gazelle gets confused by our staging/ directory, prepending an extra
# "k8s.io/kubernetes/staging/src" to the import path.
# gazelle won't follow the symlinks in vendor/, so we can't just exclude
# staging/. Instead we just fix the bad paths with sed.
find staging -name BUILD -o -name BUILD.bazel | \
  xargs ${SED} -i 's|\(importpath = "\)k8s.io/kubernetes/staging/src/\(.*\)|\1\2|'

kazel
