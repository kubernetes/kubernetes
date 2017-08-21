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

# The git commit sha1s here should match the values in $KUBE_ROOT/WORKSPACE.
kube::util::go_install_from_commit github.com/kubernetes/repo-infra/kazel 9ba3a24eeffafa3a794b9e7312103424e7da26e9
kube::util::go_install_from_commit github.com/bazelbuild/rules_go/go/tools/gazelle/gazelle 82483596ec203eb9c1849937636f4cbed83733eb

gazelle_diff=$(gazelle fix -build_file_name=BUILD,BUILD.bazel -external=vendored -mode=diff -repo_root="$(kube::realpath ${KUBE_ROOT})")
kazel_diff=$(kazel -dry-run -print-diff -root="$(kube::realpath ${KUBE_ROOT})")

if [[ -n "${gazelle_diff}" || -n "${kazel_diff}" ]]; then
  echo "${gazelle_diff}"
  echo "${kazel_diff}"
  echo
  echo "Run ./hack/update-bazel.sh"
  exit 1
fi
