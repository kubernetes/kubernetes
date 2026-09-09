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

# This script verifies that staging modules pass tests with their advertised
# Go version (from go.mod). This script requires that the host has go 1.21+
#
# Usage: `staging/test/test-modules-with-minimum-go.sh`.

set -o errexit -o nounset -o pipefail
KUBE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

# where we store test results
ARTIFACTS="${ARTIFACTS:-"${KUBE_ROOT}/_artifacts"}"
# Set to 'false' to disable reduction of the JUnit file to only the top level tests.
KUBE_PRUNE_JUNIT_TESTS=${KUBE_PRUNE_JUNIT_TESTS:-true}

staging_modules_root="${KUBE_ROOT}/staging/src/k8s.io"

# NOTE: This script intentionally does not do kube::golang::setup_env as a normal
# Kubernetes test / build script would do, because we are testing how other
# project would consume the staging modules, NOT how we would build them
# in a core Kubernetes binary.
# We will test with Go / module defaults.
#
# BE VERY CAREFUL SOURCING 'library' SCRIPTS EXCEPT IN SUBSHELLS!!
# We do not want to pollute the shell with any go build options.

# setup etcd for integration tests
. "${KUBE_ROOT}"/hack/lib/etcd.sh
cleanup_etcd(){
    kube::etcd::cleanup
}
. "${KUBE_ROOT}"/hack/install-etcd.sh
kube::etcd::start
trap cleanup_etcd EXIT

# setup gotestsum, in a subshell to avoid manipulating go env
(
    . "${KUBE_ROOT}"/hack/lib/init.sh
    kube::golang::setup_env
    GOTOOLCHAIN="$(kube::golang::hack_tools_gotoolchain)" go -C "${KUBE_ROOT}/hack/tools" install gotest.tools/gotestsum
    go -C "${KUBE_ROOT}/cmd/prune-junit-xml" install .
)
# shellcheck disable=SC2031 # we are intentionally isolating subshells
export PATH="${PATH}:${KUBE_ROOT}/_output/local/go/bin"

# NOTE: the choice of 1.21.0+auto is very deliberate
#
# This is the earliest version with GOTOOLCHAIN itself, but auto allows for
# upgrading to a newer version based on go.mod.
#
# So we will use the version in go.mod, and NOT the version the host had
# installed and NOT the version in .go-version (which we already test 
# elsewhere with 'make test').
# shellcheck disable=SC2031 # we are intentionally isolating subshells
export GOTOOLCHAIN=go1.21.0+auto

res=0
for module_dir in "${staging_modules_root}"/*; do
    module="${module_dir#"${staging_modules_root}"/}"
    module_import="k8s.io/${module}"
    junit_file="${ARTIFACTS}/junit_${module/\//_}.xml"
    (
        cd "$module_dir"
        echo >&2 "Testing ${module_import} with: $(go version)"
        gotestsum --junitfile="${junit_file}" ./...
    ) || res=$?
    prune-junit-xml -prune-tests="${KUBE_PRUNE_JUNIT_TESTS}" "${junit_file}"
done

if [[ "${res}" -eq 0 ]]; then
    echo >&2 "All tests pass!"
else
    echo >&2 "Tests failed, exiting ${res}"
fi
exit $res
