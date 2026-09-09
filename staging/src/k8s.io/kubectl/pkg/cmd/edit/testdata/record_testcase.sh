#!/usr/bin/env bash

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

set -o errexit
set -o nounset
set -o pipefail

if [[ -z "${1-}" ]]; then
    echo "Usage: record_testcase.sh testcase-name"
    exit 1
fi

# Clean up the test server
function cleanup {
    if [[ -n "${pid-}" ]]; then
        echo "Stopping recording server (${pid})"
        # kill the process `go run` launched
        pkill -P "${pid}"
        # kill the `go run` process itself
        kill -9 "${pid}"
    fi
}

testcase="${1}"

test_root="$(dirname "${BASH_SOURCE[0]}")"
testcase_dir="${test_root}/testcase-${testcase}"
mkdir -p "${testcase_dir}"

pushd "${testcase_dir}"
    export EDITOR="../record_editor.sh"
    go run "../record.go" &
    pid=$!
    trap cleanup EXIT
    echo "Started recording server (${pid})"

    # Make a kubeconfig that makes kubectl talk to our test server
    edit_kubeconfig="${TMP:-/tmp}/edit_test.kubeconfig"
    echo "apiVersion: v1
clusters:
- cluster:
    server: http://localhost:8081
  name: test
contexts:
- context:
    cluster: test
    user: test
  name: test
current-context: test
kind: Config
users: []
" > "${edit_kubeconfig}"
    export KUBECONFIG="${edit_kubeconfig}"

    echo "Starting subshell. Type exit when finished."
    bash
popd
