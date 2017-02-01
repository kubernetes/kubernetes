#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

if [[ -z "${1-}" ]]; then
    echo "Usage: record_testcase.sh testcase-name"
    exit 1
fi

# Clean up the test server
function cleanup {
    if [[ ! -z "${pid-}" ]]; then
        echo "Stopping recording server (${pid})"
        # kill the process `go run` launched
        pkill -P "${pid}"
        # kill the `go run` process itself
        kill -9 "${pid}"
    fi
}

testcase="${1}"

test_root="$(dirname "${BASH_SOURCE}")"
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
