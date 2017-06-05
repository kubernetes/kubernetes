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
set -o nounset
set -o pipefail

cd $(dirname "${BASH_SOURCE}")

GOPRJ="k8s.io/gengo/examples/deepcopy-gen"

go build .

trap "echo FAIL" EXIT

TESTDIR="$(mktemp -d /tmp/deepcopy.$$.XXXXXX)"
echo "Test logs are in ${TESTDIR}"

function PASS() {
    echo "PASS"
}

function TEST() {
    case=$1
    shift
    echo -n "Testing ${case}: "
    ./deepcopy-gen --v=10 --logtostderr "$@" > "${TESTDIR}/${case}.log" 2>&1
}

function DIFF() {
    d=$(git diff HEAD "$1" || true)
    if [ -n "${d}" ]; then
        echo "$1 changed"
        echo
        echo "${d}"
        return 1
    fi
}

TEST wholepkg -i "${GOPRJ}/test/wholepkg"
DIFF ./test/wholepkg/deepcopy_generated.go
PASS
go test -v ./test/wholepkg

trap - EXIT
