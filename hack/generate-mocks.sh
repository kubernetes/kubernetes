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

readonly CF_BIN='counterfeiter'
readonly BP_FILE="${KUBE_ROOT}/hack/boilerplate/boilerplate.generatego.txt"

mocks::createMock() {
    local srcFile="$1"
    local interfaceName="$2"
    local destFile="$3"
    local fakeName="${4:-}"
    local cmdOpts=()

    cmdOpts+=( -o "$destFile" )
    [ -n "$fakeName" ] && {
        cmdOpts+=( -fake-name "$fakeName" )
    }

    mocks::runCounterfeiter "${cmdOpts[@]}" "$srcFile" "$interfaceName" '-' \
        > "$destFile"
}

mocks::runCounterfeiter() {
    # print the boilerplate file
    cat "$BP_FILE"
    # generate the fake with counterfeiter
    "$CF_BIN" "$@" \
        | mocks::removeVendor \
        | mocks::gofmt
}

mocks::gofmt() {
    gofmt -s
}

mocks::removeVendor() {
    gsed 's@k8s.io/kubernetes/vendor/@@g'
}

mocks::validateCounterfeiter() {
    command -v "$CF_BIN" >/dev/null 2>&1 && return

    {
        echo "$CF_BIN not installed, install it with:"
        echo "  go get -u github.com/maxbrunsfeld/counterfeiter"
    } >&2
    return 1
}

mocks::validateEnv() {
    if [ -z "$KUBE_ROOT" ]
    then
        # shellcheck disable=SC2016
        echo '$KUBE_ROOT needs to be set' >&2
        return 2
    fi
}

mocks::main() {
    mocks::validateEnv
    mocks::validateCounterfeiter
    mocks::createMock "$@"
}

mocks::main "$@"
