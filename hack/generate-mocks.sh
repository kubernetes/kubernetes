#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

[ -n "${DEBUG:-}" ] && set -x

#- Usage & Arguments:
#-   $ ./hack/generate-mocks.sh <source-path> <output-path> <interface> [<fake-name>]
#-
#-   source-path
#-     Path to the file or directory containing the interface to fake.
#-
#-   output-path
#-     Path to the file for the generated fakes. This also determines the
#-     package name that will be used.
#-
#-   interface
#-     source-path is specified: Name of the interface to fake.
#-
#-   fake-name
#-     Name of the fake struct to generate. By default, 'Fake' will
#-     be prepended to the name of the original interface.
#-
#- This helper uses counterfeiter[1] to automatically generate a fake based on
#- an interface. It is a wrapper to be used with 'go generate' so users can
#- update all the fakes in one go by calling:
#-   $ go generate -run 'mock' pkg/...
#-
#- In addition to counterfeiter, this wrapper also handles the boilerplate
#- header generated go files need to be prepended with.
#-
#- Example:
#-   ,----
#-   | //go:generate $KUBE_ROOT/hack/generate-mock.sh ../path/to/the/interface.go ../path/to/the/fakes/the_fake_interface.go TheInferface TheFakeInterface
#-   `----
#-
#-     This generates a fake for the interface 'TheInterface' which is defined in
#-     the source file '../path/to/the/interface.go'. It places the new fake into its
#-     own package '../path/to/the/fakes' and names the fake 'TheFakeInterface'
#-
#- [1] https://github.com/maxbrunsfeld/counterfeiter


# shellcheck disable=SC2155,SC2128
export KUBE_ROOT="$(dirname "${BASH_SOURCE}")/.."

readonly CF_BIN='counterfeiter'
readonly BP_FILE="${KUBE_ROOT}/hack/boilerplate/boilerplate.generatego.txt"

mocks::createMock() {
    local tmpFile="$1"
    local srcFile="$2"
    local interfaceName="$3"
    local destFile="$4"
    local fakeName="${5:-}"
    local cmdOpts=()

    cmdOpts+=( -o "$destFile" )
    [ -n "$fakeName" ] && {
        cmdOpts+=( -fake-name "$fakeName" )
    }

    mkdir -p "$( dirname "$destFile" )"
    mocks::runCounterfeiter "${cmdOpts[@]}" "$srcFile" "$interfaceName" '-' > "$tmpFile"

    # only generate output if everything else worked
    cat "$tmpFile" > "$destFile"
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
    # When creating a mock for an interface in a vendored package counterfeiter
    # genertates the wrong import path -- we just clean that up here.
    # TODO(hoegaarden) check if counterfeiter v6 fixes that
    sed 's@k8s.io/kubernetes/vendor/@@g'
}

mocks::validateCounterfeiter() {
    command -v "$CF_BIN" >/dev/null 2>&1 && return

    {
        echo "$CF_BIN not installed, install it with:"
        echo "  go get -u github.com/maxbrunsfeld/counterfeiter"
    } >&2

    return 1
}

mocks::checkArgs() {
    # Needs to be called with 3 or 4 args:
    #   - source code file
    #   - source interface name
    #   - destination file for the generated fake
    #   - optional: the name of the fake
    [ "$#" -ge 3 ] && [ "$#" -le 4 ]
}

mocks::usage() {
    local usageMarker='^#- ?'
    # shellcheck disable=SC2016
    local awkProg='$0 ~ RE { gsub(RE, ""); print }'

    awk -vRE="$usageMarker" "$awkProg" <"$0" >&2
    return 1
}

mocks::getCurrentPkg() {
    go list .
}

mocks::main() {
    mocks::checkArgs "$@" || mocks::usage

    mocks::validateCounterfeiter

    local tmpFile
    tmpFile="$( mktemp )"
    # shellcheck disable=SC2064
    trap "rm -f -- '$tmpFile'" EXIT

    echo -n "$(mocks::getCurrentPkg): " >&2
    mocks::createMock "$tmpFile" "$@"
}

mocks::main "$@"
