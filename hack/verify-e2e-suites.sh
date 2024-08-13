#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

# This script checks that all E2E test suites are sane, i.e. can be
# started without an error.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::golang::setup_env

cd "${KUBE_ROOT}"

kube::util::ensure-temp-dir

res=0
for suite in $(git grep -l framework.AfterReadingAllFlags | grep -v -e ^test/e2e/framework -e ^hack | xargs -n 1 dirname | sort -u); do
    # Build a binary and run it in the root directory to get paths that are
    # relative to that instead of the package directory.
    out=""
    if (cd "$suite" && go test -c -o "${KUBE_TEMP}/e2e.bin" .) && out=$("${KUBE_TEMP}/e2e.bin" --list-tests); then
        echo "E2E suite $suite passed."
    else
        echo >&2 "ERROR: E2E test suite invocation failed for $suite."
        # shellcheck disable=SC2001
        echo "$out" | sed -e 's/^/   /'
        res=1
    fi
done
exit "$res"
