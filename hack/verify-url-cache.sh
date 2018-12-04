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
set -o nounset
set -o pipefail

export KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

cd "${KUBE_ROOT}"
urlcache=$(mktemp -d)
trap "rm -rf $urlcache" EXIT
# Silence the pre-main logging in the E2E test suite binary by providing a fake
# config for a local provider. Instead of running tests, we just tell the test
# suite to download all registered URLs.
KUBECONFIG=/dev/null go test -v ./test/e2e -args --provider=local -testfiles.url.cache-dir="$urlcache" -testfiles.url.cache-refresh

echo
if diff -r -x README.md "${KUBE_ROOT}/test/e2e/testing-manifests/url-cache" "$urlcache"; then
    echo "test/e2e/testing-manifests/url-cache is up-to-date."
else
    echo
    echo "ERROR: test/e2e/testing-manifests/url-cache is not an exact copy of the original files."
    echo
    echo "Run hack/update-url-cache.sh to synchronize it and/or update the original files"
    echo "and the URLs referencing them."
    exit 1
fi
