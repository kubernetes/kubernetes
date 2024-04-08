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

set -o errexit
set -o nounset
set -o pipefail

# This will canonicalize the path
KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd -P)
cd "${KUBE_ROOT}"

# build ginkgo and e2e.test
#
# Set DBG=1 to build with embedded filenames as filenames rather than
# module-relative names (e.g. /src/kube/foo/bar.go vs.
# k8s.io/kubernetes/foo/bar.go).  These names are used by gingko in
# `--spec-dump` which is consumed later in conformance verification. 
DBG=1 hack/make-rules/build.sh github.com/onsi/ginkgo/v2/ginkgo test/e2e/e2e.test

# dump spec
./_output/bin/ginkgo --dry-run=true --focus='[Conformance]' ./_output/bin/e2e.test -- --spec-dump "${KUBE_ROOT}/_output/specsummaries.json" > /dev/null
