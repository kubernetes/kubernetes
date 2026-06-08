#!/usr/bin/env bash

# Copyright 2019 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

# Nuke old files so we don't accidentally carry stuff forward.
rm -f staging/src/k8s.io/api/testdata/HEAD/*.{yaml,json,pb}
rm -f staging/src/k8s.io/apiextensions-apiserver/pkg/apis/testdata/HEAD/*.{yaml,json,pb}

# UPDATE_COMPATIBILITY_FIXTURE_DATA=true regenerates fixture data if needed.
# -run //HEAD only runs the test cases comparing against testdata for HEAD.
# We suppress the output because we are expecting to have changes.
# We suppress the test failure that occurs when there are changes.
UPDATE_COMPATIBILITY_FIXTURE_DATA=true go test k8s.io/api -run //HEAD >/dev/null 2>&1 || true
UPDATE_COMPATIBILITY_FIXTURE_DATA=true go test k8s.io/apiextensions-apiserver/pkg/apis -run //HEAD >/dev/null 2>&1 || true

# Now that we have regenerated data at HEAD, run the test without suppressing output or failures
go test k8s.io/api -run //HEAD -count=1
go test k8s.io/apiextensions-apiserver/pkg/apis -run //HEAD -count=1
