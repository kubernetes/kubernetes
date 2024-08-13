#!/usr/bin/env bash

# Copyright 2022 The Kubernetes Authors.
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

# This script runs to ensure that we do not violate metric stability
# policies.
# Usage: `test/instrumentation/test-update.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
source "${KUBE_ROOT}/test/instrumentation/stability-utils.sh"
source "${KUBE_ROOT}/hack/lib/version.sh"

# extract version env variables so we can pass them in
kube::version::get_version_vars

# update the documented list of metrics
kube::update::documentation::list
# now write the actual documentation file
kube::update::documentation "$KUBE_GIT_MAJOR" "$KUBE_GIT_MINOR"
