#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors.
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

# Smoke Tests a running Kubernetes cluster.
# Validates that the cluster was deployed, is accessible, and at least
# satisfies minimal functional requirements.
# Emphasis on speed and being non-destructive over thoroughness.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

SMOKE_TEST_FOCUS_REGEX="Guestbook.application"

exec "${KUBE_ROOT}/cluster/test-e2e.sh" -ginkgo.focus="${SMOKE_TEST_FOCUS_REGEX}" "$@"
