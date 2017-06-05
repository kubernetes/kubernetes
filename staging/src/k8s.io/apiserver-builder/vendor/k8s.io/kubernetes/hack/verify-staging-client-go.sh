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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
cd ${KUBE_ROOT}

# Smoke test client-go examples
echo "Smoke testing client-go examples"
go install ./staging/src/k8s.io/client-go/examples/... 2>&1 | sed 's/^/  /'

# Run update-staging-client.sh in dry-run mode, copy nothing into the staging dir, but fail on any diff
hack/update-staging-client-go.sh -d -f "$@"
