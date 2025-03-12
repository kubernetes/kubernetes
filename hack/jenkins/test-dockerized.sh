#!/usr/bin/env bash

# Copyright 2015 The Kubernetes Authors.
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
set -o xtrace

# Runs test-cmd and test-integration, intended to be run in prow.k8s.io
set -x;

# TODO: make test-integration should handle this automatically
source ./hack/install-etcd.sh

# Save the verbose stdout as well.
export KUBE_KEEP_VERBOSE_TEST_OUTPUT=y
export LOG_LEVEL=4
make test-cmd
make test-integration
