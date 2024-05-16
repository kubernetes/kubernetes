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

# This script runs `make test` command with some args for benchmark test.
# The "true" target of this makerule is `hack/make-rules/test.sh`.
# Args:
#   WHAT: Directory names to test.  All *_test.go files under these
#     directories will be run.  If not specified, "everything" will be tested.
# Usage: `hack/benchmark-go.sh`.
# Example: `hack/benchmark-go.sh WHAT=./pkg/kubelet`.

set -o errexit
set -o nounset
set -o pipefail

make test \
    WHAT="$*" \
    KUBE_COVER="" \
    KUBE_RACE=" " \
    KUBE_TEST_ARGS="-- -test.run='^X' -benchtime=1s -bench=. -benchmem" \
