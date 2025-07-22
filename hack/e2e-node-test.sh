#!/usr/bin/env bash

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

# This script is a vestigial redirection.  Please do not add "real" logic.
# The "true" target of this makerule is `hack/make-rules/test-e2e-node.sh`.

# This script runs `make test-e2e-node` command.
# The command builds and runs node end-to-end tests.
# Args:
#  FOCUS: Regexp that matches the tests to be run.  Defaults to "".
#  SKIP: Regexp that matches the tests that needs to be skipped.  Defaults
# Usage: `hack/e2e-node-test.sh `.
# Example: `hack/e2e-node-test.sh FOCUS=Kubelet SKIP=container`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

# For help output
ARGHELP=""
if [[ -n "${FOCUS:-}" ]]; then
    ARGHELP="FOCUS='${FOCUS}' "
fi
if [[ -n "${SKIP:-}" ]]; then
    ARGHELP="${ARGHELP}SKIP='${SKIP}'"
fi

echo "NOTE: $0 has been replaced by 'make test-e2e-node'"
echo
echo "This script supports a number of parameters passed as environment variables."
echo "Please see the Makefile for more details."
echo
echo "The equivalent of this invocation is: "
echo "    make test-e2e-node ${ARGHELP}"
echo
echo
make --no-print-directory -C "${KUBE_ROOT}" test-e2e-node FOCUS="${FOCUS:-}" SKIP="${SKIP:-}"
