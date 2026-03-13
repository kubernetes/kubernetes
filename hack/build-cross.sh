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

# This script is a vestigial redirection.  Please do not add "real" logic.
# The "true" target of this makerule is `hack/make-rules/cross.sh`.

# This script runs `make cross` command.
# The command sets up a go workspace locally and builds all for all appropriate
# platforms.
# Usage: `hack/build-cross.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

echo "NOTE: $0 has been replaced by 'make cross'"
echo
echo "The equivalent of this invocation is: "
echo "    make cross"
echo
echo
make --no-print-directory -C "${KUBE_ROOT}" cross
