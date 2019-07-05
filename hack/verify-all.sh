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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

# For help output
ARGHELP=""
if [[ -n "${KUBE_VERIFY_GIT_BRANCH:-}" ]]; then
    ARGHELP="BRANCH=${KUBE_VERIFY_GIT_BRANCH}"
fi

echo "NOTE: $0 has been replaced by 'make verify'"
echo
echo "The equivalent of this invocation is: "
echo "    make verify ${ARGHELP}"
echo
echo
make --no-print-directory -C "${KUBE_ROOT}" verify BRANCH="${KUBE_VERIFY_GIT_BRANCH:-}"
