#!/usr/bin/env bash
# Copyright 2018 The Kubernetes Authors.
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

# This script checks commonly misspelled English words in all files in the
# working directory by client9/misspell package.
# Usage: `hack/verify-spelling.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
export KUBE_ROOT
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BIN}"
PATH="${GOBIN}:${PATH}"

# Install tools we need
GOTOOLCHAIN="$(kube::golang::hack_tools_gotoolchain)" go -C "${KUBE_ROOT}/hack/tools" install github.com/client9/misspell/cmd/misspell

# Spell checking
# All the skipping files are defined in hack/.spelling_failures
skipping_file="${KUBE_ROOT}/hack/.spelling_failures"
failing_packages=$(sed "s| | -e |g" "${skipping_file}")
git ls-files | grep -v -e "${failing_packages}" | xargs misspell -i "Creater,creater,ect" -error -o stderr
