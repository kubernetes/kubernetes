#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# This script sets up a go workspace locally and builds all for all appropriate
# platforms.

set -o errexit
set -o nounset
set -o pipefail

LMKTFY_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${LMKTFY_ROOT}/hack/lib/init.sh"

LMKTFY_BUILD_PLATFORMS=("${LMKTFY_SERVER_PLATFORMS[@]}")
lmktfy::golang::build_binaries "${LMKTFY_SERVER_TARGETS[@]}"

LMKTFY_BUILD_PLATFORMS=("${LMKTFY_CLIENT_PLATFORMS[@]}")
lmktfy::golang::build_binaries "${LMKTFY_CLIENT_TARGETS[@]}" "${LMKTFY_TEST_TARGETS[@]}"

lmktfy::golang::place_bins
