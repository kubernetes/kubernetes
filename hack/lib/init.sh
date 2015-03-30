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

set -o errexit
set -o nounset
set -o pipefail

# The root of the build/dist directory
LMKTFY_ROOT=$(
  unset CDPATH
  lmktfy_root=$(dirname "${BASH_SOURCE}")/../..
  cd "${lmktfy_root}"
  pwd
)

LMKTFY_OUTPUT_SUBPATH="${LMKTFY_OUTPUT_SUBPATH:-_output/local}"
LMKTFY_OUTPUT="${LMKTFY_ROOT}/${LMKTFY_OUTPUT_SUBPATH}"
LMKTFY_OUTPUT_BINPATH="${LMKTFY_OUTPUT}/bin"

source "${LMKTFY_ROOT}/hack/lib/util.sh"
source "${LMKTFY_ROOT}/hack/lib/logging.sh"

lmktfy::log::install_errexit

source "${LMKTFY_ROOT}/hack/lib/version.sh"
source "${LMKTFY_ROOT}/hack/lib/golang.sh"
source "${LMKTFY_ROOT}/hack/lib/etcd.sh"

LMKTFY_OUTPUT_HOSTBIN="${LMKTFY_OUTPUT_BINPATH}/$(lmktfy::util::host_platform)"
