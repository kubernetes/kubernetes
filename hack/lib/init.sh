#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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
KUBE_ROOT=$(
  unset CDPATH
  kube_root=$(dirname "${BASH_SOURCE}")/../..
  cd "${kube_root}"
  pwd
)

KUBE_OUTPUT_SUBPATH="${KUBE_OUTPUT_SUBPATH:-_output/local}"
KUBE_OUTPUT="${KUBE_ROOT}/${KUBE_OUTPUT_SUBPATH}"
KUBE_OUTPUT_BINPATH="${KUBE_OUTPUT}/bin"

source "${KUBE_ROOT}/hack/lib/util.sh"
source "${KUBE_ROOT}/cluster/lib/util.sh"
source "${KUBE_ROOT}/cluster/lib/logging.sh"

kube::log::install_errexit

source "${KUBE_ROOT}/hack/lib/version.sh"
source "${KUBE_ROOT}/hack/lib/golang.sh"
source "${KUBE_ROOT}/hack/lib/etcd.sh"

KUBE_GIT_UPSTREAM="${KUBE_GIT_UPSTREAM:-upstream}"

KUBE_OUTPUT_HOSTBIN="${KUBE_OUTPUT_BINPATH}/$(kube::util::host_platform)"

# emulates "readlink -f" which is not available on BSD (OS X).
function readlinkdashf {
  path=$1
  # Follow links until there are no more links to follow.
  while readlink "$path"; do
    path="$(readlink $path)"
  done
  # Convert to canonical path.
  path=$(cd "$(dirname "${path}")" && pwd -P)
  echo "$path"
}

