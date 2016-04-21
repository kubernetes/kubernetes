#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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
source "${KUBE_ROOT}/build/common.sh"

kube::golang::setup_env

function prereqs() {
  kube::build::ensure_docker_in_path || return 1
  if kube::build::is_osx; then
      kube::build::docker_available_on_osx || return 1
  fi
  kube::build::ensure_docker_daemon_connectivity || return 1
}

if ! prereqs; then
  echo "Verifying protobuf requires a connection to a Docker daemon"
  echo
  echo "WARNING: Protobuf changes are not being verified"
  exit 0
fi

"${KUBE_ROOT}/hack/after-build/verify-generated-protobuf.sh" "$@"

# ex: ts=2 sw=2 et filetype=sh
