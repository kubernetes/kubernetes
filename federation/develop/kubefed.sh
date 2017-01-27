#!/bin/bash

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

set -o errexit
set -o nounset
set -o pipefail

# "-=-=-=-=-=-=-=-=-=-="
# This script is only for CI testing purposes. Don't use it in production.
# "-=-=-=-=-=-=-=-=-=-="

KUBE_ROOT=${KUBE_ROOT:-$(dirname "${BASH_SOURCE}")/../..}
source "${KUBE_ROOT}/cluster/clientbin.sh"

# If KUBEFED_PATH isn't set, gather up the list of likely places and use ls
# to find the latest one.
if [[ -z "${KUBEFED_PATH:-}" ]]; then
  kubefed=$( get_bin "kubefed" "federation/cmd/kubefed" )

  if [[ ! -x "$kubefed" ]]; then
    print_error "kubefed"
    exit 1
  fi
elif [[ ! -x "${KUBEFED_PATH}" ]]; then
  {
    echo "KUBEFED_PATH environment variable set to '${KUBEFED_PATH}', but "
    echo "this doesn't seem to be a valid executable."
  } >&2
  exit 1
fi
kubefed="${KUBEFED_PATH:-${kubefed}}"

# Use the arguments to the script if it is set, a null string
# otherwise.
"${kubefed}" "${@+$@}"
