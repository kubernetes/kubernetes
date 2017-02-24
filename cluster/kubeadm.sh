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

KUBE_ROOT=${KUBE_ROOT:-$(dirname "${BASH_SOURCE}")/..}
source "${KUBE_ROOT}/cluster/kube-util.sh"
source "${KUBE_ROOT}/cluster/clientbin.sh"

# If KUBEADM_PATH isn't set, gather up the list of likely places and use ls
# to find the latest one.
if [[ -z "${KUBEADM_PATH:-}" ]]; then
  kubeadm=$( get_bin "kubeadm" "cmd/kubeadm" )

  if [[ ! -x "$kubeadm" ]]; then
    print_error "kubeadm"
    exit 1
  fi
elif [[ ! -x "${KUBEADM_PATH}" ]]; then
  {
    echo "KUBEADM_PATH environment variable set to '${KUBEADM_PATH}', but "
    echo "this doesn't seem to be a valid executable."
  } >&2
  exit 1
fi
kubeadm="${KUBEADM_PATH:-${kubeadm}}"

"${kubeadm}" "${@+$@}"
