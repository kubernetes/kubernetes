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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::util::ensure_clean_working_dir
kube::util::ensure_godep_version v79

cd ${KUBE_ROOT}

echo "Checking whether godeps are restored"
if ! kube::util::godep_restored 2>&1 | sed 's/^/  /'; then
  echo -e '\nExecute script 'hack/godep-restore.sh' to download dependencies.' 1>&2
  exit 1
fi

echo "Running staging/copy.sh"
staging/copy.sh -u "$@" 2>&1 | sed 's/^/  /'
