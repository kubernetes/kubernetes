#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
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
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

make -C "${KUBE_ROOT}" WHAT=cmd/importverifier

# Find binary
importverifier=$(kube::util::find-binary "importverifier")

if [[ ! -x "$importverifier" ]]; then
  {
    echo "It looks as if you don't have a compiled importverifier binary"
    echo
    echo "If you are running from a clone of the git repo, please run"
    echo "'make WHAT=cmd/importverifier'."
  } >&2
  exit 1
fi

"${importverifier}" "k8s.io/" "${KUBE_ROOT}/hack/staging-import-restrictions.json"