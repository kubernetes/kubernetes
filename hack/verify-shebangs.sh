#!/usr/bin/env bash

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

# Verifies that api reference docs are up to date.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

(
  cd "${KUBE_ROOT}"
  badfiles=$(grep -irclP -I '^#!(?!\/usr\/bin\/env).*' --exclude-dir="_output" --exclude-dir="Godeps")
  if [[ ! -z "${badfiles}" ]]; then
    echo "The following files have shebangs that do not use /usr/bin/env. Please fix them. ${badfiles[@]}"
  fi
)

# ex: ts=2 sw=2 et filetype=sh
