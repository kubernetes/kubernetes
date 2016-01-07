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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env
linkcheck=$(kube::util::find-binary "linkcheck")

TYPEROOT="${KUBE_ROOT}/pkg/api/"
"${linkcheck}" "--root-dir=${TYPEROOT}" "--repo-root=${KUBE_ROOT}" "--file-suffix=types.go" "--prefix=http://releases.k8s.io/HEAD" && ret=0 || ret=$?
if [[ $ret -eq 1 ]]; then
  echo "links in ${TYPEROOT} is out of date."
  exit 1
fi
if [[ $ret -gt 1 ]]; then
  echo "Error running linkcheck"
  exit 1
fi

# ex: ts=2 sw=2 et filetype=sh
