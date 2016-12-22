#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

# TODO: remove this script when we have a standalone TOC generator.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

git_upstream=$(kube::util::git_upstream_remote_name)
: ${git_upstream:="upstream"}

kube::golang::setup_env

make -C "${KUBE_ROOT}" WHAT=cmd/mungedocs

kube::util::ensure-temp-dir

kube::util::gen-analytics "${KUBE_ROOT}"

mungedocs=$(kube::util::find-binary "mungedocs")

"${mungedocs}" "--upstream=${git_upstream}" "--root-dir=${KUBE_ROOT}/docs/" \
 && ret=0 || ret=$?

if [[ $ret -eq 1 ]]; then
  echo "${KUBE_ROOT}/docs/ requires manual changes.  See preceding errors."
  exit 1
elif [[ $ret -gt 1 ]]; then
  echo "Error running mungedocs."
  exit 1
fi

"${mungedocs}" "--upstream=${git_upstream}" \
               "--root-dir=${KUBE_ROOT}/examples/" && ret=0 || ret=$?

if [[ $ret -eq 1 ]]; then
  echo "${KUBE_ROOT}/examples/ requires manual changes.  See preceding errors."
  exit 1
elif [[ $ret -gt 1 ]]; then
  echo "Error running mungedocs."
  exit 1
fi

"${mungedocs}" "--upstream=${git_upstream}" \
               "--skip-munges=analytics" \
               "--norecurse" \
               "--root-dir=${KUBE_ROOT}/" && ret=0 || ret=$?
if [[ $ret -eq 1 ]]; then
  echo "${KUBE_ROOT}/ requires manual changes.  See preceding errors."
  exit 1
elif [[ $ret -gt 1 ]]; then
  echo "Error running mungedocs."
  exit 1
fi

# ex: ts=2 sw=2 et filetype=sh
