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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

git_upstream=$(kube::util::git_upstream_remote_name)
: ${git_upstream:="upstream"}

kube::golang::setup_env

make -C "${KUBE_ROOT}/" WHAT=cmd/mungedocs

# Find binary
mungedocs=$(kube::util::find-binary "mungedocs")

DOCROOT="${KUBE_ROOT}/docs/"
EXAMPLEROOT="${KUBE_ROOT}/examples/"

# mungedocs --verify can (and should) be run on the real docs, otherwise their
# links will be distorted. --verify means that it will not make changes.
# --verbose gives us output we can use for a diff.
"${mungedocs}" "--verify=true" "--verbose=true" "--upstream=${git_upstream}" "--root-dir=${DOCROOT}" && ret=0 || ret=$?
if [[ $ret -eq 1 ]]; then
  echo "${DOCROOT} is out of date. Please run hack/update-munge-docs.sh"
  exit 1
fi
if [[ $ret -gt 1 ]]; then
  echo "Error running mungedocs"
  exit 1
fi

"${mungedocs}" "--verify=true" "--verbose=true" "--upstream=${git_upstream}" "--root-dir=${EXAMPLEROOT}" && ret=0 || ret=$?
if [[ $ret -eq 1 ]]; then
  echo "${EXAMPLEROOT} is out of date. Please run hack/update-munge-docs.sh"
  exit 1
fi
if [[ $ret -gt 1 ]]; then
  echo "Error running mungedocs"
  exit 1
fi

"${mungedocs}" "--verify=true" "--verbose=true" \
               "--upstream=${git_upstream}" \
               "--skip-munges=unversioned-warning,analytics" \
               "--norecurse" \
               "--root-dir=${KUBE_ROOT}/" && ret=0 || ret=$?
if [[ $ret -eq 1 ]]; then
  echo "${KUBE_ROOT}/ is out of date. Please run hack/update-munge-docs.sh"
  exit 1
elif [[ $ret -gt 1 ]]; then
  echo "Error running mungedocs."
  exit 1
fi

needsanalytics=($(kube::util::gen-analytics "${KUBE_ROOT}" 1))
if [[ ${#needsanalytics[@]} -ne 0 ]]; then
  echo -e "Some md files are missing ga-beacon analytics link:"
  printf '%s\n' "${needsanalytics[@]}"
  ret=1
fi
if [[ $ret -eq 0 ]]; then
  echo "Docs are properly munged."
else
  echo "Docs need munging. Please run hack/update-munge-docs.sh"
  exit 1
fi

# ex: ts=2 sw=2 et filetype=sh
