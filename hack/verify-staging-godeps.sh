#!/usr/bin/env bash

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

readonly branch=${1:-${KUBE_VERIFY_GIT_BRANCH:-master}}
if ! [[ ${KUBE_FORCE_VERIFY_CHECKS:-} =~ ^[yY]$ ]] && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'staging/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'build/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'Godeps/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'vendor/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'hack/lib/' && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'hack/.*godep'; then
  exit 0
fi

KUBE_VERBOSE="${KUBE_VERBOSE:-3}" KUBE_RUN_COPY_OUTPUT=N ${KUBE_ROOT}/hack/update-staging-godeps.sh -d -f "$@"
