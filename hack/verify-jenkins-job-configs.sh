#!/bin/bash

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

# Tests the Jenkins job configs and computes a diff of any changes when there
# have been local changes of the configs.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

readonly branch=${1:-${KUBE_VERIFY_GIT_BRANCH:-master}}
if ! [[ ${KUBE_FORCE_VERIFY_CHECKS:-} =~ ^[yY]$ ]] && \
  ! kube::util::has_changes_against_upstream_branch "${branch}" 'hack/jenkins/job-configs/'; then
  exit 0
fi

# By using ARTIFACTS_DIR, we can write the diff out to the artifacts directory
# (and then up to GCS) when running on Jenkins.
export OUTPUT_DIR="${ARTIFACTS_DIR:+${ARTIFACTS_DIR}/jjb}"
# When running inside Docker (e.g. on Jenkins) we'll need to reference the
# host's artifacts directory for the Docker-in-Docker volume mount to work.
export DOCKER_VOLUME_OUTPUT_DIR="${HOST_ARTIFACTS_DIR:+${HOST_ARTIFACTS_DIR}/jjb}"

# This script should pass, assuming the configs are not broken. Diffs won't
# cause failures.
"${KUBE_ROOT}/hack/jenkins/diff-job-config-patch.sh"
