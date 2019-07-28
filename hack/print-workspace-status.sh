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

# This command is used by bazel as the workspace_status_command
# to implement build stamping with git information.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..

source "${KUBE_ROOT}/hack/lib/version.sh"
kube::version::get_version_vars

# Prefix with STABLE_ so that these values are saved to stable-status.txt
# instead of volatile-status.txt.
# Stamped rules will be retriggered by changes to stable-status.txt, but not by
# changes to volatile-status.txt.
# IMPORTANT: the camelCase vars should match the lists in hack/lib/version.sh
# and staging/src/k8s.io/kubectl/pkg/version/def.bzl.
cat <<EOF
STABLE_BUILD_GIT_COMMIT ${KUBE_GIT_COMMIT-}
STABLE_BUILD_SCM_STATUS ${KUBE_GIT_TREE_STATE-}
STABLE_BUILD_SCM_REVISION ${KUBE_GIT_VERSION-}
STABLE_BUILD_MAJOR_VERSION ${KUBE_GIT_MAJOR-}
STABLE_BUILD_MINOR_VERSION ${KUBE_GIT_MINOR-}
STABLE_DOCKER_TAG ${KUBE_GIT_VERSION/+/_}
STABLE_DOCKER_REGISTRY ${KUBE_DOCKER_REGISTRY:-k8s.gcr.io}
STABLE_DOCKER_PUSH_REGISTRY ${KUBE_DOCKER_PUSH_REGISTRY:-${KUBE_DOCKER_REGISTRY:-staging-k8s.gcr.io}}
gitCommit ${KUBE_GIT_COMMIT-}
gitTreeState ${KUBE_GIT_TREE_STATE-}
gitVersion ${KUBE_GIT_VERSION-}
gitMajor ${KUBE_GIT_MAJOR-}
gitMinor ${KUBE_GIT_MINOR-}
buildDate $(date \
  ${SOURCE_DATE_EPOCH:+"--date=@${SOURCE_DATE_EPOCH}"} \
 -u +'%Y-%m-%dT%H:%M:%SZ')
EOF
