#!/usr/bin/env bash

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

# This file creates release artifacts (tar files, container images) that are
# ready to distribute to install or distribute to end users.

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
source "${KUBE_ROOT}/hack/lib/logging.sh"

RELEASE_TEST_GCR_REPO=${RELEASE_TEST_GCR_REPO:-"gcr.io/k8s-staging-release-test"}

# Package the source code we built, for compliance/licensing/audit/yadda.
function kube::release::package_src_tarball() {
  if [[ ! -d "${KUBE_ROOT}/_output/dockerized/bin/linux" ]]; then
      kube::log::error "unable ${KUBE_ROOT}/_output/dockerized/bin/linux is missing"
      return 1
  fi

  mkdir -p "${KUBE_ROOT}/_output/debs/"
  docker run -it --env KUBE_USE_LOCAL_ARTIFACTS=y \
    --user `id -u` \
    --volume=$(readlink -f "${KUBE_ROOT}/_output/dockerized/bin/linux"):/src/k8s.io/kubernetes/_output/dockerized/bin/linux \
    --volume=$(readlink -f "${KUBE_ROOT}/_output/debs"):/home/builder/workspace/bin:rw \
    --rm ${RELEASE_TEST_GCR_REPO}/deb-builder
  mkdir -p "${KUBE_ROOT}/_output/rpms/"
  docker run -it --env KUBE_USE_LOCAL_ARTIFACTS=y \
    --volume=$(readlink -f "${KUBE_ROOT}/_output/dockerized/bin/linux"):/src/k8s.io/kubernetes/_output/dockerized/bin/linux \
    --volume=$(readlink -f "${KUBE_ROOT}/_output/rpms"):/home/builder/rpmbuild:rw \
    --rm ${RELEASE_TEST_GCR_REPO}/rpm-builder
}
