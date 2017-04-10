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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
KUBE_VERBOSE="${KUBE_VERBOSE:-1}"
source "${KUBE_ROOT}/hack/lib/init.sh"

ARCH_TARBALL_REGEX="^kubernetes-([^-]+)-([^.]+)\.tar.gz$"
RELEASE_TAR_OUTPUT_DIR="${KUBE_ROOT}/_output/release-tars"
RELEASE_STAGE_OUTPUT_DIR="${KUBE_ROOT}/_output/release-stage"

rm -rf "${RELEASE_TAR_OUTPUT_DIR}" "${RELEASE_STAGE_OUTPUT_DIR}"

kube::log::status "Running bazel build"
bazel build //build/release-tars

kube::log::status "Building _output/release-tars and _output/release-stage"
mkdir -p "${RELEASE_TAR_OUTPUT_DIR}"
for tarfile in "${KUBE_ROOT}"/bazel-bin/build/release-tars/*.tar.gz; do
  # Resolve the bazel-bin symlink
  real_tarfile=$(kube::realpath "${tarfile}")
  ln -sf "${real_tarfile}" "${RELEASE_TAR_OUTPUT_DIR}"
  tarfile_name="${real_tarfile##*/}"

  # Extract the arch-specific tarballs into the release-stage directory.
  # This is basically opposite of what the legacy build system does, but this is
  # the easiest way to reuse the rules in build/release-tars/BUILD.
  if [[ "${tarfile_name}" =~ ${ARCH_TARBALL_REGEX} ]]; then
    # _output/release-stage gets dirs like client/linux-amd64,
    # server/linux-amd64, etc.
    output_dir="${RELEASE_STAGE_OUTPUT_DIR}/${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    mkdir -p "${output_dir}"
    tar -C "${output_dir}" -xzf "${real_tarfile}"
  fi
done

# Special-case the "full" kubernetes tarball
output_dir="${RELEASE_STAGE_OUTPUT_DIR}/full"
mkdir -p "${output_dir}"
tar -C "${output_dir}" -xzf \
  "${KUBE_ROOT}/bazel-bin/build/release-tars/kubernetes.tar.gz"
