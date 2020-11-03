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

# NOTE: This script must be run from INSIDE the compiler image becaues it's the
# compiler's host's licenses that we want to extract back out.

# Tars up host source dependences into the _output directory, and licenses into a _output/src/LICENSES/host directory.

set -o errexit
set -o nounset
set -o pipefail

mkdir -p "${KUBE_ROOT}/${KUBE_OUTPUT_SUBPATH}/src"
rm -f "${KUBE_ROOT}/${KUBE_OUTPUT_SUBPATH}/src/*.tar"
tar cf "${KUBE_ROOT}/${KUBE_OUTPUT_SUBPATH}/src/glibc.tar" -C /usr/src glibc

HOST_LICENSES_DIR="${KUBE_ROOT}/${KUBE_OUTPUT_SUBPATH}/src/LICENSES/host"
rm -fr "${HOST_LICENSES_DIR}"

format_license() {
  local -r host_licenses_dir=$1
  local -r dep=$2
  local -r src_license=$3
  local dep_license_dir="${host_licenses_dir}/${dep}"
  mkdir -p "${dep_license_dir}"
  (
    echo
    echo "= ${dep} licensed under: ="
    echo
    cat "${src_license}"
    echo
  ) >"${dep_license_dir}/LICENSE"
}

# If you change this list, also be sure to change build/licenses.bzl.
format_license "${HOST_LICENSES_DIR}" "glibc"    /usr/src/glibc/debian/copyright
format_license "${HOST_LICENSES_DIR}" "go"       /usr/local/go/LICENSE
format_license "${HOST_LICENSES_DIR}" "goboring" /usr/local/go/src/crypto/internal/boring/LICENSE
