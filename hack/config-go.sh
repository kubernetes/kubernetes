# Copyright 2014 Google Inc. All rights reserved.
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

# This script sets up a go workspace locally and builds all go components.
# You can 'source' this file if you want to set up GOPATH in your local shell.

if [ "$(which go)" == "" ]; then
	echo "Can't find 'go' in PATH, please fix and retry."
	echo "See http://golang.org/doc/install for installation instructions."
	exit 1
fi

pushd $(dirname "${BASH_SOURCE}")/.. >/dev/null
KUBE_REPO_ROOT="${PWD}"
KUBE_TARGET="${KUBE_REPO_ROOT}/output/go"
popd >/dev/null

mkdir -p "${KUBE_TARGET}"

KUBE_GO_PACKAGE=github.com/GoogleCloudPlatform/kubernetes
export GOPATH="${KUBE_TARGET}"
KUBE_GO_PACKAGE_DIR="${GOPATH}/src/${KUBE_GO_PACKAGE}"

(
  PACKAGE_BASE=$(dirname "${KUBE_GO_PACKAGE_DIR}")
  if [ ! -d "${PACKAGE_BASE}" ]; then
    mkdir -p "${PACKAGE_BASE}"
  fi

  rm "${KUBE_GO_PACKAGE_DIR}" >/dev/null 2>&1 || true
  ln -s "${KUBE_REPO_ROOT}" "${KUBE_GO_PACKAGE_DIR}"

  # Link in each of the third party packages
  THIRD_PARTY_BASE="${KUBE_REPO_ROOT}/third_party"
  source "${THIRD_PARTY_BASE}/deps.sh"
  for p in ${PACKAGES}; do
    PACKAGE_DIR="${GOPATH}/src/${p}"
    PACKAGE_BASE=$(dirname "${PACKAGE_DIR}")

    if [ ! -d "${PACKAGE_BASE}" ]; then
      mkdir -p "${PACKAGE_BASE}"
    fi

    rm "${PACKAGE_DIR}" >/dev/null 2>&1 || true
    ln -s "${THIRD_PARTY_BASE}/${p}" "${PACKAGE_DIR}"
  done

  for p in ${PACKAGES}; do
    go install $p
  done
)
