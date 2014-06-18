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

# Travis continuous build uses a head go release that doesn't report
# a version number, so we skip this check on Travis.  Its unnecessary
# there anyway.
if [ "${TRAVIS}" != "true" ]; then
  GO_VERSION=($(go version))

  if [ ${GO_VERSION[2]} \< "go1.2" ]; then
    echo "Detected go version: ${GO_VERSION}."
    echo "Kubernetes requires go version 1.2 or greater."
    echo "Please install Go version 1.2 or later"
    exit 1
  fi
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
)
export GOPATH="${KUBE_TARGET}:${KUBE_REPO_ROOT}/third_party"
