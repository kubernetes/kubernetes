#!/bin/bash

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

# gitcommit prints the current Git commit information
function gitcommit() {
  set -o errexit
  set -o nounset
  set -o pipefail

  topdir=$(dirname "$0")/..
  cd "${topdir}"

  # TODO: when we start making tags, switch to git describe?
  if git_commit=$(git rev-parse --short "HEAD^{commit}" 2>/dev/null); then
    # Check if the tree is dirty.
    if ! dirty_tree=$(git status --porcelain) || [[ -n "${dirty_tree}" ]]; then
      echo "${git_commit}-dirty"
    else
      echo "${git_commit}"
    fi
  else
    echo "(none)"
  fi
  return 0
}

if [[ -z "$(which go)" ]]; then
  echo "Can't find 'go' in PATH, please fix and retry." >&2
  echo "See http://golang.org/doc/install for installation instructions." >&2
  exit 1
fi

if [[ -z "$(which godep)" ]]; then
  echo "Can't find 'godep' in PATH, please fix and retry." >&2
  echo "See https://github.com/GoogleCloudPlatform/kubernetes#godep-and-dependency-management" >&2
  exit 1
fi

# Travis continuous build uses a head go release that doesn't report
# a version number, so we skip this check on Travis.  Its unnecessary
# there anyway.
if [[ "${TRAVIS:-}" != "true" ]]; then
  GO_VERSION=($(go version))
  if [[ "${GO_VERSION[2]}" < "go1.2" ]]; then
    echo "Detected go version: ${GO_VERSION[*]}." >&2
    echo "Kubernetes requires go version 1.2 or greater." >&2
    echo "Please install Go version 1.2 or later" >&2
    exit 1
  fi
fi

KUBE_REPO_ROOT=$(dirname "${BASH_SOURCE:-$0}")/..
if [[ "${OSTYPE:-}" == *darwin* ]]; then
  # Make the path absolute if it is not.
  if [[ "${KUBE_REPO_ROOT}" != /* ]]; then
    KUBE_REPO_ROOT=${PWD}/${KUBE_REPO_ROOT}
  fi
else
  # Resolve symlinks.
  KUBE_REPO_ROOT=$(readlink -f "${KUBE_REPO_ROOT}")
fi

KUBE_TARGET="${KUBE_REPO_ROOT}/output/go"
mkdir -p "${KUBE_TARGET}"

KUBE_GO_PACKAGE=github.com/GoogleCloudPlatform/kubernetes
KUBE_GO_PACKAGE_DIR="${KUBE_TARGET}/src/${KUBE_GO_PACKAGE}"

KUBE_GO_PACKAGE_BASEDIR=$(dirname "${KUBE_GO_PACKAGE_DIR}")
mkdir -p "${KUBE_GO_PACKAGE_BASEDIR}"

# Create symlink under output/go/src.
ln -snf "${KUBE_REPO_ROOT}" "${KUBE_GO_PACKAGE_DIR}"

GOPATH="${KUBE_TARGET}:$(godep path)"
export GOPATH

# Unset GOBIN in case it already exsits in the current session.
unset GOBIN
