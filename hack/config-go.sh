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

# --- Helper Functions ---

# Function kube::version_ldflags() prints the value that needs to be passed to
# the -ldflags parameter of go build in order to set the Kubernetes based on the
# git tree status.
kube::version_ldflags() {
  (
    # Run this in a subshell to prevent settings/variables from leaking.
    set -o errexit
    set -o nounset
    set -o pipefail

    unset CDPATH

    cd "${KUBE_REPO_ROOT}"

    declare -a ldflags=()
    if git_commit=$(git rev-parse "HEAD^{commit}" 2>/dev/null); then
      ldflags+=(-X "${KUBE_GO_PACKAGE}/pkg/version.gitCommit" "${git_commit}")

      # Check if the tree is dirty.
      if git_status=$(git status --porcelain) && [[ -z "${git_status}" ]]; then
        git_tree_state="clean"
      else
        git_tree_state="dirty"
      fi
      ldflags+=(-X "${KUBE_GO_PACKAGE}/pkg/version.gitTreeState" "${git_tree_state}")

      # Use git describe to find the version based on annotated tags.
      if git_version=$(git describe --abbrev=14 "${git_commit}^{commit}" 2>/dev/null); then
        ldflags+=(-X "${KUBE_GO_PACKAGE}/pkg/version.gitVersion" "${git_version}")
      fi
    fi

    # The -ldflags parameter takes a single string, so join the output.
    echo "${ldflags[*]-}"
  )
}

# kube::setup_go_environment will check that the `go` commands is available in
# ${PATH}. If not running on Travis, it will also check that the Go version is
# good enough for the Kubernetes build.
#
# Also set ${GOPATH} and environment variables needed by Go.
kube::setup_go_environment() {
  if [[ -z "$(which go)" ]]; then
    echo "Can't find 'go' in PATH, please fix and retry." >&2
    echo "See http://golang.org/doc/install for installation instructions." >&2
    exit 1
  fi

  # Travis continuous build uses a head go release that doesn't report
  # a version number, so we skip this check on Travis.  Its unnecessary
  # there anyway.
  if [[ "${TRAVIS:-}" != "true" ]]; then
    local go_version
    go_version=($(go version))
    if [[ "${go_version[2]}" < "go1.2" ]]; then
      echo "Detected go version: ${go_version[*]}." >&2
      echo "Kubernetes requires go version 1.2 or greater." >&2
      echo "Please install Go version 1.2 or later" >&2
      exit 1
    fi
  fi

  # Set GOPATH to point to the tree maintained by `godep`.
  GOPATH="${KUBE_TARGET}:${KUBE_REPO_ROOT}/Godeps/_workspace"
  export GOPATH

  # Unset GOBIN in case it already exsits in the current session.
  unset GOBIN
}


# --- Environment Variables ---

# KUBE_REPO_ROOT  - Path to the top of the build tree.
# KUBE_TARGET     - Path where output Go files are saved.
# KUBE_GO_PACKAGE - Full name of the Kubernetes Go package.

# Make ${KUBE_REPO_ROOT} an absolute path.
KUBE_REPO_ROOT=$(
  set -eu
  unset CDPATH
  scripts_dir=$(dirname "${BASH_SOURCE[0]}")
  cd "${scripts_dir}"
  cd ..
  pwd
)
export KUBE_REPO_ROOT

KUBE_TARGET="${KUBE_REPO_ROOT}/_output/go"
mkdir -p "${KUBE_TARGET}"
export KUBE_TARGET

KUBE_GO_PACKAGE=github.com/GoogleCloudPlatform/kubernetes
export KUBE_GO_PACKAGE

(
  # Create symlink named ${KUBE_GO_PACKAGE} under _output/go/src.
  # So that Go knows how to import Kubernetes sources by full path.
  # Use a subshell to avoid leaking these variables.

  set -eu
  go_pkg_dir="${KUBE_TARGET}/src/${KUBE_GO_PACKAGE}"
  go_pkg_basedir=$(dirname "${go_pkg_dir}")
  mkdir -p "${go_pkg_basedir}"
  rm -f "${go_pkg_dir}"
  # TODO: This symlink should be relative.
  ln -s "${KUBE_REPO_ROOT}" "${go_pkg_dir}"
)

