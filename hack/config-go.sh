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


# --- Environment Variables used as inputs by config-go.sh ---
# Function: kube::version_ldflags()
#  These variables are only needed if you are not building from a clone of the
#  full kubernetes repository.
#    KUBE_GIT_COMMIT - Used to set the git commit id corresponding to this
#		       source code.
#    KUBE_GIT_TREE_STATE - "clean" indicates no changes since the git commit id
#			   "dirty" indicates source code changes after the git
#				   commit id
#    KUBE_GIT_VERSION - "vX.Y" used to indicate the last release version.
#
# Function: kube::setup_go_environment
#    KUBE_EXTRA_GOPATH - Value will be appended to the GOPATH after this project
#			 but before the Godeps.
#    KUBE_NO_GODEPS - If set to any value the Godeps will not be included in
#		      the final GOPATH.

# --- Environment Variables set by sourcing config-go.sh ---
# KUBE_ROOT       - Path to the top of the build tree.
# KUBE_TARGET     - Path where output Go files are saved.
# KUBE_GO_PACKAGE - Full name of the Kubernetes Go package.

# --- Environment Variables set by running functions ---
# Function: kube::setup_go_environment
#    GOPATH - Will be set to include this project, anything you have you set in
#	      in ${KUBE_EXTRA_GOPATH}, and the Godeps if ${KUBE_NO_GODEPS} is
#	      unset.


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

    cd "${KUBE_ROOT}"

    declare -a ldflags=()
    if [[ -n ${KUBE_GIT_COMMIT-} ]] || KUBE_GIT_COMMIT=$(git rev-parse "HEAD^{commit}" 2>/dev/null); then
      ldflags+=(-X "${KUBE_GO_PACKAGE}/pkg/version.gitCommit" "${KUBE_GIT_COMMIT}")

      if [[ -z ${KUBE_GIT_TREE_STATE-} ]]; then
        # Check if the tree is dirty.  default to dirty
        if git_status=$(git status --porcelain 2>/dev/null) && [[ -z ${git_status} ]]; then
          KUBE_GIT_TREE_STATE="clean"
        else
          KUBE_GIT_TREE_STATE="dirty"
        fi
      fi
      ldflags+=(-X "${KUBE_GO_PACKAGE}/pkg/version.gitTreeState" "${KUBE_GIT_TREE_STATE}")

      # Use git describe to find the version based on annotated tags.
      if [[ -n ${KUBE_GIT_VERSION-} ]] || KUBE_GIT_VERSION=$(git describe --tag --abbrev=14 "${KUBE_GIT_COMMIT}^{commit}" 2>/dev/null); then
        if [[ "${KUBE_GIT_TREE_STATE}" == "dirty" ]]; then
          # git describe --dirty only considers changes to existing files, but
          # that is problematic since new untracked .go files affect the build,
          # so use our idea of "dirty" from git status instead.
          KUBE_GIT_VERSION+="-dirty"
        fi
        ldflags+=(-X "${KUBE_GO_PACKAGE}/pkg/version.gitVersion" "${KUBE_GIT_VERSION}")

        # Try to match the "git describe" output to a regex to try to extract
        # the "major" and "minor" versions and whether this is the exact tagged
        # version or whether the tree is between two tagged versions.
        if [[ "${KUBE_GIT_VERSION}" =~ ^v([0-9]+)\.([0-9]+)([.-].*)?$ ]]; then
          git_major=${BASH_REMATCH[1]}
          git_minor=${BASH_REMATCH[2]}
          if [[ -n "${BASH_REMATCH[3]}" ]]; then
            git_minor+="+"
          fi
          ldflags+=(
            -X "${KUBE_GO_PACKAGE}/pkg/version.gitMajor" "${git_major}"
            -X "${KUBE_GO_PACKAGE}/pkg/version.gitMinor" "${git_minor}"
          )
        fi
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

  GOPATH=${KUBE_TARGET}
  # Append KUBE_EXTRA_GOPATH to the GOPATH if it is defined.
  if [[ -n ${KUBE_EXTRA_GOPATH:-} ]]; then
    GOPATH=${GOPATH}:${KUBE_EXTRA_GOPATH}
  fi
  # Append the tree maintained by `godep` to the GOPATH unless KUBE_NO_GODEPS
  # is defined.
  if [[ -z ${KUBE_NO_GODEPS:-} ]]; then
    GOPATH="${GOPATH}:${KUBE_ROOT}/Godeps/_workspace"
  fi
  export GOPATH

  # Unset GOBIN in case it already exists in the current session.
  unset GOBIN
}


# kube::default_build_targets return list of all build targets
kube::default_build_targets() {
  echo "cmd/proxy"
  echo "cmd/apiserver"
  echo "cmd/controller-manager"
  echo "cmd/e2e"
  echo "cmd/kubelet"
  echo "cmd/kubecfg"
  echo "cmd/kubectl"
  echo "plugin/cmd/scheduler"
}

# kube::binaries_from_targets take a list of build targets and return the
# full go package to be built
kube::binaries_from_targets() {
  local target
  for target; do
    echo "${KUBE_GO_PACKAGE}/${target}"
  done
}
# --- Environment Variables ---

# Make ${KUBE_ROOT} an absolute path.
KUBE_ROOT=$(
  set -o errexit
  set -o nounset
  set -o pipefail
  unset CDPATH
  kube_root=$(dirname "${BASH_SOURCE}")/..
  cd "${kube_root}"
  pwd
)
export KUBE_ROOT

KUBE_TARGET="${KUBE_ROOT}/_output/go"
mkdir -p "${KUBE_TARGET}"
export KUBE_TARGET

KUBE_GO_PACKAGE=github.com/GoogleCloudPlatform/kubernetes
export KUBE_GO_PACKAGE

(
  # Create symlink named ${KUBE_GO_PACKAGE} under _output/go/src.
  # So that Go knows how to import Kubernetes sources by full path.
  # Use a subshell to avoid leaking these variables.

  set -o errexit
  set -o nounset
  set -o pipefail

  go_pkg_dir="${KUBE_TARGET}/src/${KUBE_GO_PACKAGE}"
  go_pkg_basedir=$(dirname "${go_pkg_dir}")
  mkdir -p "${go_pkg_basedir}"
  rm -f "${go_pkg_dir}"
  # TODO: This symlink should be relative.
  ln -s "${KUBE_ROOT}" "${go_pkg_dir}"
)

