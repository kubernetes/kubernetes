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

# -----------------------------------------------------------------------------
# Version management helpers.  These functions help to set, save and load the
# following variables:
#
#    KUBE_GIT_COMMIT - The git commit id corresponding to this
#          source code.
#    KUBE_GIT_TREE_STATE - "clean" indicates no changes since the git commit id
#        "dirty" indicates source code changes after the git commit id

# Grovels through git to set a set of env variables.
#
# If KUBE_GIT_VERSION_FILE, this function will load from that file instead of
# querying git.
kube::version::get_version_vars() {
  if [[ -n ${KUBE_GIT_VERSION_FILE-} ]]; then
    kube::version::load_version_vars "${KUBE_GIT_VERSION_FILE}"
    return
  fi

  local git=(git --work-tree "${KUBE_ROOT}")

  if [[ -n ${KUBE_GIT_COMMIT-} ]] || KUBE_GIT_COMMIT=$("${git[@]}" rev-parse "HEAD^{commit}" 2>/dev/null); then
    if [[ -z ${KUBE_GIT_TREE_STATE-} ]]; then
      # Check if the tree is dirty.  default to dirty
      if git_status=$("${git[@]}" status --porcelain 2>/dev/null) && [[ -z ${git_status} ]]; then
        KUBE_GIT_TREE_STATE="clean"
      else
        KUBE_GIT_TREE_STATE="dirty"
      fi
    fi

  fi
}

# Saves the environment flags to $1
kube::version::save_version_vars() {
  local version_file=${1-}
  [[ -n ${version_file} ]] || {
    echo "!!! Internal error.  No file specified in kube::version::save_version_vars"
    return 1
  }

  cat <<EOF >"${version_file}"
KUBE_GIT_COMMIT='${KUBE_GIT_COMMIT-}'
KUBE_GIT_TREE_STATE='${KUBE_GIT_TREE_STATE-}'
EOF
}

# Loads up the version variables from file $1
kube::version::load_version_vars() {
  local version_file=${1-}
  [[ -n ${version_file} ]] || {
    echo "!!! Internal error.  No file specified in kube::version::load_version_vars"
    return 1
  }

  source "${version_file}"
}

# Prints the value that needs to be passed to the -ldflags parameter of go build
# in order to set the Kubernetes based on the git tree status.
kube::version::ldflags() {
  kube::version::get_version_vars

  local -a ldflags=()
  if [[ -n ${KUBE_GIT_COMMIT-} ]]; then
    ldflags+=(-X "${KUBE_GO_PACKAGE}/pkg/version.gitCommit" "${KUBE_GIT_COMMIT}")
    ldflags+=(-X "${KUBE_GO_PACKAGE}/pkg/version.gitTreeState" "${KUBE_GIT_TREE_STATE}")
  fi

  # The -ldflags parameter takes a single string, so join the output.
  echo "${ldflags[*]-}"
}
