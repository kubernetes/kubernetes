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
#    LMKTFY_GIT_COMMIT - The git commit id corresponding to this
#          source code.
#    LMKTFY_GIT_TREE_STATE - "clean" indicates no changes since the git commit id
#        "dirty" indicates source code changes after the git commit id
#    LMKTFY_GIT_VERSION - "vX.Y" used to indicate the last release version.
#    LMKTFY_GIT_MAJOR - The major part of the version
#    LMKTFY_GIT_MINOR - The minor component of the version

# Grovels through git to set a set of env variables.
#
# If LMKTFY_GIT_VERSION_FILE, this function will load from that file instead of
# querying git.
lmktfy::version::get_version_vars() {
  if [[ -n ${LMKTFY_GIT_VERSION_FILE-} ]]; then
    lmktfy::version::load_version_vars "${LMKTFY_GIT_VERSION_FILE}"
    return
  fi

  local git=(git --work-tree "${LMKTFY_ROOT}")

  if [[ -n ${LMKTFY_GIT_COMMIT-} ]] || LMKTFY_GIT_COMMIT=$("${git[@]}" rev-parse "HEAD^{commit}" 2>/dev/null); then
    if [[ -z ${LMKTFY_GIT_TREE_STATE-} ]]; then
      # Check if the tree is dirty.  default to dirty
      if git_status=$("${git[@]}" status --porcelain 2>/dev/null) && [[ -z ${git_status} ]]; then
        LMKTFY_GIT_TREE_STATE="clean"
      else
        LMKTFY_GIT_TREE_STATE="dirty"
      fi
    fi

    # Use git describe to find the version based on annotated tags.
    if [[ -n ${LMKTFY_GIT_VERSION-} ]] || LMKTFY_GIT_VERSION=$("${git[@]}" describe --tags --abbrev=14 "${LMKTFY_GIT_COMMIT}^{commit}" 2>/dev/null); then
      if [[ "${LMKTFY_GIT_TREE_STATE}" == "dirty" ]]; then
        # git describe --dirty only considers changes to existing files, but
        # that is problematic since new untracked .go files affect the build,
        # so use our idea of "dirty" from git status instead.
        LMKTFY_GIT_VERSION+="-dirty"
      fi

      # Try to match the "git describe" output to a regex to try to extract
      # the "major" and "minor" versions and whether this is the exact tagged
      # version or whether the tree is between two tagged versions.
      if [[ "${LMKTFY_GIT_VERSION}" =~ ^v([0-9]+)\.([0-9]+)(\.[0-9]+)?([-].*)?$ ]]; then
        LMKTFY_GIT_MAJOR=${BASH_REMATCH[1]}
        LMKTFY_GIT_MINOR=${BASH_REMATCH[2]}
        if [[ -n "${BASH_REMATCH[4]}" ]]; then
          LMKTFY_GIT_MINOR+="+"
        fi
      fi
    fi
  fi
}

# Saves the environment flags to $1
lmktfy::version::save_version_vars() {
  local version_file=${1-}
  [[ -n ${version_file} ]] || {
    echo "!!! Internal error.  No file specified in lmktfy::version::save_version_vars"
    return 1
  }

  cat <<EOF >"${version_file}"
LMKTFY_GIT_COMMIT='${LMKTFY_GIT_COMMIT-}'
LMKTFY_GIT_TREE_STATE='${LMKTFY_GIT_TREE_STATE-}'
LMKTFY_GIT_VERSION='${LMKTFY_GIT_VERSION-}'
LMKTFY_GIT_MAJOR='${LMKTFY_GIT_MAJOR-}'
LMKTFY_GIT_MINOR='${LMKTFY_GIT_MINOR-}'
EOF
}

# Loads up the version variables from file $1
lmktfy::version::load_version_vars() {
  local version_file=${1-}
  [[ -n ${version_file} ]] || {
    echo "!!! Internal error.  No file specified in lmktfy::version::load_version_vars"
    return 1
  }

  source "${version_file}"
}

# Prints the value that needs to be passed to the -ldflags parameter of go build
# in order to set the LMKTFY based on the git tree status.
lmktfy::version::ldflags() {
  lmktfy::version::get_version_vars

  local -a ldflags=()
  if [[ -n ${LMKTFY_GIT_COMMIT-} ]]; then
    ldflags+=(-X "${LMKTFY_GO_PACKAGE}/pkg/version.gitCommit" "${LMKTFY_GIT_COMMIT}")
    ldflags+=(-X "${LMKTFY_GO_PACKAGE}/pkg/version.gitTreeState" "${LMKTFY_GIT_TREE_STATE}")
  fi

  if [[ -n ${LMKTFY_GIT_VERSION-} ]]; then
    ldflags+=(-X "${LMKTFY_GO_PACKAGE}/pkg/version.gitVersion" "${LMKTFY_GIT_VERSION}")
  fi

  if [[ -n ${LMKTFY_GIT_MAJOR-} && -n ${LMKTFY_GIT_MINOR-} ]]; then
    ldflags+=(
      -X "${LMKTFY_GO_PACKAGE}/pkg/version.gitMajor" "${LMKTFY_GIT_MAJOR}"
      -X "${LMKTFY_GO_PACKAGE}/pkg/version.gitMinor" "${LMKTFY_GIT_MINOR}"
    )
  fi

  # The -ldflags parameter takes a single string, so join the output.
  echo "${ldflags[*]-}"
}
