#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# This script automates the release processes for all pre-releases and official
# releases for Kubernetes.  See docs/devel/releasing.md for more info.

set -o errexit
set -o nounset
set -o pipefail

# TODO Audit echos to make sure they're all consistent.

# Sets global DRY_RUN
function main() {
  # Parse arguments
  if [[ "$#" -ne 2 && "$#" -ne 3 ]]; then
    usage
    exit 1
  fi
  local -r new_version=${1-}
  # TODO(ihmccreery) Stop calling it githash; it's not a githash.
  local -r githash=${2-}
  DRY_RUN=true
  if [[ "${3-}" == "--no-dry-run" ]]; then
    DRY_RUN=false
  else
    echo "THIS IS A DRY RUN"
  fi

  # Get and verify version info
  local -r alpha_version_regex="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.0-alpha\\.([1-9][0-9]*)$"
  local -r official_version_regex="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"
  local -r series_version_regex="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"
  if [[ "${new_version}" =~ $alpha_version_regex ]]; then
    local -r release_type='alpha'
    local -r version_major="${BASH_REMATCH[1]}"
    local -r version_minor="${BASH_REMATCH[2]}"
    local -r version_alpha_rev="${BASH_REMATCH[3]}"
  elif [[ "${new_version}" =~ $official_version_regex ]]; then
    local -r release_type='official'
    local -r version_major="${BASH_REMATCH[1]}"
    local -r version_minor="${BASH_REMATCH[2]}"
    local -r version_patch="${BASH_REMATCH[3]}"
  elif [[ "${new_version}" =~ $series_version_regex ]]; then
    local -r release_type='series'
    local -r version_major="${BASH_REMATCH[1]}"
    local -r version_minor="${BASH_REMATCH[2]}"
  else
    usage
    echo
    echo "!!! You specified an invalid version '${new_version}'."
    exit 1
  fi

  # Get the git commit from the githash and verify it
  local -r git_commit_regex="^[0-9a-f]{7}$"
  local -r git_commit=$(echo "${githash}" | awk -F'+' '{print $2}' | head -c7)
  if ! [[ "${git_commit}" =~ $git_commit_regex ]]; then
    usage
    echo
    echo "!!! You specified an invalid githash '${githash}'."
    echo "!!! Tried to extract commit, got ${git_commit}."
    exit 1
  fi
  echo "Doing ${release_type} release '${new_version}'."

  # Set the default umask for the release. This ensures consistency
  # across our release builds.
  #
  # TODO(ihmccreery): This should be in our build process, not our release
  # process.
  local -r release_umask=${release_umask:-022}
  umask "${release_umask}"

  local -r github="https://github.com/kubernetes/kubernetes.git"
  local -r dir="/tmp/kubernetes-${release_type}-release-${new_version}-$(date +%s)"
  echo "Cloning from '${github}'..."
  git clone "${github}" "${dir}"

  # !!! REMINDER !!!
  #
  # Past this point, you are dealing with a different clone of the repo at some
  # version. Don't assume you're executing code from the same repo as this script
  # is running in. This is a version agnostic process.
  pushd "${dir}"

  if [[ "${release_type}" == 'alpha' ]]; then
    git checkout "${git_commit}"
    verify-at-git-commit "${git_commit}"
    verify-ancestor "v${version_major}.${version_minor}.0-alpha.$((${version_alpha_rev}-1))"

    alpha-release "${new_version}"
  elif [[ "${release_type}" == 'official' ]]; then
    local -r release_branch="release-${version_major}.${version_minor}"
    local -r beta_version="v${version_major}.${version_minor}.$((${version_patch}+1))-beta"

    git checkout "${release_branch}"
    verify-at-git-commit "${git_commit}"
    verify-ancestor "${new_version}-beta"

    official-release "${new_version}"
    beta-release "${beta_version}"
  else # [[ "${release_type}" == 'series' ]]
    local -r release_branch="release-${version_major}.${version_minor}"
    local -r alpha_version="v${version_major}.$((${version_minor}+1)).0-alpha.0"
    local -r beta_version="v${version_major}.${version_minor}.0-beta"

    git checkout "${git_commit}"
    verify-at-git-commit "${git_commit}"
    # NOTE: We check the second alpha version, ...-alpha.1, because ...-alpha.0
    # is the branch point for the previous release cycle, so could provide a
    # false positive if we accidentally try to release off of the old release
    # branch.
    verify-ancestor "v${version_major}.${version_minor}.0-alpha.1"

    alpha-release "${alpha_version}"

    echo "Branching ${release_branch}."
    git checkout -b "${release_branch}"

    beta-release "${beta_version}"
  fi

  cleanup "${dir}"
}

function usage() {
  echo "Usage: ${0} <version> <githash> [--no-dry-run]"
  echo
  echo "See docs/devel/releasing.md for more info."
}

function alpha-release() {
  local -r alpha_version="${1}"

  echo "Doing an alpha release of ${alpha_version}."

  echo "Tagging ${alpha_version} at $(current-git-commit)."
  git tag -a -m "Kubernetes pre-release ${alpha_version}" "${alpha_version}"
  git-push "${alpha_version}"
  build
  # TODO prompt for GitHub bits
  echo "FAKE prompt GitHub bits for ${alpha_version}"
}

function beta-release() {
  local -r beta_version="${1}"

  echo "Doing a beta release of ${beta_version}."

  versionize-docs-and-commit "${beta_version}"
  rev-version-and-commit

  echo "Tagging ${beta_version} at $(current-git-commit)."
  git tag -a -m "Kubernetes pre-release ${beta_version}" "${beta_version}"
  git-push "${beta_version}"
  build
  # TODO prompt for GitHub bits
  echo "FAKE prompt GitHub bits for ${beta_version}"
}

function official-release() {
  local -r official_version="${1}"

  echo "Doing an official release of ${official_version}."

  versionize-docs-and-commit "${official_version}"
  rev-version-and-commit

  echo "Tagging ${official_version} at $(current-git-commit)."
  git tag -a -m "Kubernetes release ${official_version}" "${official_version}"
  git-push "${official_version}"
  build
  # TODO prompt for GitHub bits
  echo "FAKE prompt GitHub bits for ${official_version}"
}

function verify-at-git-commit() {
  local -r git_commit="${1}"
  echo "Verifying we are at ${git_commit}."
  if [[ $(current-git-commit) != ${git_commit} ]]; then
    echo "!!! We are not at commit ${git_commit}!"
    cleanup "${dir}"
    exit 1
  fi
}

function current-git-commit() {
  git rev-parse --short HEAD
}

function verify-ancestor() {
  local -r ancestor="${1}"
  # Check to make sure the/a previous version is an ancestor.
  echo "Checking that previous version '${ancestor}' is an ancestor."
  if ! git merge-base --is-ancestor "${ancestor}" HEAD; then
    echo "!!! Previous version '${ancestor}' is not an ancestor!"
    cleanup "${dir}"
    exit 1
  fi
}

function git-push() {
  if ${DRY_RUN}; then
    echo "FAKE git push "$@""
  else
    echo "OH NO!!! YOU'RE NOT REALLY IN DRY_RUN! git push "$@""
    echo "You don't really want to push, do you?"
    # git push "$@"
  fi
}

function versionize-docs-and-commit() {
  local -r version="${1}"
  echo "Versionizing docs and committing."
  # NOTE: This is using versionize-docs.sh at the release point.
  ./build/versionize-docs.sh ${version}
  git commit -am "Versioning docs and examples for ${version}."
}

# TODO(ihmccreery): Review and fix this function.
function rev-version-and-commit() {
  echo "FAKE rev-version-and-commit"
  # SED=sed
  # if which gsed &>/dev/null; then
  #   SED=gsed
  # fi
  # if ! ($SED --version 2>&1 | grep -q GNU); then
  #   echo "!!! GNU sed is required.  If on OS X, use 'brew install gnu-sed'."
  #   cleanup
  #   exit 1
  # fi

  # VERSION_FILE="./pkg/version/base.go"

  # GIT_MINOR="${version_minor}.${VERSION_PATCH}"
  # echo "+++ Updating to ${NEW_VERSION}"
  # $SED -ri -e "s/gitMajor\s+string = \"[^\"]*\"/gitMajor string = \"${version_major}\"/" "${VERSION_FILE}"
  # $SED -ri -e "s/gitMinor\s+string = \"[^\"]*\"/gitMinor string = \"${GIT_MINOR}\"/" "${VERSION_FILE}"
  # $SED -ri -e "s/gitVersion\s+string = \"[^\"]*\"/gitVersion string = \"$NEW_VERSION-${release_branch}+\$Format:%h\$\"/" "${VERSION_FILE}"
  # gofmt -s -w "${VERSION_FILE}"

  # echo "+++ Committing version change"
  # git add "${VERSION_FILE}"
  # git commit -m "Kubernetes version ${NEW_VERSION}"
}

# TODO What's in this step?
function build() {
  echo "FAKE build"
  # TODO Do we need to publish the build to GCS?
}

function cleanup() {
  local -r dir="${1}"
  if ${DRY_RUN}; then
    echo "Dry run:"
    echo "  pushd ${dir}"
    echo "to have a look around."
  else
    popd
  fi
}

main "$@"

# echo ""
# echo "Success you must now:"
# echo ""
# echo "- Push the tags:"
# echo "   git push ${push_url} ${NEW_VERSION}"
# echo "   git push ${push_url} ${beta_ver}"
# 
# if [[ "${VERSION_PATCH}" == "0" ]]; then
#   echo "- Push the alpha tag:"
#   echo "   git push ${push_url} ${alpha_ver}"
#   echo "- Push the new release branch:"
#   echo "   git push ${push_url} ${current_branch}:${release_branch}"
#   echo "- DO NOTHING TO MASTER. You were done with master when you pushed the alpha tag."
# else
#   echo "- Send branch: ${current_branch} as a PR to ${release_branch} <-- NOTE THIS"
#   echo "- In the contents of the PR, include the PRs in the release:"
#   echo "    hack/cherry_pick_list.sh ${current_branch}^1"
#   echo "  This helps cross-link PRs to patch releases they're part of in GitHub."
#   echo "- Have someone review the PR. This is a mechanical review to ensure it contains"
#   echo "  the ${NEW_VERSION} commit, which was tagged at ${newtag}."
# fi
