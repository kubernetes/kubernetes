#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# TODO What does this script do?

set -o errexit
set -o nounset
set -o pipefail

# TODO Audit echos to make sure they're all consistent.
# TODO Refactor to remove globals?

function main() {
  if [ "$#" -gt 3 ]; then
    usage
    exit 1
  fi

  declare -r NEW_VERSION=${1-}
  # TODO(ihmccreery) Stop calling it githash; it's not a githash.
  declare -r GITHASH=${2-}
  DRY_RUN=true
  if [[ "${3-}" == "--no-dry-run" ]]; then
    DRY_RUN=false
  else
    echo "!!! THIS IS A DRY RUN"
  fi

  declare -r ALPHA_VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.0-alpha\\.([1-9][0-9]*)$"
  declare -r OFFICIAL_VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"
  declare -r SERIES_VERSION_REGEX="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"
  if [[ "${NEW_VERSION}" =~ $ALPHA_VERSION_REGEX ]]; then
    RELEASE_TYPE='alpha'
    VERSION_MAJOR="${BASH_REMATCH[1]}"
    VERSION_MINOR="${BASH_REMATCH[2]}"
    VERSION_ALPHA_REV="${BASH_REMATCH[3]}"
    ANCESTOR="v${VERSION_MAJOR}.${VERSION_MINOR}.0-alpha.$((VERSION_ALPHA_REV-1))"
  elif [[ "${NEW_VERSION}" =~ $OFFICIAL_VERSION_REGEX ]]; then
    RELEASE_TYPE='official'
    VERSION_MAJOR="${BASH_REMATCH[1]}"
    VERSION_MINOR="${BASH_REMATCH[2]}"
    VERSION_PATCH="${BASH_REMATCH[3]}"
    ANCESTOR="${NEW_VERSION}-beta"
    RELEASE_BRANCH="release-${VERSION_MAJOR}.${VERSION_MINOR}"
  elif [[ "${NEW_VERSION}" =~ $SERIES_VERSION_REGEX ]]; then
    RELEASE_TYPE='series'
    VERSION_MAJOR="${BASH_REMATCH[1]}"
    VERSION_MINOR="${BASH_REMATCH[2]}"
    # NOTE: We check the second alpha version, ...-alpha.1, because ...-alpha.0
    # is the branch point for the previous release cycle, so could provide a
    # false positive if we accidentally try to release off of the old release
    # branch.
    ANCESTOR="v${VERSION_MAJOR}.${VERSION_MINOR}.0-alpha.1"
    RELEASE_BRANCH="release-${VERSION_MAJOR}.${VERSION_MINOR}"
  else
    usage
    echo
    echo "!!! You specified an invalid version '${NEW_VERSION}'."
    exit 1
  fi

  # Get the git commit from the githash and verify it
  declare -r GIT_COMMIT_REGEX="^[0-9a-f]{7}$"
  declare -r GIT_COMMIT=$(echo "${GITHASH}" | awk -F'+' '{print $2}' | head -c7)
  if ! [[ "${GIT_COMMIT}" =~ $GIT_COMMIT_REGEX ]]; then
    usage
    echo
    echo "!!! You specified an invalid githash '${GITHASH}'."
    echo "!!! Tried to extract commit, got ${GIT_COMMIT}."
    exit 1
  fi
  echo "Doing ${RELEASE_TYPE} release '${NEW_VERSION}'."

  # Set the default umask for the release. This ensures consistency
  # across our release builds.
  #
  # TODO(ihmccreery): This should be in our build process, not our release
  # process.
  declare -r RELEASE_UMASK=${RELEASE_UMASK:-022}
  umask "${RELEASE_UMASK}"

  declare -r GITHUB="https://github.com/kubernetes/kubernetes.git"
  declare -r DIR="/tmp/kubernetes-${RELEASE_TYPE}-release-${NEW_VERSION}-$(date +%s)"
  echo "Cloning from '${GITHUB}'..."
  git clone "${GITHUB}" "${DIR}"

  # !!! REMINDER !!!
  #
  # Past this point, you are dealing with a different clone of the repo at some
  # version. Don't assume you're executing code from the same repo as this script
  # is running in. This is a version agnostic process.
  pushd "${DIR}"

  if [[ "${RELEASE_TYPE}" == 'alpha' ]]; then
    git checkout "${GIT_COMMIT}"
    verify-at-git-commit
    verify-ancestor

    alpha-release "${NEW_VERSION}"
  elif [[ "${RELEASE_TYPE}" == 'official' ]]; then
    declare -r RELEASE_BRANCH="release-${VERSION_MAJOR}.${VERSION_MINOR}"
    declare -r BETA_VERSION="v${VERSION_MAJOR}.${VERSION_MINOR}.$((${VERSION_PATCH}+1))-beta"

    git checkout "${RELEASE_BRANCH}"
    verify-at-git-commit
    # TODO uncomment this once we've pushed v1.1.1-beta
    #verify-ancestor

    official-release "${NEW_VERSION}"
    beta-release "${BETA_VERSION}"
  else # [[ "${RELEASE_TYPE}" == 'series' ]]
    declare -r RELEASE_BRANCH="release-${VERSION_MAJOR}.${VERSION_MINOR}"
    declare -r ALPHA_VERSION="v${VERSION_MAJOR}.$((${VERSION_MINOR}+1)).0-alpha.0"
    declare -r BETA_VERSION="v${VERSION_MAJOR}.${VERSION_MINOR}.0-beta"

    git checkout "${GIT_COMMIT}"
    verify-at-git-commit
    verify-ancestor

    # TODO (Fix versioning.md if you don't do this.)  We maybe could actually do
    # the alpha rev (in a series release) at HEAD, and patch the version/base.go
    # logic then.  We'd then have some part of the tree between the branch of
    # vX.Y series and the vX.(Y+1).0-alpha.0 tag, but I don't think that's a
    # problem.
    alpha-release "${ALPHA_VERSION}"

    echo "Branching ${RELEASE_BRANCH}."
    git checkout -b "${RELEASE_BRANCH}"

    beta-release "${BETA_VERSION}"
  fi

  cleanup
}

function usage() {
  echo "Usage: ${0} <version> <githash> [--no-dry-run]"
  echo
  echo "See docs/devel/releasing.md for more info."
}

function alpha-release() {
  local -r alpha_version="${1}"

  echo "Doing an alpha release of ${alpha_version}."

  echo "Tagging ${alpha_version} at $(git rev-parse --short HEAD)."
  git tag -a -m "Kubernetes pre-release ${alpha_version}" "${alpha_version}"
  git-push "${alpha_version}"
  build
  # TODO prompt for GitHub bits
  echo "FAKE prompt GitHub bits for ${alpha_version}"
}

function beta-release() {
  local -r beta_version="${1}"

  echo "Doing a beta release of ${beta_version}."

  # TODO We need to rev something so that we have a separate commit on the
  # release-X.Y branch, since we don't want master to pick up with beta tag.
  # TODO rev-docs and/or version?
  versionize-docs-and-commit
  rev-version-and-commit

  echo "Tagging ${beta_version} at $(git rev-parse --short HEAD)."
  git tag -a -m "Kubernetes pre-release ${beta_version}" "${beta_version}"
  git-push "${beta_version}"
  build
  # TODO prompt for GitHub bits
  echo "FAKE prompt GitHub bits for ${beta_version}"
}

function official-release() {
  local -r official_version="${1}"

  echo "Doing an official release of ${official_version}."

  # TODO rev-docs and/or version?
  versionize-docs-and-commit
  rev-version-and-commit

  echo "Tagging ${official_version} at $(git rev-parse --short HEAD)."
  git tag -a -m "Kubernetes release ${official_version}" "${official_version}"
  git-push "${official_version}"
  build
  # TODO prompt for GitHub bits
  echo "FAKE prompt GitHub bits for ${official_version}"
}

function verify-at-git-commit() {
  echo "Verifying we are at ${GIT_COMMIT}."
  if [[ $(git rev-parse --short HEAD) != ${GIT_COMMIT} ]]; then
    echo "!!! We are not at commit ${GIT_COMMIT}!"
    cleanup
    exit 1
  fi
}

function verify-ancestor() {
  # Check to make sure the/a previous version is an ancestor.
  echo "Checking that previous version '${ANCESTOR}' is an ancestor."
  if ! git merge-base --is-ancestor "${ANCESTOR}" HEAD; then
    echo "!!! Previous version '${ANCESTOR}' is not an ancestor!"
    cleanup
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

# TODO(ihmccreery): Review and fix this function.
function versionize-docs-and-commit() {
  echo "FAKE versionize-docs-and-commit"
  # # NOTE: This is using versionize-docs.sh at the release point.
  # ./build/versionize-docs.sh ${NEW_VERSION}
  # git commit -am "Versioning docs and examples for ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}"
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

  # GIT_MINOR="${VERSION_MINOR}.${VERSION_PATCH}"
  # echo "+++ Updating to ${NEW_VERSION}"
  # $SED -ri -e "s/gitMajor\s+string = \"[^\"]*\"/gitMajor string = \"${VERSION_MAJOR}\"/" "${VERSION_FILE}"
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
  if ${DRY_RUN}; then
    echo "Dry run:"
    echo "  pushd ${DIR}"
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
