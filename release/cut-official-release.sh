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

# Sets DIR, INSTRUCTIONS
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
    echo "!!! This NOT is a dry run."
    DRY_RUN=false
  else
    echo "This is a dry run."
  fi

  check-prereqs

  # Get and verify version info
  local -r alpha_version_regex="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.0-alpha\\.([1-9][0-9]*)$"
  local -r beta_version_regex="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)-beta\\.([1-9][0-9]*)$"
  local -r official_version_regex="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"
  local -r series_version_regex="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"
  if [[ "${new_version}" =~ $alpha_version_regex ]]; then
    local -r release_type='alpha'
    local -r version_major="${BASH_REMATCH[1]}"
    local -r version_minor="${BASH_REMATCH[2]}"
    local -r version_alpha_rev="${BASH_REMATCH[3]}"
  elif [[ "${new_version}" =~ $beta_version_regex ]]; then
    local -r release_type='beta'
    local -r version_major="${BASH_REMATCH[1]}"
    local -r version_minor="${BASH_REMATCH[2]}"
    local -r version_patch="${BASH_REMATCH[3]}"
    local -r version_beta_rev="${BASH_REMATCH[4]}"
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

  local -r github="git@github.com:kubernetes/kubernetes.git"
  declare -r DIR=$(mktemp -d "/tmp/kubernetes-${release_type}-release-${new_version}-XXXXXXX")

  # Start a tmp file that will hold instructions for the user.
  declare -r INSTRUCTIONS=$(mktemp "/tmp/kubernetes-${release_type}-release-${new_version}-instructions-XXXXXXX")
  if $DRY_RUN; then
    cat > "${INSTRUCTIONS}" <<- EOM
Success on dry run!  Do

  pushd ${DIR}

to see what happened.

You would now do the following, if not a dry run:

EOM
  else
    cat > "${INSTRUCTIONS}" <<- EOM
Success!  You must now do the following (you may want to cut and paste these
instructions elsewhere):

EOM
  fi

  echo "Cloning from '${github}'..."
  git clone "${github}" "${DIR}"

  # !!! REMINDER !!!
  #
  # Past this point, you are dealing with a different clone of the repo at some
  # version. Don't assume you're executing code from the same repo as this script
  # is running in. This is a version agnostic process.
  pushd "${DIR}"

  if [[ "${release_type}" == 'alpha' ]]; then
    local -r ancestor="v${version_major}.${version_minor}.0-alpha.$((${version_alpha_rev}-1))"

    git checkout "${git_commit}"
    verify-at-git-commit "${git_commit}"
    verify-ancestor "${ancestor}"

    alpha-release "${new_version}"
  elif [[ "${release_type}" == 'beta' ]]; then
    local -r release_branch="release-${version_major}.${version_minor}"
    local -r ancestor="v${version_major}.${version_minor}.${version_patch}-beta.$((${version_beta_rev}-1))"

    git checkout "${release_branch}"
    verify-at-git-commit "${git_commit}"
    verify-ancestor "${ancestor}"

    beta-release "${new_version}"

    git-push ${release_branch}
  elif [[ "${release_type}" == 'official' ]]; then
    local -r release_branch="release-${version_major}.${version_minor}"
    local -r beta_version="v${version_major}.${version_minor}.$((${version_patch}+1))-beta"
    local -r ancestor="${new_version}-beta"

    git checkout "${release_branch}"
    verify-at-git-commit "${git_commit}"
    verify-ancestor "${ancestor}"

    official-release "${new_version}"
    beta-release "${beta_version}"

    git-push ${release_branch}
  else # [[ "${release_type}" == 'series' ]]
    local -r release_branch="release-${version_major}.${version_minor}"
    local -r alpha_version="v${version_major}.$((${version_minor}+1)).0-alpha.0"
    local -r beta_version="v${version_major}.${version_minor}.0-beta"
    # NOTE: We check the second alpha version, ...-alpha.1, because ...-alpha.0
    # is the branch point for the previous release cycle, so could provide a
    # false positive if we accidentally try to release off of the old release
    # branch.
    local -r ancestor="v${version_major}.${version_minor}.0-alpha.1"

    git checkout "${git_commit}"
    verify-at-git-commit "${git_commit}"
    verify-ancestor "${ancestor}"

    alpha-release "${alpha_version}"

    echo "Branching ${release_branch}."
    git checkout -b "${release_branch}"
    versionize-docs-and-commit "${release_branch}"

    beta-release "${beta_version}"

    git-push ${release_branch}
  fi

  echo
  cat "${INSTRUCTIONS}"
}

function usage() {
  echo "Usage: ${0} <version> <githash> [--no-dry-run]"
  echo
  echo "See docs/devel/releasing.md for more info."
}

function check-prereqs() {
  SED=sed
  if which gsed &>/dev/null; then
    SED=gsed
  fi
  if ! ($SED --version 2>&1 | grep -q GNU); then
    echo "!!! GNU sed is required.  If on OS X, use 'brew install gnu-sed'."
    exit 1
  fi
}

function alpha-release() {
  local -r alpha_version="${1}"

  echo "Doing an alpha release of ${alpha_version}."

  echo "Tagging ${alpha_version} at $(current-git-commit)."
  git tag -a -m "Kubernetes pre-release ${alpha_version}" "${alpha_version}"
  git-push "${alpha_version}"

  cat >> "${INSTRUCTIONS}" <<- EOM
- Finish the ${alpha_version} release build:
  - From this directory (clone of upstream/master),
      ./release/build-official-release.sh ${alpha_version}
  - Prep release notes:
    - Figure out what the PR numbers for this release and last release are, and
      get an api-token from GitHub (https://github.com/settings/tokens).  From a
      clone of kubernetes/contrib at upstream/master,
        go run release-notes/release-notes.go --last-release-pr=<number> --current-release-pr=<number> --api-token=<token>
      Feel free to prune.
EOM
}

function beta-release() {
  local -r beta_version="${1}"

  echo "Doing a beta release of ${beta_version}."

  rev-version-and-commit "${beta_version}"

  echo "Tagging ${beta_version} at $(current-git-commit)."
  git tag -a -m "Kubernetes pre-release ${beta_version}" "${beta_version}"
  git-push "${beta_version}"

  # NOTE: We currently don't publish beta release notes, since they'll go out
  # with the official release, so we don't prompt for compiling them here.
  cat >> "${INSTRUCTIONS}" <<- EOM
- Finish the ${beta_version} release build:
  - From this directory (clone of upstream/master),
      ./release/build-official-release.sh ${beta_version}
EOM
}

function official-release() {
  local -r official_version="${1}"

  echo "Doing an official release of ${official_version}."

  rev-version-and-commit "${official_version}"

  echo "Tagging ${official_version} at $(current-git-commit)."
  git tag -a -m "Kubernetes release ${official_version}" "${official_version}"
  git-push "${official_version}"

  cat >> "${INSTRUCTIONS}" <<- EOM
- Finish the ${official_version} release build:
  - From this directory (clone of upstream/master),
      ./release/build-official-release.sh ${official_version}
  - Prep release notes:
    - From this directory (clone of upstream/master), run
        ./hack/cherry_pick_list.sh ${official_version}
      to get the release notes for the patch release you just created. Feel
      free to prune anything internal, but typically for patch releases we tend
      to include everything in the release notes.
    - If this is a first official release (vX.Y.0), scan through the release
      notes for all of the alpha releases since the last cycle, and include
      anything important in release notes.
EOM
}

function verify-at-git-commit() {
  local -r git_commit="${1}"
  echo "Verifying we are at ${git_commit}."
  if [[ $(current-git-commit) != ${git_commit} ]]; then
    cat <<- EOM
!!! We are not at commit ${git_commit}! (If you're cutting a beta or official
release, that probably means your release branch isn't frozen, so the commit
you want to release isn't at HEAD of the release branch.)"
EOM
    exit 1
  fi
}

function current-git-commit() {
  git rev-parse --short HEAD
}

function verify-ancestor() {
  local -r ancestor="${1}"
  # Check to make sure the previous version is an ancestor.
  echo "Checking that previous version '${ancestor}' is an ancestor."
  if ! git merge-base --is-ancestor "${ancestor}" HEAD; then
    echo "!!! Previous version '${ancestor}' is not an ancestor!"
    exit 1
  fi
}

function versionize-docs-and-commit() {
  local -r release_branch="${1}"
  echo "Versionizing docs for ${release_branch} and committing."
  # NOTE: This is using versionize-docs.sh at the release point.
  ./build/versionize-docs.sh ${release_branch}
  git commit -am "Versioning docs and examples for ${release_branch}."
}

function rev-version-and-commit() {
  local -r version="${1}"
  local -r version_file="pkg/version/base.go"

  local -r version_regex="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(-beta\\.(0|[1-9][0-9]*))?$"
  if [[ "${version}" =~ $version_regex ]]; then
    local -r version_major="${BASH_REMATCH[1]}"
    # We append a '+' to the minor version on a beta build per hack/lib/version.sh's logic.
    if [[ -z "${BASH_REMATCH[4]}" ]]; then
      local -r version_minor="${BASH_REMATCH[2]}"
    else
      local -r version_minor="${BASH_REMATCH[2]}+"
    fi
  else
    echo "!!! Something went wrong.  Tried to rev version to invalid version; should not have gotten to this point."
    exit 1
  fi

  echo "Updating ${version_file} to ${version}"
  $SED -ri -e "s/gitMajor\s+string = \"[^\"]*\"/gitMajor string = \"${version_major}\"/" "${version_file}"
  $SED -ri -e "s/gitMinor\s+string = \"[^\"]*\"/gitMinor string = \"${version_minor}\"/" "${version_file}"
  $SED -ri -e "s/gitVersion\s+string = \"[^\"]*\"/gitVersion string = \"${version}+\$Format:%h\$\"/" "${version_file}"
  gofmt -s -w "${version_file}"

  echo "Committing version change ${version}"
  git add "${version_file}"
  git commit -m "Kubernetes version ${version}"
}

function git-push() {
  local -r object="${1}"
  if $DRY_RUN; then
    echo "Dry run: would have done git push origin ${object}"
  else
    git push origin "${object}"
  fi
}

main "$@"
