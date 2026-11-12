#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors.
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
#        "archive" indicates the tree was produced by 'git archive'
#    KUBE_GIT_VERSION - "vX.Y" used to indicate the last release version.
#    KUBE_GIT_MAJOR - The major part of DefaultKubeBinaryVersion
#    KUBE_GIT_MINOR - The minor part of DefaultKubeBinaryVersion

# Derives KUBE_GIT_MAJOR and KUBE_GIT_MINOR from DefaultKubeBinaryVersion, the
# same way as k8s.io/component-base/version does.
kube::version::set_major_minor() {
  local base_go="${KUBE_ROOT}/staging/src/k8s.io/component-base/version/base.go"
  local default_version
  default_version=$(sed -n 's/^[[:space:]]*DefaultKubeBinaryVersion = "\([0-9]*\)\.\([0-9]*\)"$/\1.\2/p' "${base_go}")
  if [[ -z "${default_version}" ]]; then
    kube::log::error "Unable to extract DefaultKubeBinaryVersion from ${base_go}"
    exit 1
  fi
  KUBE_GIT_MAJOR="${default_version%%.*}"
  KUBE_GIT_MINOR="${default_version#*.}"
}

# Prints the version of the source code (vX.Y.Z[-pre]) as defined by the
# constants in k8s.io/component-base/version.
kube::version::source_version() {
  local base_go="${KUBE_ROOT}/staging/src/k8s.io/component-base/version/base.go"
  local patch pre
  patch=$(sed -n 's/^[[:space:]]*KubePatchVersion = \([0-9]*\)$/\1/p' "${base_go}")
  pre=$(sed -n 's/^[[:space:]]*KubePreReleaseVersion = "\(.*\)"$/\1/p' "${base_go}")
  if [[ -z "${patch}" ]]; then
    kube::log::error "Unable to extract KubePatchVersion from ${base_go}"
    exit 1
  fi
  kube::version::set_major_minor
  echo "v${KUBE_GIT_MAJOR}.${KUBE_GIT_MINOR}.${patch}${pre:+-${pre}}"
}

# Grovels through git to set a set of env variables.
#
# If KUBE_GIT_VERSION_FILE, this function will load from that file instead of
# querying git. The values from the file are also injected into the binaries
# via ldflags (see kube::version::ldflags), overriding what Go records.
#
# TODO: reconsider whether KUBE_GIT_VERSION_FILE is needed. Removing it would
# simplify the source code because ldflags injection would no longer be
# necessary for anything except buildDate.
kube::version::get_version_vars() {
  if [[ -n ${KUBE_GIT_VERSION_FILE-} ]]; then
    kube::version::load_version_vars "${KUBE_GIT_VERSION_FILE}"
    return
  fi

  # If the kubernetes source was exported through git archive, then
  # we likely don't have a git tree, but these magic values may be filled in.
  # shellcheck disable=SC2016,SC2050
  # Disabled as we're not expanding these at runtime, but rather expecting
  # that another tool may have expanded these and rewritten the source (!)
  if [[ '$Format:%%$' == "%" ]]; then
    KUBE_GIT_COMMIT='$Format:%H$'
    KUBE_GIT_TREE_STATE="archive"
    # When a 'git archive' is exported, the '$Format:%D$' below will look
    # something like 'HEAD -> release-1.8, tag: v1.8.3' where then 'tag: '
    # can be extracted from it.
    if [[ '$Format:%D$' =~ tag:\ (v[^ ,]+) ]]; then
     KUBE_GIT_VERSION="${BASH_REMATCH[1]}"
    fi
  fi

  kube::version::set_major_minor

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

    # Use git describe to find the version based on tags.
    if [[ -n ${KUBE_GIT_VERSION-} ]] || KUBE_GIT_VERSION=$("${git[@]}" describe --tags --match='v*' --abbrev=14 "${KUBE_GIT_COMMIT}^{commit}" 2>/dev/null); then
      # This translates the "git describe" to an actual semver.org
      # compatible semantic version that looks something like this:
      #   v1.1.0-alpha.0.6+84c76d1142ea4d
      #
      # TODO: We continue calling this "git version" because so many
      # downstream consumers are expecting it there.
      #
      # These regexes are painful enough in sed...
      # We don't want to do them in pure shell, so disable SC2001
      # shellcheck disable=SC2001
      DASHES_IN_VERSION=$(echo "${KUBE_GIT_VERSION}" | sed "s/[^-]//g")
      if [[ "${DASHES_IN_VERSION}" == "---" ]] ; then
        # shellcheck disable=SC2001
        # We have distance to subversion (v1.1.0-subversion-1-gCommitHash)
        KUBE_GIT_VERSION=$(echo "${KUBE_GIT_VERSION}" | sed "s/-\([0-9]\{1,\}\)-g\([0-9a-f]\{14\}\)$/.\1\+\2/")
      elif [[ "${DASHES_IN_VERSION}" == "--" ]] ; then
        # shellcheck disable=SC2001
        # We have distance to base tag (v1.1.0-1-gCommitHash)
        KUBE_GIT_VERSION=$(echo "${KUBE_GIT_VERSION}" | sed "s/-g\([0-9a-f]\{14\}\)$/+\1/")
      fi
      if [[ "${KUBE_GIT_TREE_STATE}" == "dirty" ]]; then
        # git describe --dirty only considers changes to existing files, but
        # that is problematic since new untracked .go files affect the build,
        # so use our idea of "dirty" from git status instead.
        KUBE_GIT_VERSION+="-dirty"
      fi


      # If HEAD is exactly on a tag (no commit distance, i.e. no build metadata),
      # then the tag must match the constants in the source code because
      # those are what the binaries report.
      if [[ "${KUBE_GIT_VERSION%-dirty}" != *+* ]]; then
        local source_version
        source_version=$(kube::version::source_version)
        if [[ "${KUBE_GIT_VERSION%-dirty}" != "${source_version}" ]]; then
          kube::log::error "Tag ${KUBE_GIT_VERSION%-dirty} does not match the version in staging/src/k8s.io/component-base/version/base.go: ${source_version}"
          kube::log::error "Update DefaultKubeBinaryVersion, KubePatchVersion and KubePreReleaseVersion there."
          exit 1
        fi
      fi

      # If KUBE_GIT_VERSION is not a valid Semantic Version, then refuse to build.
      if ! [[ "${KUBE_GIT_VERSION}" =~ ^v([0-9]+)\.([0-9]+)(\.[0-9]+)?(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$ ]]; then
          kube::log::error "KUBE_GIT_VERSION should be a valid Semantic Version. Current value: ${KUBE_GIT_VERSION}"
          kube::log::error "Please see more details here: https://semver.org"
          exit 1
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
KUBE_GIT_VERSION='${KUBE_GIT_VERSION-}'
KUBE_GIT_MAJOR='${KUBE_GIT_MAJOR-}'
KUBE_GIT_MINOR='${KUBE_GIT_MINOR-}'
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
# IMPORTANT: if you update any of these, also update the lists in
# hack/print-workspace-status.sh.
kube::version::ldflags() {
  kube::version::get_version_vars

  local -a ldflags
  function add_ldflag() {
    local key=${1}
    local val=${2}
    ldflags+=(
      "-X 'k8s.io/client-go/pkg/version.${key}=${val}'"
      "-X 'k8s.io/component-base/version.${key}=${val}'"
    )
  }

  kube::util::ensure-gnu-compatible-date

  add_ldflag "buildDate" "$(${DATE} ${SOURCE_DATE_EPOCH:+"--date=@${SOURCE_DATE_EPOCH}"} -u +'%Y-%m-%dT%H:%M:%SZ')"

  # Normally the Go toolchain records the VCS information in the binary. Only
  # when the version information was provided externally, it must be injected.
  if [[ -n ${KUBE_GIT_VERSION_FILE-} ]]; then
    if [[ -n ${KUBE_GIT_COMMIT-} ]]; then
      add_ldflag "gitCommit" "${KUBE_GIT_COMMIT}"
      add_ldflag "gitTreeState" "${KUBE_GIT_TREE_STATE}"
    fi
    if [[ -n ${KUBE_GIT_VERSION-} ]]; then
      add_ldflag "gitVersion" "${KUBE_GIT_VERSION}"
    fi
  fi

  # The -ldflags parameter takes a single string, so join the output.
  echo "${ldflags[*]-}"
}
