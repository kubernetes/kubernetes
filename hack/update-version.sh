#!/usr/bin/env bash
# Copyright The Kubernetes Authors.
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

# Updates the version constants in component-base and client-go.
#
# Usage:
#   hack/update-version.sh                          # bump the number at the end of KubePreReleaseVersion;
#                                                   # if empty, start the next minor version at alpha.0
#   hack/update-version.sh <major>.<minor>.<patch>[-<pre-release>]   # set all values, e.g. 1.38.0-beta.1

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
cd "${KUBE_ROOT}"

COMPONENT_BASE="staging/src/k8s.io/component-base/version/base.go"
CLIENT_GO="staging/src/k8s.io/client-go/pkg/version/base.go"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [<major>.<minor>.<patch>[-<pre-release>]]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  if [[ ! $1 =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-([0-9A-Za-z.-]+))?$ ]]; then
    echo "Invalid version '$1', expected <major>.<minor>.<patch>[-<pre-release>]" >&2
    exit 1
  fi
  binary_version="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
  patch_version="${BASH_REMATCH[3]}"
  pre_release="${BASH_REMATCH[5]}"
else
  current=$(sed -n -E 's/^[[:space:]]*KubePreReleaseVersion = "(.*)"$/\1/p' "${COMPONENT_BASE}")
  if [[ -z ${current} ]]; then
    # After a final release, the next development cycle starts with the next minor version.
    current_binary=$(sed -n -E 's/^[[:space:]]*DefaultKubeBinaryVersion = "(.*)"$/\1/p' "${COMPONENT_BASE}")
    if [[ ! ${current_binary} =~ ^([0-9]+)\.([0-9]+)$ ]]; then
      echo "Cannot parse DefaultKubeBinaryVersion '${current_binary}' in ${COMPONENT_BASE}." >&2
      exit 1
    fi
    binary_version="${BASH_REMATCH[1]}.$((10#${BASH_REMATCH[2]} + 1))"
    patch_version=0
    pre_release="alpha.0"
  elif [[ ${current} =~ ^(.*[^0-9])([0-9]+)$ ]]; then
    binary_version=""
    patch_version=""
    pre_release="${BASH_REMATCH[1]}$((10#${BASH_REMATCH[2]} + 1))"
  else
    echo "KubePreReleaseVersion '${current}' in ${COMPONENT_BASE} does not end in a number, pass the new version as parameter." >&2
    exit 1
  fi
fi

# Snapshots the API testdata of the release that is about to be left behind.
# This uses the current tree, before the version info changes.
snapshot_testdata() {
  local current_binary current_patch release testdata
  current_binary=$(sed -n -E 's/^[[:space:]]*DefaultKubeBinaryVersion = "(.*)"$/\1/p' "${COMPONENT_BASE}")
  current_patch=$(sed -n -E 's/^[[:space:]]*KubePatchVersion = ([0-9]+)$/\1/p' "${COMPONENT_BASE}")
  release="v${current_binary}.${current_patch}"
  testdata="staging/src/k8s.io/api/testdata"
  if [[ -d "${testdata}/${release}" ]]; then
    echo "${testdata}/${release} already exists, refusing to overwrite it." >&2
    exit 1
  fi
  # Should be up-to-date, but let's make sure.
  UPDATE_COMPATIBILITY_FIXTURE_DATA=true go test ./staging/src/k8s.io/api -run //HEAD
  # Now snapshot HEAD.
  cp -fr "${testdata}/HEAD" "${testdata}/${release}"
  # New files are easy to forget in a commit, so stage them automatically.
  git add "${testdata}/${release}"
}

# $1: file, $2-$4: names of the binary version, patch and pre-release constants
update_file() {
  local file="$1" default_name="$2" patch_name="$3" pre_name="$4"
  if [[ -n ${binary_version} ]]; then
    sed -i -E \
      -e "s/^([[:space:]]*${default_name}[[:space:]]*=[[:space:]]*)\"[^\"]*\"/\1\"${binary_version}\"/" \
      -e "s/^([[:space:]]*${patch_name}[[:space:]]*=[[:space:]]*)[0-9]+/\1${patch_version}/" \
      "${file}"
  fi
  sed -i -E "s/^([[:space:]]*${pre_name}[[:space:]]*=[[:space:]]*)\"[^\"]*\"/\1\"${pre_release}\"/" "${file}"
}

if [[ -n ${binary_version} ]]; then
  current_binary_version=$(sed -n -E 's/^[[:space:]]*DefaultKubeBinaryVersion = "(.*)"$/\1/p' "${COMPONENT_BASE}")
  if [[ ${binary_version} != "${current_binary_version}" ]]; then
    snapshot_testdata
  fi
fi

update_file "${COMPONENT_BASE}" DefaultKubeBinaryVersion KubePatchVersion KubePreReleaseVersion
update_file "${CLIENT_GO}" defaultKubeBinaryVersion kubePatchVersion kubePreReleaseVersion

gofmt -l -w "${COMPONENT_BASE}" "${CLIENT_GO}"
git --no-pager diff -- "${COMPONENT_BASE}" "${CLIENT_GO}"

# Update all code which might depend on the version.
hack/update-openapi-spec.sh
git --no-pager status
