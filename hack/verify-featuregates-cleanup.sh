#!/usr/bin/env bash

# Copyright 2025 The Kubernetes Authors.
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

# This script verifies that there are no feature gates that have
# been locked to default for 5+ minor releases.
# Usage: `hack/verify-featuregates-cleanup.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

FEATURE_LIST="${KUBE_ROOT}/test/compatibility_lifecycle/reference/versioned_feature_list.yaml"

# Returns features that have been locked to their default and the versions at which they were locked.
function extract_locked_features() {
  # shellcheck disable=SC2016
  # Disabled because `$name` is part of the yq expression and shouldn't be expanded by Bash.
  yq '.[] |
    select(.versionedSpecs != null) | 
    .name as $name | 
    .versionedSpecs[] | 
    select(.lockToDefault == true) | 
    $name + ":" + .version' "${FEATURE_LIST}"
}

# Returns 0 if the feature gate needs cleanup, 1 otherwise.
function needs_cleanup() {
  local locked_version="$1"
  local current_version="$2"
  
  local locked_major locked_minor current_major current_minor
  
  IFS='.' read -r locked_major locked_minor <<< "${locked_version}"
  IFS='.' read -r current_major current_minor <<< "${current_version}"

  locked_minor=$((10#${locked_minor}))
  # The current version passed to this function might contain a trailing "+" (e.g., "1.34+"). Strip it for numeric comparison.
  current_minor=${current_minor%+}
  current_minor=$((10#${current_minor}))
  
  # TODO: In case of a major version upgrade, when would we require feature gates to be cleaned up?
  if [[ "${locked_major}" != "${current_major}" ]]; then
    echo "Error: can't verify feature gate cleanup across major versions (${locked_version} vs ${current_version})" >&2
    exit 1
  fi
  
  # Check if versions are >=5 minor versions apart. 
  # If a feature is locked to its default at version X.Y, it is still needed in the code base
  # for version emulation until the release cycle of X.Y+3 ends. We can be certain 
  # that the release cycle for X.Y+3 ended only once we're in the release cycle for X.Y+5 
  # because the release cycle for X.Y+4 overlaps with the release cycle for X.Y+3.
  if [[ $((current_minor - locked_minor)) -ge 5 ]]; then
    return 0
  else
    return 1
  fi
}

function main() {
  kube::version::get_version_vars
  local current_version="${KUBE_GIT_MAJOR}.${KUBE_GIT_MINOR}"
  echo "Current version: ${current_version}"
  
  local locked_features
  readarray -t locked_features < <(extract_locked_features)
  
  local features_needing_cleanup=()
  for feature in "${locked_features[@]}"; do
    local feature_name locked_version
    IFS=':' read -r feature_name locked_version <<< "${feature}"
    
    if needs_cleanup "${locked_version}" "${current_version}"; then
      features_needing_cleanup+=("${feature_name} (locked since ${locked_version})")
    fi
  done
  
  if [[ ${#features_needing_cleanup[@]} -gt 0 ]]; then
    echo "The following feature gates have been locked to default for >=5 releases and should be cleaned up:" >&2
    for feature in "${features_needing_cleanup[@]}"; do
      echo "  - ${feature}" >&2
    done
    exit 1
  else
    echo "No feature gates need cleanup"
    exit 0
  fi
}

main "$@"
