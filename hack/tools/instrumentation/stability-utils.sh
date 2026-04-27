#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

stability_check_setup() {
  kube::util::ensure-temp-dir
  cd "${KUBE_ROOT}"
  kube::golang::setup_env
  export GOBIN="${KUBE_OUTPUT_BIN}"
  echo "Building instrumentation tools..."
  GOTOOLCHAIN="$(kube::golang::hack_tools_gotoolchain)" go -C "${KUBE_ROOT}/hack/tools/instrumentation" install . ./sort ./documentation
}

# shellcheck disable=SC2120
function find_files_to_check() {
    # Similar to find but faster and easier to understand.  We want to include
    # modified and untracked files because this might be running against code
    # which is not tracked by git yet.
    git ls-files -cmo --exclude-standard \
        ':!:vendor/*'        `# catches vendor/...` \
        ':!:*/vendor/*'      `# catches any subdir/vendor/...` \
        ':!:*/testdata/*'    `# catches any subdir/testdata/...` \
        ':!:third_party/*'   `# catches third_party/...` \
        ':!:*/third_party/*' `# catches third_party/...` \
        ':!:hack/*'          `# catches hack/...` \
        ':!:*/hack/*'        `# catches any subdir/hack/...` \
        ':!:test/*'          `# catches test/...` \
        ':!:*/test/*'        `# catches any subdir/test/...` \
        ':!:*/*_test.go' \
        ':!:hack/tools/instrumentation' \
        ':(glob)**/*.go' \
        "$@"
}

# shellcheck disable=SC2120
function find_test_files() {
  git ls-files -cmo --exclude-standard \
      "$@"
}

red=$(tput setaf 1)
green=$(tput setaf 2)
reset=$(tput sgr0)

function kube::validate::stablemetrics() {
  stability_check_setup
  temp_file=$(mktemp)
  
  find_files_to_check \
      | KUBE_ROOT=${KUBE_ROOT} "${GOBIN}/instrumentation" \
            -- \
            - \
            1>"${temp_file}"

  echo -e "${green}Diffing hack/tools/instrumentation/testdata/stable-metrics-list.yaml\n${reset}"
  if diff -u "$KUBE_ROOT/hack/tools/instrumentation/testdata/stable-metrics-list.yaml" "$temp_file"; then
    echo -e "${green}\nPASS metrics stability verification ${reset}"
    return 0
  fi
  echo "${red}!!! Metrics Stability static analysis has failed!${reset}" >&2
  echo "${red}!!! Please run ./hack/update-generated-stable-metrics.sh to update the golden list.${reset}" >&2
  exit 1
}

function kube::validate::test::stablemetrics() {
  stability_check_setup
  temp_file=$(mktemp)
  cd "${KUBE_ROOT}/hack/tools/instrumentation"
  
  find_test_files \
      | KUBE_ROOT=${KUBE_ROOT} "${GOBIN}/instrumentation" \
            -- \
            - \
            1>"${temp_file}"

  echo -e "${green}Diffing hack/tools/instrumentation/testdata/test-stable-metrics-list.yaml\n${reset}"
  if diff -u "$KUBE_ROOT/hack/tools/instrumentation/testdata/test-stable-metrics-list.yaml" "$temp_file"; then
    echo -e "${green}\nPASS metrics stability verification ${reset}"
    return 0
  fi

  echo "${red}!!! Metrics stability static analysis test has failed!${reset}" >&2
  echo "${red}!!! Please run './hack/tools/instrumentation/test-update.sh' to update the golden list.${reset}" >&2
  exit 1
}

function kube::update::stablemetrics() {
  stability_check_setup
  temp_file=$(mktemp)
  
  find_files_to_check \
      | KUBE_ROOT=${KUBE_ROOT} "${GOBIN}/instrumentation" \
            -- \
            - \
            1>"${temp_file}"

  mv -f "$temp_file" "${KUBE_ROOT}/hack/tools/instrumentation/testdata/stable-metrics-list.yaml"
  echo "${green}Updated golden list of stable metrics.${reset}"
}

function kube::update::documentation::list() {
  stability_check_setup
  temp_file=$(mktemp)
  
  find_files_to_check \
      | KUBE_ROOT=${KUBE_ROOT} "${GOBIN}/instrumentation" \
            --allstabilityclasses \
            --endpoint-mappings="hack/tools/instrumentation/endpoint-mappings.yaml" \
            -- \
            - \
            1>"${temp_file}"

  mv -f "$temp_file" "${KUBE_ROOT}/hack/tools/instrumentation/documentation/documentation-list.yaml"
  echo "${green}Updated list of metrics for documentation ${reset}"
}

function kube::update::documentation() {
  stability_check_setup
  temp_file=$(mktemp)
  arg1=$1
  arg2=$2
  doUpdateDocs=$("${GOBIN}/documentation" --major "$arg1" --minor "$arg2" -- 1>"${temp_file}")
  if ! $doUpdateDocs; then
    echo "${red}!!! updating documentation has failed! ${reset}" >&2
    exit 1
  fi
  mv -f "$temp_file" "${KUBE_ROOT}/hack/tools/instrumentation/documentation/documentation.md"
  echo "${green}Updated documentation of metrics.${reset}"
}

function kube::update::test::stablemetrics() {
  stability_check_setup
  temp_file=$(mktemp)
  cd "${KUBE_ROOT}/hack/tools/instrumentation"
  
  find_test_files \
      | KUBE_ROOT=${KUBE_ROOT} "${GOBIN}/instrumentation" \
            -- \
            - \
            1>"${temp_file}"

  mv -f "$temp_file" "${KUBE_ROOT}/hack/tools/instrumentation/testdata/test-stable-metrics-list.yaml"
  echo "${green}Updated test list of stable metrics.${reset}"
}
