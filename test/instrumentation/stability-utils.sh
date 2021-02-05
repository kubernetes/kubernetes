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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

stability_check_setup() {
  kube::golang::verify_go_version
  kube::util::ensure-temp-dir
  cd "${KUBE_ROOT}"
  export KUBE_EXTRA_GOPATH=$KUBE_TEMP
  kube::golang::setup_env
  pushd "${KUBE_EXTRA_GOPATH}" >/dev/null
    GO111MODULE=on go get "gopkg.in/yaml.v2"
  popd >/dev/null
}

find_files_to_check() {
  find . -not \( \
      \( \
        -wholename './output' \
        -o -wholename './_output' \
        -o -wholename './_gopath' \
        -o -wholename './release' \
        -o -wholename './target' \
        -o -wholename '*/third_party/*' \
        -o -wholename '*/vendor/*' \
        -o -wholename '*/hack/*' \
        -o -wholename '**/*_test.go' \
        \) -prune \
    \) \
    \( -wholename '**/*.go' \
  \)
}

red=$(tput setaf 1)
green=$(tput setaf 2)
reset=$(tput sgr0)

kube::validate::stablemetrics() {
  stability_check_setup
  static_checked_files=$(find_files_to_check | grep -E ".*.go" | grep -v ".*_test.go") || true
  temp_file=$(mktemp)
  gopathfiles=$(find_files_to_check | grep -E 'test/instrumentation/.*.go' | grep -v "test/instrumentation/main.*go" | cut -c 3-)
  for i in "${static_checked_files[@]}"
  do
    # Deliberately allow word split here
    # shellcheck disable=SC2086
    stabilitycheck=$(go run "test/instrumentation/main.go" $gopathfiles -- $i 1>$temp_file)
    if $stabilitycheck; then 
      echo -e "${green}Diffing test/instrumentation/testdata/stable-metrics-list.yaml\n${reset}"
      if diff -u "$KUBE_ROOT/test/instrumentation/testdata/stable-metrics-list.yaml" "$temp_file"; then
        echo -e "${green}\nPASS metrics stability verification ${reset}"
        return 0
      fi
    fi
    echo "${red}!!! Metrics Stability static analysis has failed!${reset}" >&2
    echo "${red}!!! Please run ./hack/update-generated-stable-metrics.sh to update the golden list.${reset}" >&2
    exit 1
  done
}

kube::update::stablemetrics() {
  stability_check_setup
  static_checked_files=$(find_files_to_check | grep -E ".*.go" | grep -v ".*_test.go") || true
  temp_file=$(mktemp)
  gopathfiles=$(find_files_to_check | grep -E 'test/instrumentation/.*.go' | grep -v "test/instrumentation/main.*go" |     cut -c 3-)
  for i in "${static_checked_files[@]}"
  do
    # Deliberately allow word split here
    # shellcheck disable=SC2086
    stabilitycheck=$(go run "test/instrumentation/main.go" $gopathfiles -- $i 1>$temp_file)
    if ! $stabilitycheck; then
      echo "${red}!!! updating golden list of metrics has failed! ${reset}" >&2
      exit 1
    fi
    mv -f "$temp_file" "${KUBE_ROOT}/test/instrumentation/testdata/stable-metrics-list.yaml"
    echo "${green}Updated golden list of stable metrics.${reset}"
  done
}

