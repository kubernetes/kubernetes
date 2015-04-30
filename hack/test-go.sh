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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

kube::test::find_dirs() {
  (
    cd ${KUBE_ROOT}
    find . -not \( \
        \( \
          -wholename './output' \
          -o -wholename './_output' \
          -o -wholename './release' \
          -o -wholename './target' \
          -o -wholename '*/third_party/*' \
          -o -wholename '*/Godeps/*' \
          -o -wholename '*/contrib/podex/*' \
          -o -wholename '*/test/integration/*' \
        \) -prune \
      \) -name '*_test.go' -print0 | xargs -0n1 dirname | sed 's|^\./||' | sort -u
  )
}

# -covermode=atomic becomes default with -race in Go >=1.3
KUBE_TIMEOUT=${KUBE_TIMEOUT:--timeout 120s}
KUBE_COVER=${KUBE_COVER:-n} # set to 'y' to enable coverage collection
KUBE_COVERMODE=${KUBE_COVERMODE:-atomic}
# How many 'go test' instances to run simultaneously when running tests in
# coverage mode.
KUBE_COVERPROCS=${KUBE_COVERPROCS:-4}
KUBE_RACE=${KUBE_RACE:-}   # use KUBE_RACE="-race" to enable race testing
# Set to the goveralls binary path to report coverage results to Coveralls.io.
KUBE_GOVERALLS_BIN=${KUBE_GOVERALLS_BIN:-}
# Comma separated list of API Versions that should be tested.
KUBE_TEST_API_VERSIONS=${KUBE_TEST_API_VERSIONS:-"v1beta1,v1beta3"}
# Prefixes for etcd paths (standard and customized)
ETCD_STANDARD_PREFIX="registry"
ETCD_CUSTOM_PREFIX="kubernetes.io/registry"

kube::test::usage() {
  kube::log::usage_from_stdin <<EOF
usage: $0 [OPTIONS] [TARGETS]

OPTIONS:
  -i <number>   : number of times to run each test, must be >= 1
EOF
}

isnum() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

iterations=1
while getopts "hi:" opt ; do
  case $opt in
    h)
      kube::test::usage
      exit 0
      ;;
    i)
      iterations="$OPTARG"
      if ! isnum "${iterations}" || [[ "${iterations}" -le 0 ]]; then
        kube::log::usage "'$0': argument to -i must be numeric and greater than 0"
        kube::test::usage
        exit 1
      fi
      ;;
    ?)
      kube::test::usage
      exit 1
      ;;
    :)
      kube::log::usage "Option -$OPTARG <value>"
      kube::test::usage
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

# Use eval to preserve embedded quoted strings.
eval "goflags=(${KUBE_GOFLAGS:-})"

# Filter out arguments that start with "-" and move them to goflags.
testcases=()
for arg; do
  if [[ "${arg}" == -* ]]; then
    goflags+=("${arg}")
  else
    testcases+=("${arg}")
  fi
done
if [[ ${#testcases[@]} -eq 0 ]]; then
  testcases=($(kube::test::find_dirs))
fi
set -- "${testcases[@]+${testcases[@]}}"

runTests() {
  # TODO: this should probably be refactored to avoid code duplication with the
  # coverage version.
  if [[ $iterations -gt 1 ]]; then
    if [[ $# -eq 0 ]]; then
      set -- $(kube::test::find_dirs)
    fi
    kube::log::status "Running ${iterations} times"
    fails=0
    for arg; do
      trap 'exit 1' SIGINT
      pkg=${KUBE_GO_PACKAGE}/${arg}
      kube::log::status "${pkg}"
      # keep going, even if there are failures
      pass=0
      count=0
      for i in $(seq 1 ${iterations}); do
        if go test "${goflags[@]:+${goflags[@]}}" \
            ${KUBE_RACE} ${KUBE_TIMEOUT} "${pkg}"; then
          pass=$((pass + 1))
        else
          fails=$((fails + 1))
        fi
        count=$((count + 1))
      done 2>&1
      kube::log::status "${pass} / ${count} passed"
    done
    if [[ ${fails} -gt 0 ]]; then
      return 1
    else
      return 0
    fi
  fi

  # If we're not collecting coverage, run all requested tests with one 'go test'
  # command, which is much faster.
  if [[ ! ${KUBE_COVER} =~ ^[yY]$ ]]; then
    kube::log::status "Running unit tests without code coverage"
    go test "${goflags[@]:+${goflags[@]}}" \
      ${KUBE_RACE} ${KUBE_TIMEOUT} "${@+${@/#/${KUBE_GO_PACKAGE}/}}"
    return 0
  fi

  # Create coverage report directories.
  cover_report_dir="/tmp/k8s_coverage/${KUBE_API_VERSION}/$(kube::util::sortable_date)"
  cover_profile="coverage.out"  # Name for each individual coverage profile
  kube::log::status "Saving coverage output in '${cover_report_dir}'"
  mkdir -p "${@+${@/#/${cover_report_dir}/}}"

  # Run all specified tests, collecting coverage results. Go currently doesn't
  # support collecting coverage across multiple packages at once, so we must issue
  # separate 'go test' commands for each package and then combine at the end.
  # To speed things up considerably, we can at least use xargs -P to run multiple
  # 'go test' commands at once.
  printf "%s\n" "${@}" | xargs -I{} -n1 -P${KUBE_COVERPROCS} \
      go test "${goflags[@]:+${goflags[@]}}" \
          ${KUBE_RACE} \
          ${KUBE_TIMEOUT} \
          -cover -covermode="${KUBE_COVERMODE}" \
          -coverprofile="${cover_report_dir}/{}/${cover_profile}" \
          "${cover_params[@]+${cover_params[@]}}" \
          "${KUBE_GO_PACKAGE}/{}"

  COMBINED_COVER_PROFILE="${cover_report_dir}/combined-coverage.out"
  {
    # The combined coverage profile needs to start with a line indicating which
    # coverage mode was used (set, count, or atomic). This line is included in
    # each of the coverage profiles generated when running 'go test -cover', but
    # we strip these lines out when combining so that there's only one.
    echo "mode: ${KUBE_COVERMODE}"

    # Include all coverage reach data in the combined profile, but exclude the
    # 'mode' lines, as there should be only one.
    for x in `find "${cover_report_dir}" -name "${cover_profile}"`; do
      cat $x | grep -h -v "^mode:" || true
    done
  } >"${COMBINED_COVER_PROFILE}"

  coverage_html_file="${cover_report_dir}/combined-coverage.html"
  go tool cover -html="${COMBINED_COVER_PROFILE}" -o="${coverage_html_file}"
  kube::log::status "Combined coverage report: ${coverage_html_file}"
}

reportCoverageToCoveralls() {
  if [[ -x "${KUBE_GOVERALLS_BIN}" ]]; then
    ${KUBE_GOVERALLS_BIN} -coverprofile="${COMBINED_COVER_PROFILE}" || true
  fi
}

# Convert the CSV to an array of API versions to test
IFS=',' read -a apiVersions <<< "${KUBE_TEST_API_VERSIONS}"
ETCD_PREFIX=${ETCD_STANDARD_PREFIX}
for apiVersion in "${apiVersions[@]}"; do
  echo "Running tests for APIVersion: $apiVersion"
  KUBE_API_VERSION="${apiVersion}" ETCD_PREFIX=${ETCD_STANDARD_PREFIX} runTests "$@"
done
echo "Using custom etcd path prefix: ${ETCD_CUSTOM_PREFIX}"
KUBE_API_VERSION="${apiVersions[-1]}" ETCD_PREFIX=${ETCD_CUSTOM_PREFIX} runTests "$@"

# We might run the tests for multiple versions, but we want to report only
# one of them to coveralls. Here we report coverage from the last run.
reportCoverageToCoveralls
