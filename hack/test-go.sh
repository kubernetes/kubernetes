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

trap 'exit 1' SIGINT

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
        \) -prune \
      \) -name '*_test.go' -print0 | xargs -0n1 dirname | sed 's|^\./||' | sort -u
  )
}

# -covermode=atomic becomes default with -race in Go >=1.3
KUBE_TIMEOUT=${KUBE_TIMEOUT:--timeout 120s}
KUBE_COVER=${KUBE_COVER:-} # set to nonempty string to enable coverage collection
KUBE_COVERMODE=${KUBE_COVERMODE:-atomic}
KUBE_RACE=${KUBE_RACE:-}   # use KUBE_RACE="-race" to enable race testing
# Set to the goveralls binary path to report coverage results to Coveralls.io.
KUBE_GOVERALLS_BIN=${KUBE_GOVERALLS_BIN:-}

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

if [[ -n "${1-}" ]]; then
  test_dirs=$@
else
  test_dirs=$(kube::test::find_dirs)
fi

# Filter out arguments that start with "-" and move them to goflags.
testcases=()
for arg; do
  if [[ "${arg}" == -* ]]; then
    goflags+=("${arg}")
  else
    testcases+=("${arg}")
  fi
done
set -- "${testcases[@]+${testcases[@]}}"

# TODO: this should probably be refactored to avoid code duplication with the
# coverage version.
if [[ $iterations -gt 1 ]]; then
  if [[ $# -eq 0 ]]; then
    set -- $(kube::test::find_dirs)
  fi
  kube::log::status "Running ${iterations} times"
  fails=0
  for arg; do
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
    exit 1
  else
    exit 0
  fi
fi

cover_report_dir=""
combined_cover_profile=""
if [[ -n "${KUBE_COVER}" ]]; then
  cover_report_dir="/tmp/k8s_coverage/$(kube::util::sortable_date)"
  combined_cover_profile="${cover_report_dir}/combined-coverage.out"
  kube::log::status "Saving coverage output in '${cover_report_dir}'"
  mkdir -p ${cover_report_dir}
  # The combined coverage profile needs to start with a line indicating which
  # coverage mode was used (set, count, or atomic). This line is included in
  # each of the coverage profiles generated when running 'go test -cover', but
  # we strip these lines out when combining so that there's only one.
  echo "mode: ${KUBE_COVERMODE}" >${combined_cover_profile}

  # Run all specified tests, optionally collecting coverage if KUBE_COVER is set.
  for arg in ${test_dirs}; do
    pkg=${KUBE_GO_PACKAGE}/${arg}

    cover_profile=${cover_report_dir}/${arg}/coverage.out
    mkdir -p "${cover_report_dir}/${arg}"
    cover_params=(-cover -covermode="${KUBE_COVERMODE}" -coverprofile="${cover_profile}")

    go test "${goflags[@]:+${goflags[@]}}" \
        ${KUBE_RACE} \
        ${KUBE_TIMEOUT} \
        "${cover_params[@]+${cover_params[@]}}" \
        "${pkg}"
    if [[ -f "${cover_profile}" ]]; then
      # Include all coverage reach data in the combined profile, but exclude the
      # 'mode' lines, as there should be only one.
      grep -h -v "^mode:" ${cover_profile} >>${combined_cover_profile} || true
    fi
  done
else
  dirs=(${test_dirs})
  go test "${goflags[@]:+${goflags[@]}}" \
      ${KUBE_RACE} \
      ${KUBE_TIMEOUT} \
      $(printf "${KUBE_GO_PACKAGE}/%s " "${dirs[@]:+${dirs[@]}}")
fi

if [[ -f ${combined_cover_profile} ]]; then
  coverage_html_file="${cover_report_dir}/combined-coverage.html"
  go tool cover -html="${combined_cover_profile}" -o="${coverage_html_file}"
  kube::log::status "Combined coverage report: ${coverage_html_file}"
  if [[ -x "${KUBE_GOVERALLS_BIN}" ]]; then
    ${KUBE_GOVERALLS_BIN} -coverprofile="${combined_cover_profile}" || true
  fi
fi
