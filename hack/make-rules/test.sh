#!/bin/bash

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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

# start the cache mutation detector by default so that cache mutators will be found
KUBE_CACHE_MUTATION_DETECTOR="${KUBE_CACHE_MUTATION_DETECTOR:-true}"
export KUBE_CACHE_MUTATION_DETECTOR

# panic the server on watch decode errors since they are considered coder mistakes
KUBE_PANIC_WATCH_DECODE_ERROR="${KUBE_PANIC_WATCH_DECODE_ERROR:-true}"
export KUBE_PANIC_WATCH_DECODE_ERROR

# Handle case where OS has sha#sum commands, instead of shasum.
if which shasum >/dev/null 2>&1; then
  SHA1SUM="shasum -a1"
elif which sha1sum >/dev/null 2>&1; then
  SHA1SUM="sha1sum"
else
  echo "Failed to find shasum or sha1sum utility." >&2
  exit 1
fi

kube::test::find_dirs() {
  (
    cd ${KUBE_ROOT}
    find -L . -not \( \
        \( \
          -path './_artifacts/*' \
          -o -path './bazel-*/*' \
          -o -path './_output/*' \
          -o -path './_gopath/*' \
          -o -path './cmd/kubeadm/test/*' \
          -o -path './contrib/podex/*' \
          -o -path './output/*' \
          -o -path './release/*' \
          -o -path './target/*' \
          -o -path './test/e2e/*' \
          -o -path './test/e2e_node/*' \
          -o -path './test/integration/*' \
          -o -path './third_party/*' \
          -o -path './staging/*' \
          -o -path './vendor/*' \
        \) -prune \
      \) -name '*_test.go' -print0 | xargs -0n1 dirname | sed "s|^\./|${KUBE_GO_PACKAGE}/|" | LC_ALL=C sort -u

    find -L . \
        -path './_output' -prune \
        -o -path './vendor/k8s.io/client-go/*' \
        -o -path './vendor/k8s.io/apiserver/*' \
        -o -path './test/e2e_node/system/*' \
      -name '*_test.go' -print0 | xargs -0n1 dirname | sed "s|^\./|${KUBE_GO_PACKAGE}/|" | LC_ALL=C sort -u

    # run tests for client-go
    find ./staging/src/k8s.io/client-go -name '*_test.go' \
      -name '*_test.go' -print0 | xargs -0n1 dirname | sed 's|^\./staging/src/|./vendor/|' | LC_ALL=C sort -u

    # run tests for apiserver
    find ./staging/src/k8s.io/apiserver -name '*_test.go' \
      -name '*_test.go' -print0 | xargs -0n1 dirname | sed 's|^\./staging/src/|./vendor/|' | LC_ALL=C sort -u

    # run tests for apimachinery
    find ./staging/src/k8s.io/apimachinery -name '*_test.go' \
      -name '*_test.go' -print0 | xargs -0n1 dirname | sed 's|^\./staging/src/|./vendor/|' | LC_ALL=C sort -u

    find ./staging/src/k8s.io/kube-aggregator -name '*_test.go' \
      -name '*_test.go' -print0 | xargs -0n1 dirname | sed 's|^\./staging/src/|./vendor/|' | LC_ALL=C sort -u

    find ./staging/src/k8s.io/apiextensions-apiserver -not \( \
        \( \
          -path '*/test/integration/*' \
        \) -prune \
      \) -name '*_test.go' \
      -name '*_test.go' -print0 | xargs -0n1 dirname | sed 's|^\./staging/src/|./vendor/|' | LC_ALL=C sort -u

    find ./staging/src/k8s.io/sample-apiserver -name '*_test.go' \
      -name '*_test.go' -print0 | xargs -0n1 dirname | sed 's|^\./staging/src/|./vendor/|' | LC_ALL=C sort -u
  )
}

KUBE_TIMEOUT=${KUBE_TIMEOUT:--timeout 120s}
KUBE_COVER=${KUBE_COVER:-n} # set to 'y' to enable coverage collection
KUBE_COVERMODE=${KUBE_COVERMODE:-atomic}
# How many 'go test' instances to run simultaneously when running tests in
# coverage mode.
KUBE_COVERPROCS=${KUBE_COVERPROCS:-4}
KUBE_RACE=${KUBE_RACE:-}   # use KUBE_RACE="-race" to enable race testing
# Set to the goveralls binary path to report coverage results to Coveralls.io.
KUBE_GOVERALLS_BIN=${KUBE_GOVERALLS_BIN:-}
# Lists of API Versions of each groups that should be tested, groups are
# separated by comma, lists are separated by semicolon. e.g.,
# "v1,compute/v1alpha1,experimental/v1alpha2;v1,compute/v2,experimental/v1alpha3"
# FIXME: due to current implementation of a test client (see: pkg/api/testapi/testapi.go)
# ONLY the last version is tested in each group.
ALL_VERSIONS_CSV=$(IFS=',';echo "${KUBE_AVAILABLE_GROUP_VERSIONS[*]// /,}";IFS=$),federation/v1beta1
KUBE_TEST_API_VERSIONS="${KUBE_TEST_API_VERSIONS:-${ALL_VERSIONS_CSV}}"
# once we have multiple group supports
# Create a junit-style XML test report in this directory if set.
KUBE_JUNIT_REPORT_DIR=${KUBE_JUNIT_REPORT_DIR:-}
# Set to 'y' to keep the verbose stdout from tests when KUBE_JUNIT_REPORT_DIR is
# set.
KUBE_KEEP_VERBOSE_TEST_OUTPUT=${KUBE_KEEP_VERBOSE_TEST_OUTPUT:-n}

kube::test::usage() {
  kube::log::usage_from_stdin <<EOF
usage: $0 [OPTIONS] [TARGETS]

OPTIONS:
  -p <number>   : number of parallel workers, must be >= 1
EOF
}

isnum() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

PARALLEL="${PARALLEL:-1}"
while getopts "hp:i:" opt ; do
  case $opt in
    h)
      kube::test::usage
      exit 0
      ;;
    p)
      PARALLEL="$OPTARG"
      if ! isnum "${PARALLEL}" || [[ "${PARALLEL}" -le 0 ]]; then
        kube::log::usage "'$0': argument to -p must be numeric and greater than 0"
        kube::test::usage
        exit 1
      fi
      ;;
    i)
      kube::log::usage "'$0': use GOFLAGS='-count <num-iterations>'"
      kube::test::usage
      exit 1
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
eval "goflags=(${GOFLAGS:-})"
eval "testargs=(${KUBE_TEST_ARGS:-})"

# Used to filter verbose test output.
go_test_grep_pattern=".*"

# The go-junit-report tool needs full test case information to produce a
# meaningful report.
if [[ -n "${KUBE_JUNIT_REPORT_DIR}" ]] ; then
  goflags+=(-v)
  # Show only summary lines by matching lines like "status package/test"
  go_test_grep_pattern="^[^[:space:]]\+[[:space:]]\+[^[:space:]]\+/[^[[:space:]]\+"
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
if [[ ${#testcases[@]} -eq 0 ]]; then
  testcases=($(kube::test::find_dirs))
fi
set -- "${testcases[@]+${testcases[@]}}"

junitFilenamePrefix() {
  if [[ -z "${KUBE_JUNIT_REPORT_DIR}" ]]; then
    echo ""
    return
  fi
  mkdir -p "${KUBE_JUNIT_REPORT_DIR}"
  # This filename isn't parsed by anything, and we must avoid
  # exceeding 255 character filename limit. KUBE_TEST_API
  # barely fits there and in coverage mode test names are
  # appended to generated file names, easily exceeding
  # 255 chars in length. So let's just use a sha1 hash of it.
  local KUBE_TEST_API_HASH="$(echo -n "${KUBE_TEST_API//\//-}"| ${SHA1SUM} |awk '{print $1}')"
  echo "${KUBE_JUNIT_REPORT_DIR}/junit_${KUBE_TEST_API_HASH}_$(kube::util::sortable_date)"
}

produceJUnitXMLReport() {
  local -r junit_filename_prefix=$1
  if [[ -z "${junit_filename_prefix}" ]]; then
    return
  fi

  local test_stdout_filenames
  local junit_xml_filename
  test_stdout_filenames=$(ls ${junit_filename_prefix}*.stdout)
  junit_xml_filename="${junit_filename_prefix}.xml"
  if ! command -v go-junit-report >/dev/null 2>&1; then
    kube::log::error "go-junit-report not found; please install with " \
      "go get -u github.com/jstemmer/go-junit-report"
    return
  fi
  cat ${test_stdout_filenames} | go-junit-report > "${junit_xml_filename}"
  if [[ ! ${KUBE_KEEP_VERBOSE_TEST_OUTPUT} =~ ^[yY]$ ]]; then
    rm ${test_stdout_filenames}
  fi
  kube::log::status "Saved JUnit XML test report to ${junit_xml_filename}"
}

runTests() {
  local junit_filename_prefix
  junit_filename_prefix=$(junitFilenamePrefix)

  # If we're not collecting coverage, run all requested tests with one 'go test'
  # command, which is much faster.
  if [[ ! ${KUBE_COVER} =~ ^[yY]$ ]]; then
    kube::log::status "Running tests without code coverage"
    # `go test` does not install the things it builds. `go test -i` installs
    # the build artifacts but doesn't run the tests.  The two together provide
    # a large speedup for tests that do not need to be rebuilt.
    go test -i "${goflags[@]:+${goflags[@]}}" \
      ${KUBE_RACE} ${KUBE_TIMEOUT} "${@}" \
     "${testargs[@]:+${testargs[@]}}"
    go test "${goflags[@]:+${goflags[@]}}" \
      ${KUBE_RACE} ${KUBE_TIMEOUT} "${@}" \
     "${testargs[@]:+${testargs[@]}}" \
     | tee ${junit_filename_prefix:+"${junit_filename_prefix}.stdout"} \
     | grep "${go_test_grep_pattern}" && rc=$? || rc=$?
    produceJUnitXMLReport "${junit_filename_prefix}"
    return ${rc}
  fi

  # Create coverage report directories.
  KUBE_TEST_API_HASH="$(echo -n "${KUBE_TEST_API//\//-}"| ${SHA1SUM} |awk '{print $1}')"
  cover_report_dir="/tmp/k8s_coverage/${KUBE_TEST_API_HASH}/$(kube::util::sortable_date)"
  cover_profile="coverage.out"  # Name for each individual coverage profile
  kube::log::status "Saving coverage output in '${cover_report_dir}'"
  mkdir -p "${@+${@/#/${cover_report_dir}/}}"

  # Run all specified tests, collecting coverage results. Go currently doesn't
  # support collecting coverage across multiple packages at once, so we must issue
  # separate 'go test' commands for each package and then combine at the end.
  # To speed things up considerably, we can at least use xargs -P to run multiple
  # 'go test' commands at once.
  # To properly parse the test results if generating a JUnit test report, we
  # must make sure the output from PARALLEL runs is not mixed. To achieve this,
  # we spawn a subshell for each PARALLEL process, redirecting the output to
  # separate files.

  # ignore paths:
  # vendor/k8s.io/code-generator/cmd/generator: is fragile when run under coverage, so ignore it for now.
  #                            https://github.com/kubernetes/kubernetes/issues/24967
  # vendor/k8s.io/client-go/1.4/rest: causes cover internal errors
  #                            https://github.com/golang/go/issues/16540
  cover_ignore_dirs="vendor/k8s.io/code-generator/cmd/generator|vendor/k8s.io/client-go/1.4/rest"
  for path in $(echo $cover_ignore_dirs | sed 's/|/ /g'); do
      echo -e "skipped\tk8s.io/kubernetes/$path"
  done
  #
  # `go test` does not install the things it builds. `go test -i` installs
  # the build artifacts but doesn't run the tests.  The two together provide
  # a large speedup for tests that do not need to be rebuilt.
  printf "%s\n" "${@}" \
    | grep -Ev $cover_ignore_dirs \
    | xargs -I{} -n 1 -P ${KUBE_COVERPROCS} \
    bash -c "set -o pipefail; _pkg=\"\$0\"; _pkg_out=\${_pkg//\//_}; \
      go test -i ${goflags[@]:+${goflags[@]}} \
        ${KUBE_RACE} \
        ${KUBE_TIMEOUT} \
        -cover -covermode=\"${KUBE_COVERMODE}\" \
        -coverprofile=\"${cover_report_dir}/\${_pkg}/${cover_profile}\" \
        \"\${_pkg}\" \
        ${testargs[@]:+${testargs[@]}}
      go test ${goflags[@]:+${goflags[@]}} \
        ${KUBE_RACE} \
        ${KUBE_TIMEOUT} \
        -cover -covermode=\"${KUBE_COVERMODE}\" \
        -coverprofile=\"${cover_report_dir}/\${_pkg}/${cover_profile}\" \
        \"\${_pkg}\" \
        ${testargs[@]:+${testargs[@]}} \
      | tee ${junit_filename_prefix:+\"${junit_filename_prefix}-\$_pkg_out.stdout\"} \
      | grep \"${go_test_grep_pattern}\"" \
    {} \
    && test_result=$? || test_result=$?

  produceJUnitXMLReport "${junit_filename_prefix}"

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

  return ${test_result}
}

reportCoverageToCoveralls() {
  if [[ ${KUBE_COVER} =~ ^[yY]$ ]] && [[ -x "${KUBE_GOVERALLS_BIN}" ]]; then
    kube::log::status "Reporting coverage results to Coveralls for service ${CI_NAME:-}"
    ${KUBE_GOVERALLS_BIN} -coverprofile="${COMBINED_COVER_PROFILE}" \
    ${CI_NAME:+"-service=${CI_NAME}"} \
    ${COVERALLS_REPO_TOKEN:+"-repotoken=${COVERALLS_REPO_TOKEN}"} \
      || true
  fi
}

checkFDs() {
  # several unittests panic when httptest cannot open more sockets
  # due to the low default files limit on OS X.  Warn about low limit.
  local fileslimit="$(ulimit -n)"
  if [[ $fileslimit -lt 1000 ]]; then
    echo "WARNING: ulimit -n (files) should be at least 1000, is $fileslimit, may cause test failure";
  fi
}

checkFDs


# Convert the CSVs to arrays.
IFS=';' read -a apiVersions <<< "${KUBE_TEST_API_VERSIONS}"
apiVersionsCount=${#apiVersions[@]}
for (( i=0; i<${apiVersionsCount}; i++ )); do
  apiVersion=${apiVersions[i]}
  echo "Running tests for APIVersion: $apiVersion"
  # KUBE_TEST_API sets the version of each group to be tested.
  KUBE_TEST_API="${apiVersion}" runTests "$@"
done

# We might run the tests for multiple versions, but we want to report only
# one of them to coveralls. Here we report coverage from the last run.
reportCoverageToCoveralls
