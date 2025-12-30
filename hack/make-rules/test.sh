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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env
kube::util::require-jq

# start the cache mutation detector by default so that cache mutators will be found
KUBE_CACHE_MUTATION_DETECTOR="${KUBE_CACHE_MUTATION_DETECTOR:-true}"
export KUBE_CACHE_MUTATION_DETECTOR

# panic the server on watch decode errors since they are considered coder mistakes
KUBE_PANIC_WATCH_DECODE_ERROR="${KUBE_PANIC_WATCH_DECODE_ERROR:-true}"
export KUBE_PANIC_WATCH_DECODE_ERROR

kube::test::find_go_packages() {
  (
    cd "${KUBE_ROOT}"

    # Get a list of all the modules in this workspace.
    local -a workspace_module_patterns
    kube::util::read-array workspace_module_patterns < <(go list -m -json | jq -r '.Path + "/..."')

    # Get a list of all packages which have test files, but filter out ones
    # that we don't want to run by default (i.e. are not unit-tests).
    go list -find \
        -f '{{if or (gt (len .TestGoFiles) 0) (gt (len .XTestGoFiles) 0)}}{{.ImportPath}}{{end}}' \
        "${workspace_module_patterns[@]}" \
        | grep -vE \
            -e '^k8s.io/kubernetes/third_party(/.*)?$' \
            -e '^k8s.io/kubernetes/cmd/kubeadm/test(/.*)?$' \
            -e '^k8s.io/kubernetes/test/e2e$' \
            -e '^k8s.io/kubernetes/test/e2e_dra$' \
            -e '^k8s.io/kubernetes/test/e2e_node(/.*)?$' \
            -e '^k8s.io/kubernetes/test/e2e_kubeadm(/.*)?$' \
            -e '^k8s.io/.*/test/integration(/.*)?$'
  )
}

set -x

# TODO: This timeout should really be lower, this is a *long* time to test one
# package, however pkg/api/testing in particular will fail with a lower timeout
# currently. We should attempt to lower this over time.
KUBE_TIMEOUT=${KUBE_TIMEOUT:--timeout=180s}
KUBE_COVER=${KUBE_COVER:-n} # set to 'y' to enable coverage collection
KUBE_COVERMODE=${KUBE_COVERMODE:-atomic}
# The directory to save test coverage reports to, if generating them. If unset,
# a semi-predictable temporary directory will be used.
KUBE_COVER_REPORT_DIR="${KUBE_COVER_REPORT_DIR:-}"
# use KUBE_RACE="" to disable the race detector
# this is defaulted to "-race" in make test as well
# NOTE: DO NOT ADD A COLON HERE. KUBE_RACE="" is meaningful!
KUBE_RACE=${KUBE_RACE-"-race"}
# Set to the goveralls binary path to report coverage results to Coveralls.io.
KUBE_GOVERALLS_BIN=${KUBE_GOVERALLS_BIN:-}
# once we have multiple group supports
# Create a junit-style XML test report in this directory if set.
KUBE_JUNIT_REPORT_DIR=${KUBE_JUNIT_REPORT_DIR:-}
# If KUBE_JUNIT_REPORT_DIR is unset, and ARTIFACTS is set, then have them match.
if [[ -z "${KUBE_JUNIT_REPORT_DIR:-}" && -n "${ARTIFACTS:-}" ]]; then
    export KUBE_JUNIT_REPORT_DIR="${ARTIFACTS}"
fi
# Set to 'y' to keep the verbose stdout from tests when KUBE_JUNIT_REPORT_DIR is
# set.
KUBE_KEEP_VERBOSE_TEST_OUTPUT=${KUBE_KEEP_VERBOSE_TEST_OUTPUT:-n}
# Set to 'false' to disable reduction of the JUnit file to only the top level tests.
KUBE_PRUNE_JUNIT_TESTS=${KUBE_PRUNE_JUNIT_TESTS:-true}

set +x

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
  case ${opt} in
    h)
      kube::test::usage
      exit 0
      ;;
    p)
      PARALLEL="${OPTARG}"
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
    :)
      kube::log::usage "Option -${OPTARG} <value>"
      kube::test::usage
      exit 1
      ;;
    ?)
      kube::test::usage
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

# Use eval to preserve embedded quoted strings.
#
# KUBE_TEST_ARGS contains arguments for `go test` (like -short)
# and may end with `-args <arguments for test binary>`, so it
# has to be passed to `go test` at the end of the invocation.
testargs=()
eval "testargs=(${KUBE_TEST_ARGS:-})"

# gotestsum --format value
# "standard-quiet" let's some stderr log messages through, "pkgname-and-test-fails" is similar and doesn't (https://github.com/kubernetes/kubernetes/issues/130934#issuecomment-2739957840).
gotestsum_format=pkgname-and-test-fails
if [[ -n "${FULL_LOG:-}" ]] ; then
  gotestsum_format=standard-verbose
fi

goflags=()

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
  # If the user passed no targets in, we want ~everything.
  kube::util::read-array testcases < <(kube::test::find_go_packages)
else
  # If the user passed targets, we should normalize them.
  # This can be slow for large numbers of inputs.
  kube::log::status "Normalizing Go targets"
  kube::util::read-array testcases < <(kube::golang::normalize_go_targets "${testcases[@]}")
fi
set -- "${testcases[@]+${testcases[@]}}"

if [[ -n "${KUBE_RACE}" ]] ; then
  goflags+=("${KUBE_RACE}")
fi

junitFilenamePrefix() {
  if [[ -z "${KUBE_JUNIT_REPORT_DIR}" ]]; then
    echo ""
    return
  fi
  mkdir -p "${KUBE_JUNIT_REPORT_DIR}"
  echo "${KUBE_JUNIT_REPORT_DIR}/junit_$(kube::util::sortable_date)"
}

installTools() {
  if ! command -v gotestsum >/dev/null 2>&1; then
    kube::log::status "gotestsum not found; installing from ./hack/tools"
    GOTOOLCHAIN="$(kube::golang::hack_tools_gotoolchain)" go -C "${KUBE_ROOT}/hack/tools" install gotest.tools/gotestsum
  fi

  if ! command -v prune-junit-xml >/dev/null 2>&1; then
    kube::log::status "prune-junit-xml not found; installing from ./cmd"
    go -C "${KUBE_ROOT}/cmd/prune-junit-xml" install .
  fi
}

runTests() {
  local junit_filename_prefix
  junit_filename_prefix=$(junitFilenamePrefix)

  installTools

  # Enable coverage data collection?
  local cover_msg
  local COMBINED_COVER_PROFILE

  if [[ ${KUBE_COVER} =~ ^[yY]$ ]]; then
    cover_msg="with code coverage"
    if [[ -z "${KUBE_COVER_REPORT_DIR}" ]]; then
      cover_report_dir="/tmp/k8s_coverage/$(kube::util::sortable_date)"
    else
      cover_report_dir="${KUBE_COVER_REPORT_DIR}"
    fi
    kube::log::status "Saving coverage output in '${cover_report_dir}'"
    mkdir -p "${@+${@/#/${cover_report_dir}/}}"
    COMBINED_COVER_PROFILE="${cover_report_dir}/combined-coverage.out"
    goflags+=(-cover -covermode="${KUBE_COVERMODE}" -coverprofile="${COMBINED_COVER_PROFILE}")
  else
    cover_msg="without code coverage"
  fi

  # Keep the raw JSON output in addition to the JUnit file?
  local jsonfile=""
  if [[ -n "${junit_filename_prefix}" ]] && [[ ${KUBE_KEEP_VERBOSE_TEST_OUTPUT} =~ ^[yY]$ ]]; then
      jsonfile="${junit_filename_prefix}.stdout"
  fi

  kube::log::status "Running tests ${cover_msg} ${KUBE_RACE:+"and with ${KUBE_RACE}"}"
  kube::log::run gotestsum --format="${gotestsum_format}" \
            --jsonfile="${jsonfile}" \
            --junitfile="${junit_filename_prefix:+"${junit_filename_prefix}.xml"}" \
            --raw-command \
            -- \
            go test -json \
            "${goflags[@]:+${goflags[@]}}" \
            "${KUBE_TIMEOUT}" \
            "$@" \
            "${testargs[@]:+${testargs[@]}}" \
    && rc=$? || rc=$?

  if [[ -n "${junit_filename_prefix}" ]]; then
    prune-junit-xml -prune-tests="${KUBE_PRUNE_JUNIT_TESTS}" "${junit_filename_prefix}.xml"
  fi

  if [[ ${KUBE_COVER} =~ ^[yY]$ ]]; then
    coverage_html_file="${cover_report_dir}/combined-coverage.html"
    go tool cover -html="${COMBINED_COVER_PROFILE}" -o="${coverage_html_file}"
    kube::log::status "Combined coverage report: ${coverage_html_file}"
  fi

  return "${rc}"
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
  local fileslimit
  fileslimit="$(ulimit -n)"
  if [[ ${fileslimit} -lt 1000 ]]; then
    echo "WARNING: ulimit -n (files) should be at least 1000, is ${fileslimit}, may cause test failure";
  fi
}

# monitor periodically dumps information about running processes
# in a format that helps determine what ran in parallel and
# which commands consumed CPU cycles.
#
# Lines where excessive CPU consumption was detected are prefixed
# with ERROR.
#
# Example:
# INFO 03:00m config.test,0,0s,22.4% entrypoint,0,0s,0.0% link,10,1s,81.4% bash,0,0s,0.0% bash,0,0s,0.0% bash,0,0s,0.0% go,0,24s,14.0% gotestsum,0,0s,0.0% make,0,0s,0.0% ps,0,0s,0.0%
# ...
# ERROR 04:21m devicetainteviction.test,0,12s,280%(BAD) endpoint.test,0,0s,7.1% endpointslice.test,0,0s,8.5% endpointslicemirroring.test,0,0s,84.3% metrics.test,0,0s,3.1% entrypoint,0,0s,0.0% compile,10,1s,63.9% link,10,0s,58.8% bash,0,0s,0.0% bash,0,0s,0.0% bash,0,0s,0.0% go,0,30s,12.0% gotestsum,0,1s,0.6% make,0,0s,0.0% ps,0,0s,0.0% 
monitor() {
    start=$SECONDS
    echo "# ELAPSED: COMMAND,NICE,CPUTIME,%CPU ..."
    # We sample frequently and then only print lines with excessive CPU consumption or one line every 30 seconds.
    periodic=30
    delay=1
    while sleep "${delay}"; do
        elapsed=$((SECONDS - $start))
        # -o comm truncates the command name, so here we use cmd and then only use the first word of it,
        # stripping directories. The test binaries are in a tmp directory, so we only see the last part
        # of the package (devicetainteviction.test instead of k8s.io/kubernetes/pkg/controller/devicetainteviction).
        ps --no-headers --cols 240 -o nice,cputimes,%cpu,cmd --sort cmd | (
            header=INFO
            line=$(printf "%02d:%02dm " $((elapsed / 60)) $((elapsed % 60)))
            # Not a pipe, we need the while ... do to run in the current shell!
            while read -r nice cputimes cpu comm args; do
                line+="$(basename "${comm}"),${nice},${cputimes}s,${cpu}%"
                # Excessive CPU consumption happens if:
                # - > 5 seconds of CPU time consumed (short-lived tests are less problematic).
                # - More than one CPU kept busy, with 200% as threshold to avoid potential false positives.
                # - Priority not reduced, i.e. it might have stolen CPU time from other processes.
                if [[ "${cputimes}" -gt 5 ]] && [[ "${cpu%.*}" -gt 200 ]] && [[ "${nice}" -eq 0 ]]; then
                    line+="(BAD)"
                    header=ERROR
                fi
                line+=" "
            done
            if [[ "${header}" = ERROR ]] || [[ $((elapsed % "${periodic}")) -eq 0 ]]; then
                echo "${header} ${line}"
            fi
        )
    done
}

# check_monitor stops monitoring, then checks for problematic commands.
check_monitor() {
    kill "${monitor_pid}"
    problems=$(
        for cmd in $(sed -e 's/ /\n/g' "${ARTIFACTS}/commands.log" | grep '(BAD)' | sed -e 's/,.*//' | sort -u); do
            sed -e 's/ /\n/g' "${ARTIFACTS}/commands.log" | grep '(BAD)' | grep "^${cmd}" | sort -n -k4 -t, | tail -1 | sed -e 's/,0,/ /' -e 's/,/ /' -e 's/(BAD)//'
        done
            )
    if [[ -n "${problems}" ]]; then
        cat >"${ARTIFACTS}/junit_test_sh_$$.xml" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
<testsuite tests="1" failures="1" name="hack/make-rules/test.sh">
<testcase classname="hack/make-rules/test.sh" name="CPU consumption">
<failure message="Excessive CPU consumption detected." type="">

COMMAND CPUTIMES %CPU
${problems}

COMMAND: binary name (not necessarily unique)
CPUTIMES: total amount of CPU time in seconds at the time when the check was triggered
%CPU: average percent of CPUs used since starting the command, >100% when using more than one CPU

These commands kept more than one CPU busy for
prolonged periods of time without reducing their
priority. When running multiple test binaries in parallel,
this can prevent other tests from running long enough
such that they run into timeouts and flake.

Go packages which use more than one CPU during unit testing
can reduce their priority with a main_test.go which contains:

   import (
      _ "k8s.io/apimachinery/pkg/util/benice" // Lower process priority.
   )

In such tests it may be necessary to use synctest to decouple
from real-world time.

For details on command execution see artifacts/commands.log.
</failure>
</testcase>
</testsuite>
</testsuites>
EOF
        exit 1
    fi
}


checkFDs

if [[ -n "${ARTIFACTS:-}" ]]; then
    monitor >"${ARTIFACTS}/commands.log" &
    monitor_pid=$!
    kube::util::trap_add check_monitor EXIT
fi

runTests "$@"

# We might run the tests for multiple versions, but we want to report only
# one of them to coveralls. Here we report coverage from the last run.
reportCoverageToCoveralls
