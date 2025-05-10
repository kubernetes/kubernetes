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

# This script replaces hack/ginkgo-e2e.sh when not using a cloud provider.
#
# It always runs e2e tests without provider=skeleton and reads cluster info only
# from KUBECONFIG, with no special GCP setup code and no dependency on cluster/
#
# It does not pass any cloud provider related flags and is not compatible with
# ginkgo-e2e.sh, please use ginkgo-e2e.sh when testing with the existing cloud
# provider support and tests that require this.
#
#
# Usage: `hack/e2e.sh`.
#
# Make sure the current KUBECONFIG points to the desired cluster to test.
#
# You also need to build e2e.test and ginkgo, and you should build kubectl though
# it can fall back to the system kubectl.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Options
################################################################################
# GINKGO_PARALLEL: set to 'y' to run tests in parallel
GINKGO_PARALLEL=${GINKGO_PARALLEL:-n}
# GINKGO_SILENCE_SKIPS: set to 'n' to see S character for each skipped test
GINKGO_SILENCE_SKIPS=${GINKGO_SILENCE_SKIPS:-y}
# GINKGO_FORCE_NEWLINES: set to 'y' to print a newline after each S or o character
GINKGO_FORCE_NEWLINES=${GINKGO_FORCE_NEWLINES:-$( if [ "${CI:-false}" = "true" ]; then echo "y"; else echo "n"; fi )}
# GINKGO_NO_COLOR: If 'y', Ginkgo's reporter will not use escape sequence to color output.
#
# Since Kubernetes 1.25, the default is to use colors only when connected to
# a terminal. That is the right choice for all Prow jobs (The UI doesn't
# render them properly).
GINKGO_NO_COLOR=${GINKGO_NO_COLOR:-$(if [ -t 2 ]; then echo n; else echo y; fi)}
# E2E_TEST_DEBUG_TOOL: If set, the command executed will be:
# - `dlv exec` if set to "delve"
# - `gdb` if set to "gdb"
# NOTE: for this to work the e2e.test binary has to be compiled with
# make DBG=1 WHAT=test/e2e/e2e.test
E2E_TEST_DEBUG_TOOL=${E2E_TEST_DEBUG_TOOL:-}
# KUBE_DNS_DOMAIN: defaults to cluster.local, passed to tests.
KUBE_DNS_DOMAIN="${KUBE_DNS_DOMAIN:-cluster.local}"
# PREPULL_IMAGES: if set, tell e2e.test to pre-pull test images.
PREPULL_IMAGES="${PREPULL_IMAGES:-false}"
# KUBECTL: the kubectl binary to use, defaults to the one found in _output
KUBECTL="${KUBECTL:-"$(kube::util::find-binary "kubectl")"}"
if [[ ! -x "$KUBECTL" ]]; then
    echo "WARNING: No kubectl found in _output, falling back on system kubectl ..."
    KUBECTL="kubectl"
    if ! command -v "${KUBECTL}"; then
        echo "ERROR: Failed to find system kubectl fallback, kubectl is required to test. Exiting"
        exit -1
    fi
fi
export KUBECTL
# NUM_NODES: If set, explicitly inform the tests how many worker nodes exist
# Otherwise e2e.test will attempt to auto-detect it.
# Not passed if not set.
################################################################################

# Find the locally compiled ginkgo and e2e.test binary
ginkgo=$(kube::util::find-binary "ginkgo")
e2e_test=$(kube::util::find-binary "e2e.test")


# TODO: e2e.test should probably handle this inside the binary instead of bash
local cc
cc=$("${KUBECTL}" config view -o jsonpath="{.current-context}")
if [[ -n "${KUBE_CONTEXT:-}" ]]; then
    cc="${KUBE_CONTEXT}"
fi
local cluster
cluster=$("${KUBECTL}" config view -o jsonpath="{.contexts[?(@.name == \"${cc}\")].context.cluster}")
KUBE_MASTER_URL=$("${KUBECTL}" config view -o jsonpath="{.clusters[?(@.name == \"${cluster}\")].cluster.server}")


# These arguments are understood by both Ginkgo test suite and CLI.
# Some arguments (like --nodes) are only supported when using the CLI.
# Those get set below when choosing the program.
ginkgo_args=(
  "--poll-progress-after=${GINKGO_POLL_PROGRESS_AFTER:-60m}"
  "--poll-progress-interval=${GINKGO_POLL_PROGRESS_INTERVAL:-5m}"
  "--source-root=${KUBE_ROOT}"
)

# NOTE: Ginkgo's default timeout has been reduced from 24h to 1h in V2, 
# set it manually here as "24h" for backward compatibility purpose.
ginkgo_args+=("--timeout=${GINKGO_TIMEOUT:-24h}")

if [[ -n "${CONFORMANCE_TEST_SKIP_REGEX:-}" ]]; then
  ginkgo_args+=("--skip=${CONFORMANCE_TEST_SKIP_REGEX}")
  ginkgo_args+=("--seed=1436380640")
fi

if [[ "${GINKGO_UNTIL_IT_FAILS:-}" == true ]]; then
  ginkgo_args+=("--until-it-fails=true")
fi

if [[ "${GINKGO_SILENCE_SKIPS}" == "y" ]]; then
  ginkgo_args+=("--silence-skips")
fi

if [[ "${GINKGO_FORCE_NEWLINES}" == "y" ]]; then
  ginkgo_args+=("--force-newlines")
fi

if [[ "${GINKGO_NO_COLOR}" == "y" ]]; then
  ginkgo_args+=("--no-color")
fi

# The --host setting is used only when providing --auth_config
# If --kubeconfig is used, the host to use is retrieved from the .kubeconfig
# file and the one provided with --host is ignored.
# Add path for things like running kubectl binary.
PATH=$(dirname "${e2e_test}"):"${PATH}"
export PATH

# Choose the program to execute and additional arguments for it.
#
# Note that the e2e.test binary must have been built with "make DBG=1"
# when using debuggers, otherwise debug information gets stripped.
case "${E2E_TEST_DEBUG_TOOL:-ginkgo}" in
  ginkgo)
    program=("${ginkgo}")
    if [[ -n "${GINKGO_PARALLEL_NODES:-}" ]]; then
      program+=("--nodes=${GINKGO_PARALLEL_NODES}")
    elif [[ ${GINKGO_PARALLEL} =~ ^[yY]$ ]]; then
      program+=("--nodes=25")
    fi
    program+=("${ginkgo_args[@]:+${ginkgo_args[@]}}")
    ;;
  delve) program=("dlv" "exec") ;;
  gdb) program=("gdb") ;;
  *) kube::log::error_exit "Unsupported E2E_TEST_DEBUG_TOOL=${E2E_TEST_DEBUG_TOOL}" ;;
esac

# Move Ginkgo arguments that are understood by the suite when not using
# the CLI.
suite_args=()
if [ "${E2E_TEST_DEBUG_TOOL:-ginkgo}" != "ginkgo" ]; then
  for arg in "${ginkgo_args[@]}"; do
    suite_args+=("--ginkgo.${arg#--}")
  done
fi

# Generate full dumps of the test result and progress in <report-dir>/ginkgo/,
# using the Ginkgo-specific JSON format and JUnit XML. Ignored if --report-dir
# is not used.
suite_args+=(--report-complete-ginkgo --report-complete-junit)

# When SIGTERM doesn't reach the E2E test suite binaries, ginkgo will exit
# without collecting information from about the currently running and
# potentially stuck tests. This seems to happen when Prow shuts down a test
# job because of a timeout.
#
# It's useful to print one final progress report in that case,
# so GINKGO_PROGRESS_REPORT_ON_SIGTERM (enabled by default when CI=true)
# catches SIGTERM and forwards it to all processes spawned by ginkgo.
#
# Manual invocations can trigger a similar report with `killall -USR1 e2e.test`
# without having to kill the test run.
GINKGO_CLI_PID=
signal_handler() {
  if [ -n "${GINKGO_CLI_PID}" ]; then
    cat <<EOF

*** $0: received $1 signal -> asking Ginkgo to stop.
***
*** Beware that a timeout may have been caused by some earlier test,
*** not necessarily the one which gets interrupted now.
*** See the "Spec runtime" for information about how long the
*** interrupted test was running.

EOF
    # This goes to the process group, which is important because we
    # need to reach the e2e.test processes forked by the Ginkgo CLI.
    kill -TERM "-${GINKGO_CLI_PID}" || true

    echo "Waiting for Ginkgo with pid ${GINKGO_CLI_PID}..."
    wait "{$GINKGO_CLI_PID}"
    echo "Ginkgo terminated."
  fi
}
case "${GINKGO_PROGRESS_REPORT_ON_SIGTERM:-${CI:-no}}" in
  y|yes|true)
    kube::util::trap_add "signal_handler INT" INT
    kube::util::trap_add "signal_handler TERM" TERM
    # Job control is needed to make the Ginkgo CLI and all workers run
    # in their own process group.
    set -m
    ;;
esac

# The following invocation is fairly complex. Let's dump it to simplify
# determining what the final options are. Enabled by default in CI
# environments like Prow.
case "${GINKGO_SHOW_COMMAND:-${CI:-no}}" in y|yes|true) set -x ;; esac

"${program[@]}" "${e2e_test}" -- \
  "${auth_config[@]:+${auth_config[@]}}" \
  --host="${KUBE_MASTER_URL}" \
  --provider=skeleton \
  --repo-root="${KUBE_ROOT}" \
  --dns-domain="${KUBE_DNS_DOMAIN}" \
  --prepull-images="${PREPULL_IMAGES}" \
  ${NUM_NODES:+"--num-nodes=${NUM_NODES}"} \
  ${E2E_REPORT_DIR:+"--report-dir=${E2E_REPORT_DIR}"} \
  ${E2E_REPORT_PREFIX:+"--report-prefix=${E2E_REPORT_PREFIX}"} \
  "${suite_args[@]:+${suite_args[@]}}" \
  "${@}" &

set +x
GINKGO_CLI_PID=$!
wait "${GINKGO_CLI_PID}"
