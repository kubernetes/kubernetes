#!/usr/bin/env bash
# Copyright 2018 The Kubernetes Authors.
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

# Shutdown the tests gracefully then save the results
# shellcheck disable=SC2317 # false positive
shutdown () {
    E2E_SUITE_PID=$(pgrep e2e.test)
    echo "sending TERM to ${E2E_SUITE_PID}"
    kill -s TERM "${E2E_SUITE_PID}"

    # Kind of a hack to wait for this pid to finish.
    # Since it's not a child of this shell we cannot use wait.
    tail --pid "${E2E_SUITE_PID}" -f /dev/null
    saveResults
}

saveResults() {
    cd "${RESULTS_DIR}" || exit
    tar -czf e2e.tar.gz ./*
    # mark the done file as a termination notice.
    echo -n "${RESULTS_DIR}/e2e.tar.gz" > "${RESULTS_DIR}/done"
}

# Optional Golang runner alternative to the bash script.
# Entry provided via env var to simplify invocation.
if [[ -n ${E2E_USE_GO_RUNNER:-} ]]; then
    set -x
    /gorunner && ret=0 || ret=$?
    exit "${ret}"
fi

# We get the TERM from kubernetes and handle it gracefully
trap shutdown TERM

ginkgo_args=()
if [[ -n ${E2E_DRYRUN:-} ]]; then
    ginkgo_args+=("--dry-run=true")
fi

# NOTE: Ginkgo's default timeout has been reduced from 24h to 1h in V2, set it manually here as "24h"
# for backward compatibility purpose.
ginkgo_args+=("--timeout=24h")

case ${E2E_PARALLEL} in
    'y'|'Y'|'true')
        # The flag '--p' will automatically detect the optimal number of ginkgo nodes.
        ginkgo_args+=("--p")
        # Skip serial tests if parallel mode is enabled.
        E2E_SKIP="\\[Serial\\]|${E2E_SKIP}" ;;
esac

ginkgo_args+=(
    "--focus=${E2E_FOCUS}"
    "--skip=${E2E_SKIP}"
    "--no-color=true"
)

set -x
/usr/local/bin/ginkgo "${ginkgo_args[@]}" /usr/local/bin/e2e.test -- --disable-log-dump --repo-root=/kubernetes --provider="${E2E_PROVIDER}" --report-dir="${RESULTS_DIR}" --kubeconfig="${KUBECONFIG}" -v="${E2E_VERBOSITY}" > >(tee "${RESULTS_DIR}"/e2e.log) && ret=0 || ret=$?
set +x
saveResults
exit "${ret}"
