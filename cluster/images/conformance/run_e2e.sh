#!/bin/bash
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
shutdown () {
    E2E_SUITE_PID=$(pgrep e2e.test)
    echo "sending TERM to ${E2E_SUITE_PID}"
    kill -s TERM "${E2E_SUITE_PID}"

    # Kind of a hack to wait for this pid to finish.
    # Since it's not a child of this shell we cannot use wait.
    tail --pid ${E2E_SUITE_PID} -f /dev/null
    saveResults
}

saveResults() {
    cd "${RESULTS_DIR}" || exit
    tar -czf e2e.tar.gz ./*
    # mark the done file as a termination notice.
    echo -n "${RESULTS_DIR}/e2e.tar.gz" > "${RESULTS_DIR}/done"
}

# We get the TERM from kubernetes and handle it gracefully
trap shutdown TERM

ginkgo_args=(
    "--focus=${E2E_FOCUS}"
    "--skip=${E2E_SKIP}"
    "--noColor=true"
)

case ${E2E_PARALLEL} in
    'y'|'Y')           ginkgo_args+=("--nodes=25") ;;
    [1-9]|[1-9][0-9]*) ginkgo_args+=("--nodes=${E2E_PARALLEL}") ;;
esac

echo "/usr/local/bin/ginkgo ${ginkgo_args[@]} /usr/local/bin/e2e.test -- --disable-log-dump --repo-root=/kubernetes --provider=\"${E2E_PROVIDER}\" --report-dir=\"${RESULTS_DIR}\" --kubeconfig=\"${KUBECONFIG}\""
/usr/local/bin/ginkgo "${ginkgo_args[@]}" /usr/local/bin/e2e.test -- --disable-log-dump --repo-root=/kubernetes --provider="${E2E_PROVIDER}" --report-dir="${RESULTS_DIR}" --kubeconfig="${KUBECONFIG}" | tee ${RESULTS_DIR}/e2e.log &
# $! is the pid of tee, not ginkgo
wait $(pgrep ginkgo)
saveResults
