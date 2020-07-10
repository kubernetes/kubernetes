#!/bin/bash

set -o nounset
set -o errexit
set -o pipefail

# This script is executes kubernetes e2e tests against an openshift
# cluster. It is intended to be copied to the kubernetes-tests image
# for use in CI and should have no dependencies beyond oc, kubectl and
# e2e.test.

# Set KUBE_E2E_TEST_ARGS to configure test arguments like
# --ginkgo.skip and --ginkgo.focus.
KUBE_E2E_TEST_ARGS="${KUBE_E2E_TEST_ARGS:---ginkgo.skip=\[Slow\]|\[Serial\]|\[Disruptive\]|\[Flaky\]|\[Feature:.+\]}"

# e2e.test is expected to be in the path in CI. Outside of CI, ensure
# e2e.test is built and available in PATH.
if ! which e2e.test &> /dev/null; then
  make WHAT=test/e2e/e2e.test
  ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd -P)"
  PATH="${ROOT_PATH}/_output/local/bin/$(go env GOHOSTOS)/$(go env GOARCH):${PATH}"
  export PATH
fi

# Execute OpenShift prerequisites
# Disable container security
oc adm policy add-scc-to-group privileged system:authenticated system:serviceaccounts
oc adm policy add-scc-to-group anyuid system:authenticated system:serviceaccounts
# Mark the master nodes as unschedulable so tests ignore them
oc get nodes -o name -l 'node-role.kubernetes.io/master' | xargs -L1 oc adm cordon
unschedulable="$( ( oc get nodes -o name -l 'node-role.kubernetes.io/master'; ) | wc -l )"

test_report_dir="${ARTIFACTS:-/tmp/artifacts}"
mkdir -p "${test_report_dir}"

# shellcheck disable=SC2086
e2e.test -num-nodes 4 -ginkgo.noColor \
  ${KUBE_E2E_TEST_ARGS} \
  -report-dir "${test_report_dir}" \
  -allowed-not-ready-nodes ${unschedulable} \
  2>&1 | tee -a "${test_report_dir}/e2e.log"
