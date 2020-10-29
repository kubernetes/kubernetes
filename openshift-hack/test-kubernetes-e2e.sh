#!/bin/bash

set -o nounset
set -o errexit
set -o pipefail

# This script is executes kubernetes e2e tests against an openshift
# cluster. It is intended to be copied to the kubernetes-tests image
# for use in CI and should have no dependencies beyond oc, kubectl and
# k8s-e2e.test.

# Identify the platform under test to allow skipping tests that are
# not compatible.
CLUSTER_TYPE="${CLUSTER_TYPE:-gcp}"
case "${CLUSTER_TYPE}" in
  gcp)
    # gce is used as a platform label instead of gcp
    PLATFORM=gce
    ;;
  *)
    PLATFORM="${CLUSTER_TYPE}"
    ;;
esac

# openshift-tests will check the cluster's network configuration and
# automatically skip any incompatible tests. We have to do that manually
# here.
NETWORK_SKIPS="\[Skipped:Network/OpenShiftSDN\]|\[Feature:Networking-IPv6\]|\[Feature:IPv6DualStack.*\]|\[Feature:SCTPConnectivity\]"

# Support serial and parallel test suites
TEST_SUITE="${TEST_SUITE:-parallel}"
COMMON_SKIPS="\[Slow\]|\[Disruptive\]|\[Flaky\]|\[Disabled:.+\]|\[Skipped:${PLATFORM}\]|${NETWORK_SKIPS}"
case "${TEST_SUITE}" in
serial)
  DEFAULT_TEST_ARGS="-focus=\[Serial\] -skip=${COMMON_SKIPS}"
  NODES=1
  ;;
parallel)
  DEFAULT_TEST_ARGS="-skip=\[Serial\]|${COMMON_SKIPS}"
  # Use the same number of nodes - 30 - as specified for the parallel
  # suite defined in origin.
  NODES=${NODES:-30}
  ;;
*)
  echo >&2 "Unsupported test suite '${TEST_SUITE}'"
  exit 1
  ;;
esac

# Set KUBE_E2E_TEST_ARGS to configure test arguments like
# -skip and -focus.
KUBE_E2E_TEST_ARGS="${KUBE_E2E_TEST_ARGS:-${DEFAULT_TEST_ARGS}}"

# k8s-e2e.test and ginkgo are expected to be in the path in
# CI. Outside of CI, ensure k8s-e2e.test and ginkgo are built and
# available in PATH.
if ! which k8s-e2e.test &> /dev/null; then
  make WHAT=vendor/github.com/onsi/ginkgo/v2/ginkgo
  make WHAT=openshift-hack/e2e/k8s-e2e.test
  ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd -P)"
  PATH="${ROOT_PATH}/_output/local/bin/$(go env GOHOSTOS)/$(go env GOARCH):${PATH}"
  export PATH
fi

# Execute OpenShift prerequisites
# Disable container security
oc adm policy add-scc-to-group privileged system:authenticated system:serviceaccounts
oc adm policy add-scc-to-group anyuid system:authenticated system:serviceaccounts
unschedulable="$( ( oc get nodes -o name -l 'node-role.kubernetes.io/master'; ) | wc -l )"

test_report_dir="${ARTIFACTS:-/tmp/artifacts}"
mkdir -p "${test_report_dir}"

# Retrieve the hostname of the server to enable kubectl testing
SERVER=
SERVER="$( kubectl config view | grep server | head -n 1 | awk '{print $2}' )"

# shellcheck disable=SC2086
ginkgo \
  --flake-attempts=3 \
  --timeout="24h" \
  --output-interceptor-mode=none \
  -nodes "${NODES}" -no-color ${KUBE_E2E_TEST_ARGS} \
  "$( which k8s-e2e.test )" -- \
  -report-dir "${test_report_dir}" \
  -host "${SERVER}" \
  -allowed-not-ready-nodes ${unschedulable} \
  2>&1 | tee -a "${test_report_dir}/k8s-e2e.log"
