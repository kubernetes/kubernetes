#!/usr/bin/env bash

# shellcheck source=openshift-hack/lib/init.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/init.sh"

./hack/install-etcd.sh
PATH="${OS_ROOT}/third_party/etcd:${PATH}"

ARTIFACTS="${ARTIFACTS:-/tmp/artifacts}"
mkdir -p "${ARTIFACTS}"

export KUBERNETES_SERVICE_HOST=
export KUBE_JUNIT_REPORT_DIR="${ARTIFACTS}"
export KUBE_KEEP_VERBOSE_TEST_OUTPUT=y
# The KUBE_RACE variable has existed for ~5 years but does not appear to have
# functioned prior to https://github.com/kubernetes/kubernetes/pull/132053
# (about three months before the time of writing). Upstream has marked it as an
# optional test in a separate job due to known failures
# (https://github.com/kubernetes/test-infra/pull/34908), so temporarily disabling
# here as well.
# export KUBE_RACE=-race
export KUBE_TEST_ARGS='-p 8'
export LOG_LEVEL=4
export PATH

make test-integration
