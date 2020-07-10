#!/usr/bin/env bash

# shellcheck source=openshift-hack/lib/init.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/init.sh"

# Upstream testing requires recent bash (>= 4.3). If the system bash
# is not recent (e.g openshift ci and macos), download and compile a
# newer bash and make it available in the path.
PATH="$( os::deps::path_with_recent_bash )"

/usr/bin/env bash --version

./hack/install-etcd.sh
PATH="${OS_ROOT}/third_party/etcd:${PATH}"

ARTIFACTS="${ARTIFACTS:-/tmp/artifacts}"
mkdir -p "${ARTIFACTS}"

export KUBERNETES_SERVICE_HOST=
export KUBE_JUNIT_REPORT_DIR="${ARTIFACTS}"
export KUBE_KEEP_VERBOSE_TEST_OUTPUT=y
export KUBE_RACE=-race
export KUBE_TEST_ARGS='-p 8'
export LOG_LEVEL=4
export PATH

make test-integration
