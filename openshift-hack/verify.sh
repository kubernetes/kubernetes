#!/usr/bin/env bash

# shellcheck source=openshift-hack/lib/init.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/init.sh"

# Required for openapi verification
PATH="$(pwd)/third_party/etcd:${PATH}"

# Attempt to verify without docker if it is not available.
OS_RUN_WITHOUT_DOCKER=
if ! which docker &> /dev/null; then
  os::log::warning "docker not available, attempting to run verify without it"
  OS_RUN_WITHOUT_DOCKER=y

  # Without docker, shellcheck may need to be installed.
  PATH="$( os::deps::path_with_shellcheck )"
fi
export OS_RUN_WITHOUT_DOCKER

export PATH

ARTIFACTS="${ARTIFACTS:-/tmp/artifacts}"
mkdir -p "${ARTIFACTS}"
export KUBE_JUNIT_REPORT_DIR="${ARTIFACTS}"

make verify
