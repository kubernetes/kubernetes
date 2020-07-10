#!/usr/bin/env bash

# shellcheck source=openshift-hack/lib/init.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib/init.sh"

# Upstream verify requires recent bash (>= 4.3). If the system bash is
# not recent (e.g openshift ci and macos), download and compile a
# newer bash and make it available in the path.
PATH="$( os::deps::path_with_recent_bash )"

/usr/bin/env bash --version

# Upstream verify requires protoc (>= 3.0.0). If not present, download
# a recent version and make it available in the path.
PATH="$( os::deps::path_with_protoc )"

/usr/bin/env protoc --version

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
