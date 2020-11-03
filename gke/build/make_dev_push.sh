#!/usr/bin/env bash

# Like make_dev_simple.sh, but pushes to GCS.
#
# Note that license and source injection steps are still skipped. If you want to
# enable them, just pass in "SKIP_DOCKER_LICENSE_INJECTION=0" and
# "SKIP_DOCKER_SOURCE_INJECTION=0" as arguments to this script.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

"${SCRIPT_DIR}"/make_custom.sh \
  GKE_BUILD_ACTIONS=compile,package,validate,push-gcs \
  GKE_BUILD_CONFIG="${SCRIPT_DIR}/config/common.yaml,${SCRIPT_DIR}/config/dev_simple.yaml" \
  SKIP_BCHECK=1 \
  SKIP_DOCKER_LICENSE_INJECTION=1 \
  SKIP_DOCKER_SOURCE_INJECTION=1 \
  VERSION_SUFFIX="${USER}" \
  INJECT_DEV_VERSION_MARKER=1 \
  "$@"
