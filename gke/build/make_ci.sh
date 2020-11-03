#!/usr/bin/env bash

# Same as make_louhi_prod.sh, but with Docker image pushes disabled.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
# shellcheck source=./lib_gke.sh
source "${SCRIPT_DIR}"/lib_gke.sh

gke_build_entrypoint \
  GKE_BUILD_CONFIG="${SCRIPT_DIR}/config/common.yaml,${SCRIPT_DIR}/config/ci.yaml" \
  GKE_BUILD_ACTIONS=compile,package,validate,push-gcs \
  SKIP_DOCKER=1 \
  INJECT_DEV_VERSION_MARKER=1 \
  "$@"
