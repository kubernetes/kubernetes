#!/usr/bin/env bash

# Louhi entrypoint for building GKE Kubernetes for a non-production-release
# branch, most notably for `master'.
#
# The differences vs make_louhi_prod.sh are:
#
#   1) The KUBE_GIT_VERSION is *ALWAYS* set to a dev version (with
#   `INJECT_DEV_VERSION_MARKER=1')
#   2) All GCS artifacts are always pushed to a test bucket, not the prod
#   buckets in louhi_prod.yaml
#   3) Docker images are not built (SKIP_DOCKER=1)
#   4) GCR push is skipped

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
# shellcheck source=./lib_louhi.sh
source "${SCRIPT_DIR}"/lib_louhi.sh
louhi_hook

# You can override the GKE_BUILD_CONFIG given above by just specifying it in the
# command line as an argument, like how you would invoke make.sh.
gke_build_entrypoint \
  GKE_BUILD_CONFIG="${SCRIPT_DIR}/config/common.yaml,${SCRIPT_DIR}/config/louhi_ci.yaml" \
  GKE_BUILD_ACTIONS=compile,package,validate,push-gcs \
  INJECT_DEV_VERSION_MARKER=1 \
  SKIP_DOCKER=1 \
  "$@"
