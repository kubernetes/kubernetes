#!/usr/bin/env bash

# Same as make_louhi_prod.sh, but with a different GKE_BUILD_CONFIG.
#
# This script is intended to be used to test changes to Louhi configs *outside*
# of the scope of these build scripts (e.g., when editing Louhi stage types or
# other parameters).

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

"${SCRIPT_DIR}"/make_louhi_prod.sh \
  GKE_BUILD_CONFIG="${SCRIPT_DIR}/config/common.yaml,${SCRIPT_DIR}/config/louhi_test.yaml" \
  INJECT_DEV_VERSION_MARKER=1 \
  "$@"
