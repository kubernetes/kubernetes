#!/usr/bin/env bash

# This script wraps around the make_louhi_{ci,prod}.sh entrypoints and chooses
# either one based on the branch name.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

case "${_LOUHI_BRANCH_NAME}" in
  release-*-gke.*)
    "${SCRIPT_DIR}"/make_louhi_prod.sh
  ;;
  *)
    "${SCRIPT_DIR}"/make_louhi_ci.sh
  ;;
esac
