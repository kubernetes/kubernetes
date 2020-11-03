#!/usr/bin/env bash

# Louhi entrypoint for building GKE Kubernetes for production.
#
# This is like ./make_custom.sh, but the only difference is that we make some
# tweaks to how gke_build_entrypoint() (the build entrypoint function that
# drives how GKE is built) is invoked. The goal is to make the entrypoint from a
# Louhi stage type into a 1-liner (namely, just invoking this script alone).
#
# This script is only meant to produce versions that are viable as-is for
# eventual PRODUCTION usage. Therefore, there are certain invariants that are
# asserted as part of this invocation, namely:
#
#   1) The KUBE_GIT_VERSION must be a production-like version.
#   2) The artifacts are always pushed to the production buckets/destinations.
#
# The first invariant means that this script will always fail with an error on
# the `master' branch, because this branch will never have a
# GKE-production-compatible tag ancestor. Such production-ready tags (in the
# form `<vX.Y.Z-gke.Nnn[n]>') are only found on GKE-owned release branches (in
# the form `release-X.Y.Z-gke.N').

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
  GKE_BUILD_CONFIG="${SCRIPT_DIR}/config/common.yaml,${SCRIPT_DIR}/config/louhi_prod.yaml" \
  ASSERT_PROD_VERSION=1 \
  "$@"
