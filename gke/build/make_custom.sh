#!/usr/bin/env bash

# Build GKE. This is just a thin wrapper around lib.sh. Among the other
# ./make_*.sh scripts, this is the only one that assumes nothing about the
# build. That is, this script is the entrypoint where you have the most control.
# It's a blank slate for customizations!

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
# shellcheck source=./lib_gke.sh
source "${SCRIPT_DIR}"/lib_gke.sh

# We pass along any additional arguments specified in the command line over to
# the function, so that users can tweak the settings (esp. things like setting
# GKE_BUILD_CONFIG and GKE_BUILD_ACTIONS).
gke_build_entrypoint "$@"
