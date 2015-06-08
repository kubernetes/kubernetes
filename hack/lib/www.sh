#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The source, binary and binary paths.
readonly KUBE_BUILD_WWW_SOURCE_PATH="${KUBE_ROOT}/www"
readonly KUBE_BUILD_WWW_SOURCES=(
  master
)
readonly KUBE_BUILD_WWW_BINARY_PATH="${KUBE_ROOT}/www"
readonly KUBE_BUILD_WWW_BINARIES=(
  app
)
readonly KUBE_BUILD_WWW_OUTPUT_PATH="${KUBE_OUTPUT}/www"

# Compares two semver formatted version numbers. Returns
# true if the have version number is ge the want version number.
# $1 is the have version number.
# $2 is the want version number.
kube::www:check_version() {
  [[ "${1:-0}" == "${2:-0}" ]] && return 1

  have_major=`echo "${1:-0}" | cut -d "." -f -1`
  have_minor=`echo "${1:-0}" | cut -d "." -f 2-`

  want_major=`echo "${2:-0}" | cut -d "." -f -1`
  want_minor=`echo "${2:-0}" | cut -d "." -f 2-`

  if [[ "${have_major}" != "${1:-0}" ]] || [[ "${want_major}" != "${2:-0}" ]]; then
    [[ "${have_major}" -gt "${want_major}" ]] && return 1
    [[ "${have_major}" -lt "${want_major}" ]] && return 0

    [[ "${have_major}" == "${1:-0}" ]] || [[ -z "${have_minor}" ]] && have_minor=0
    [[ "${want_major}" == "${2:-0}" ]] || [[ -z "${want_minor}" ]] && want_minor=0
    kube::www:check_version "${have_minor}" "${want_minor}"
    return $?
  else
    [[ "${1:-0}" -gt "${2:-0}" ]] && return 1 || return 0
  fi
}

# Checks that the `npm` command is available in ${PATH}. If not running on 
# Travis, it also checks that the npm version is good enough for this build.
kube::www::verify_prereqs() {
  if [[ -z "$(which npm)" ]]; then
    kube::log::usage_from_stdin <<EOF

Can't find 'npm' in PATH, please fix and retry.
See https://nodejs.org/download for installation instructions.

EOF
    exit 2
  fi

  # Check that npm is later than nodejs 0.10 release
  if [[ "${TRAVIS:-}" != "true" ]]; then
    local npm_version
    npm_version="$(npm --version)"
    if kube::www:check_version "${npm_version}" "1.2.14"; then
      kube::log::usage_from_stdin <<EOF

Detected npm version: ${npm_version}.
Kubernetes requires npm version 1.2.14 or greater.
Please install npm version 1.2.14 or later.

EOF
      exit 2
    fi
  fi
}

# Copy binaries from ${KUBE_BUILD_WWW_BINARY_PATH} to ${KUBE_BUILD_WWW_OUTPUT_PATH}
kube::www::place_bins() {
  mkdir -p "${KUBE_BUILD_WWW_OUTPUT_PATH}"
  kube::log::status "Placing binaries in ${KUBE_BUILD_WWW_OUTPUT_PATH}"

  local binary
  for binary in "${KUBE_BUILD_WWW_BINARIES[@]:+${KUBE_BUILD_WWW_BINARIES[@]}}"; do
    local directory="${KUBE_BUILD_WWW_BINARY_PATH}/$binary"
    if [[ -d "${directory}" ]]; then
      ( cd "${KUBE_BUILD_WWW_BINARY_PATH}" && tar cf - "${binary}" ) | ( cd "${KUBE_BUILD_WWW_OUTPUT_PATH}" && tar xf - )
    fi
  done
}

kube::www::build_binaries_from_sources() {
  for source in "${sources[@]:+${sources[@]}}"; do
    local directory="${KUBE_BUILD_WWW_SOURCE_PATH}/$source"
    if [[ -d "${directory}" ]]; then
      (
        kube::log::status "Building binaries for ${directory}"
        cd "${directory}"
        kube::log::status "Installing dependencies...."
        npm install >/dev/null 2>&1
        kube::log::status "Running build scripts...."
        npm run build >/dev/null 2>&1
      )
    fi
  done
}

# Build binaries from sources.
# $@ - sources and npm flags. If sources are empty then all sources are built.
kube::www::build_binaries() {
  # Create a sub-shell so that we don't pollute the outer environment
  (
    kube::www::verify_prereqs

    # Use eval to preserve embedded quoted strings.
    local npmflags
    eval "npmflags=(${KUBE_NPMFLAGS:-})"

    local -a sources=()
    local arg
    for arg; do
      if [[ "${arg}" == -* ]]; then
        # Assume arguments starting with a dash are flags to pass to npm.
        npmflags+=("${arg}")
      else
        sources+=("${arg}")
      fi
    done

    if [[ ${#sources[@]} -eq 0 ]]; then
      sources=("${KUBE_BUILD_WWW_SOURCES[@]}")
    fi

    kube::www::build_binaries_from_sources
  )
}
