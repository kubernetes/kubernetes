#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# The source, asset and output paths.
readonly KUBE_BUILD_SOURCE_PATH="${KUBE_ROOT}/www"
readonly KUBE_BUILD_SOURCES=(
  master
)
readonly KUBE_BUILD_ASSET_PATH="${KUBE_ROOT}/www"
readonly KUBE_BUILD_OUTPUT="${KUBE_OUTPUT}/www"

# Compares two semver formatted version numbers. Returns 
# true if the supplied version number is ge the required version number.
# $1 is the supplied version number.
# $2 is the required version number.
kube::www:check_version() {
  [ "$1" == "$2" ] && return 1

  supplied_major=`echo $1 | cut -d "." -f -1`
  supplied_minor=`echo $1 | cut -d "." -f 2-`

  required_major=`echo $2 | cut -d "." -f -1`
  required_minor=`echo $2 | cut -d "." -f 2-`

  if [ "$supplied_major" != "$1" ] || [ "$required_major" != "$2" ]; then
    [ "$supplied_major" -gt "$required_major" ] && return 1
    [ "$supplied_major" -lt "$required_major" ] && return 0

    [ "$supplied_major" == "$1" ] || [ -z "$supplied_minor" ] && supplied_minor=0
    [ "$required_major" == "$2" ] || [ -z "$required_minor" ] && required_minor=0
    kube::www:check_version "$supplied_minor" "$required_minor"
    return $?
  else
    [ "$1" -gt "$2" ] && return 1 || return 0
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
Kubernetes requires npm version 2.7.3 or greater.
Please install npm version 2.7.3 or later.

EOF
      exit 2
    fi
  fi
}

# Copy assets from ${KUBE_BUILD_ASSET_PATH} to ${KUBE_BUILD_OUTPUT}
kube::www::place_assets() {
  mkdir -p "${KUBE_BUILD_OUTPUT}"

  local asset
  for asset in "${KUBE_BUILD_ASSETS[@]}"; do
    local package="${KUBE_BUILD_ASSET_PATH}/$asset"
    if [[ -d "${package}" ]]; then
      kube::log::status "Placing assets from:${package}"
      ( cd "${KUBE_BUILD_ASSET_PATH}" && tar cf - "${asset}" ) | ( cd "${KUBE_BUILD_OUTPUT}" && tar xf - )
    fi
  done
}

kube::www::build_assets_from_sources() {
  for source in "${sources[@]:-}"; do
    local package="${KUBE_BUILD_SOURCE_PATH}/$source"
    if [[ -d "${package}" ]]; then
      (
        kube::log::status "Building assets from:${package}"
        cd $package
        kube::log::status "Installing dependencies...."
        npm install >/dev/null 2>&1
        kube::log::status "Running build scripts...."
        npm run compile >/dev/null 2>&1 
      )
    fi
  done
}

# Build assets from sources.
# $@ - sources and npm flags. If sources are empty then all sources are built.
kube::www::build_assets() {
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
      sources=("${KUBE_BUILD_SOURCES[@]}")
    fi

    kube::www::build_assets_from_sources
  )
}
