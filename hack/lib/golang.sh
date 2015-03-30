#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# The golang package that we are building.
readonly LMKTFY_GO_PACKAGE=github.com/GoogleCloudPlatform/lmktfy
readonly LMKTFY_GOPATH="${LMKTFY_OUTPUT}/go"

# The set of server targets that we are only building for Linux
readonly LMKTFY_SERVER_TARGETS=(
  cmd/lmktfy-proxy
  cmd/lmktfy-apiserver
  cmd/lmktfy-controller-manager
  cmd/lmktfylet
  cmd/hyperlmktfy
  cmd/lmktfy
  plugin/cmd/lmktfy-scheduler
)
readonly LMKTFY_SERVER_BINARIES=("${LMKTFY_SERVER_TARGETS[@]##*/}")

# The server platform we are building on.
readonly LMKTFY_SERVER_PLATFORMS=(
  linux/amd64
)

# The set of client targets that we are building for all platforms
readonly LMKTFY_CLIENT_TARGETS=(
  cmd/lmktfyctl
)
readonly LMKTFY_CLIENT_BINARIES=("${LMKTFY_CLIENT_TARGETS[@]##*/}")
readonly LMKTFY_CLIENT_BINARIES_WIN=("${LMKTFY_CLIENT_BINARIES[@]/%/.exe}")

# The set of test targets that we are building for all platforms
readonly LMKTFY_TEST_TARGETS=(
  cmd/e2e
  cmd/integration
  cmd/gendocs
  cmd/genman
  examples/k8petstore/web-server
)
readonly LMKTFY_TEST_BINARIES=("${LMKTFY_TEST_TARGETS[@]##*/}")
readonly LMKTFY_TEST_BINARIES_WIN=("${LMKTFY_TEST_BINARIES[@]/%/.exe}")
readonly LMKTFY_TEST_PORTABLE=(
  contrib/for-tests/network-tester/rc.json
  contrib/for-tests/network-tester/service.json
  hack/e2e.go
  hack/e2e-suite
  hack/e2e-internal
  hack/ginkgo-e2e.sh
)

# If we update this we need to also update the set of golang compilers we build
# in 'build/build-image/Dockerfile'
readonly LMKTFY_CLIENT_PLATFORMS=(
  linux/amd64
  linux/386
  linux/arm
  darwin/amd64
  darwin/386
  windows/amd64
)

readonly LMKTFY_ALL_TARGETS=(
  "${LMKTFY_SERVER_TARGETS[@]}"
  "${LMKTFY_CLIENT_TARGETS[@]}"
  "${LMKTFY_TEST_TARGETS[@]}"
)
readonly LMKTFY_ALL_BINARIES=("${LMKTFY_ALL_TARGETS[@]##*/}")

readonly LMKTFY_STATIC_LIBRARIES=(
  lmktfy-apiserver
  lmktfy-controller-manager
  lmktfy-scheduler
)

lmktfy::golang::is_statically_linked_library() {
  local e
  for e in "${LMKTFY_STATIC_LIBRARIES[@]}"; do [[ "$1" == *"/$e" ]] && return 0; done;
  return 1;
}

# lmktfy::binaries_from_targets take a list of build targets and return the
# full go package to be built
lmktfy::golang::binaries_from_targets() {
  local target
  for target; do
    echo "${LMKTFY_GO_PACKAGE}/${target}"
  done
}

# Asks golang what it thinks the host platform is.  The go tool chain does some
# slightly different things when the target platform matches the host platform.
lmktfy::golang::host_platform() {
  echo "$(go env GOHOSTOS)/$(go env GOHOSTARCH)"
}

lmktfy::golang::current_platform() {
  local os="${GOOS-}"
  if [[ -z $os ]]; then
    os=$(go env GOHOSTOS)
  fi

  local arch="${GOARCH-}"
  if [[ -z $arch ]]; then
    arch=$(go env GOHOSTARCH)
  fi

  echo "$os/$arch"
}

# Takes the the platform name ($1) and sets the appropriate golang env variables
# for that platform.
lmktfy::golang::set_platform_envs() {
  [[ -n ${1-} ]] || {
    lmktfy::log::error_exit "!!! Internal error.  No platform set in lmktfy::golang::set_platform_envs"
  }

  export GOOS=${platform%/*}
  export GOARCH=${platform##*/}
}

lmktfy::golang::unset_platform_envs() {
  unset GOOS
  unset GOARCH
}

# Create the GOPATH tree under $LMKTFY_OUTPUT
lmktfy::golang::create_gopath_tree() {
  local go_pkg_dir="${LMKTFY_GOPATH}/src/${LMKTFY_GO_PACKAGE}"
  local go_pkg_basedir=$(dirname "${go_pkg_dir}")

  mkdir -p "${go_pkg_basedir}"
  rm -f "${go_pkg_dir}"

  # TODO: This symlink should be relative.
  ln -s "${LMKTFY_ROOT}" "${go_pkg_dir}"
}

# lmktfy::golang::setup_env will check that the `go` commands is available in
# ${PATH}. If not running on Travis, it will also check that the Go version is
# good enough for the LMKTFY build.
#
# Input Vars:
#   LMKTFY_EXTRA_GOPATH - If set, this is included in created GOPATH
#   LMKTFY_NO_GODEPS - If set, we don't add 'Godeps/_workspace' to GOPATH
#
# Output Vars:
#   export GOPATH - A modified GOPATH to our created tree along with extra
#     stuff.
#   export GOBIN - This is actively unset if already set as we want binaries
#     placed in a predictable place.
lmktfy::golang::setup_env() {
  lmktfy::golang::create_gopath_tree

  if [[ -z "$(which go)" ]]; then
    lmktfy::log::usage_from_stdin <<EOF

Can't find 'go' in PATH, please fix and retry.
See http://golang.org/doc/install for installation instructions.

EOF
    exit 2
  fi

  # Travis continuous build uses a head go release that doesn't report
  # a version number, so we skip this check on Travis.  It's unnecessary
  # there anyway.
  if [[ "${TRAVIS:-}" != "true" ]]; then
    local go_version
    go_version=($(go version))
    if [[ "${go_version[2]}" < "go1.2" ]]; then
      lmktfy::log::usage_from_stdin <<EOF

Detected go version: ${go_version[*]}.
LMKTFY requires go version 1.2 or greater.
Please install Go version 1.2 or later.

EOF
      exit 2
    fi
  fi

  GOPATH=${LMKTFY_GOPATH}

  # Append LMKTFY_EXTRA_GOPATH to the GOPATH if it is defined.
  if [[ -n ${LMKTFY_EXTRA_GOPATH:-} ]]; then
    GOPATH="${GOPATH}:${LMKTFY_EXTRA_GOPATH}"
  fi

  # Append the tree maintained by `godep` to the GOPATH unless LMKTFY_NO_GODEPS
  # is defined.
  if [[ -z ${LMKTFY_NO_GODEPS:-} ]]; then
    GOPATH="${GOPATH}:${LMKTFY_ROOT}/Godeps/_workspace"
  fi
  export GOPATH

  # Unset GOBIN in case it already exists in the current session.
  unset GOBIN
}

# This will take binaries from $GOPATH/bin and copy them to the appropriate
# place in ${LMKTFY_OUTPUT_BINDIR}
#
# Ideally this wouldn't be necessary and we could just set GOBIN to
# LMKTFY_OUTPUT_BINDIR but that won't work in the face of cross compilation.  'go
# install' will place binaries that match the host platform directly in $GOBIN
# while placing cross compiled binaries into `platform_arch` subdirs.  This
# complicates pretty much everything else we do around packaging and such.
lmktfy::golang::place_bins() {
  local host_platform
  host_platform=$(lmktfy::golang::host_platform)

  lmktfy::log::status "Placing binaries"

  local platform
  for platform in "${LMKTFY_CLIENT_PLATFORMS[@]}"; do
    # The substitution on platform_src below will replace all slashes with
    # underscores.  It'll transform darwin/amd64 -> darwin_amd64.
    local platform_src="/${platform//\//_}"
    if [[ $platform == $host_platform ]]; then
      platform_src=""
    fi

    local full_binpath_src="${LMKTFY_GOPATH}/bin${platform_src}"
    if [[ -d "${full_binpath_src}" ]]; then
      mkdir -p "${LMKTFY_OUTPUT_BINPATH}/${platform}"
      find "${full_binpath_src}" -maxdepth 1 -type f -exec \
        rsync -pt {} "${LMKTFY_OUTPUT_BINPATH}/${platform}" \;
    fi
  done
}

# Build binaries targets specified
#
# Input:
#   $@ - targets and go flags.  If no targets are set then all binaries targets
#     are built.
#   LMKTFY_BUILD_PLATFORMS - Incoming variable of targets to build for.  If unset
#     then just the host architecture is built.
lmktfy::golang::build_binaries() {
  # Create a sub-shell so that we don't pollute the outer environment
  (
    # Check for `go` binary and set ${GOPATH}.
    lmktfy::golang::setup_env

    # Fetch the version.
    local version_ldflags
    version_ldflags=$(lmktfy::version::ldflags)

    local host_platform
    host_platform=$(lmktfy::golang::host_platform)

    # Use eval to preserve embedded quoted strings.
    local goflags
    eval "goflags=(${LMKTFY_GOFLAGS:-})"

    local use_go_build
    local -a targets=()
    local arg
    for arg; do
      if [[ "${arg}" == "--use_go_build" ]]; then
        use_go_build=true
      elif [[ "${arg}" == -* ]]; then
        # Assume arguments starting with a dash are flags to pass to go.
        goflags+=("${arg}")
      else
        targets+=("${arg}")
      fi
    done

    if [[ ${#targets[@]} -eq 0 ]]; then
      targets=("${LMKTFY_ALL_TARGETS[@]}")
    fi

    local -a platforms=("${LMKTFY_BUILD_PLATFORMS[@]:+${LMKTFY_BUILD_PLATFORMS[@]}}")
    if [[ ${#platforms[@]} -eq 0 ]]; then
      platforms=("${host_platform}")
    fi

    local binaries
    binaries=($(lmktfy::golang::binaries_from_targets "${targets[@]}"))
    
    local platform
    for platform in "${platforms[@]}"; do
      lmktfy::golang::set_platform_envs "${platform}"
      lmktfy::log::status "Building go targets for ${platform}:" "${targets[@]}"
      if [[ -n ${use_go_build:-} ]]; then
        # Try and replicate the native binary placement of go install without calling go install
        local output_path="${LMKTFY_GOPATH}/bin"
        if [[ $platform != $host_platform ]]; then
          output_path="${output_path}/${platform//\//_}"
        fi

        for binary in "${binaries[@]}"; do
          local bin=$(basename "${binary}")
          if [[ ${GOOS} == "windows" ]]; then
            bin="${bin}.exe"
          fi
          
          if lmktfy::golang::is_statically_linked_library "${binary}"; then
            CGO_ENABLED=0 go build -installsuffix cgo -o "${output_path}/${bin}" \
              "${goflags[@]:+${goflags[@]}}" \
              -ldflags "${version_ldflags}" \
              "${binary}"
          else
            go build -o "${output_path}/${bin}" \
              "${goflags[@]:+${goflags[@]}}" \
              -ldflags "${version_ldflags}" \
              "${binary}"
          fi
        done
      else
        for binary in "${binaries[@]}"; do
          if lmktfy::golang::is_statically_linked_library "${binary}"; then
            CGO_ENABLED=0 go install -installsuffix cgo "${goflags[@]:+${goflags[@]}}" \
              -ldflags "${version_ldflags}" \
              "${binary}"
          else
            go install "${goflags[@]:+${goflags[@]}}" \
              -ldflags "${version_ldflags}" \
              "${binary}"
          fi
        done
      fi
    done
  )
}
