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

# The golang package that we are building.
readonly KUBE_GO_PACKAGE=k8s.io/kubernetes
readonly KUBE_GOPATH="${KUBE_OUTPUT}/go"

# Load contrib target functions
if [ -n "${KUBERNETES_CONTRIB:-}" ]; then
  for contrib in "${KUBERNETES_CONTRIB}"; do
    source "${KUBE_ROOT}/contrib/${contrib}/target.sh"
  done
fi

# The set of server targets that we are only building for Linux
# Note: if you are adding something here, you might need to add it to
# kube::build::source_targets in build/common.sh as well.
kube::golang::server_targets() {
  local targets=(
    cmd/kube-proxy
    cmd/kube-apiserver
    cmd/kube-controller-manager
    cmd/kubelet
    cmd/kubemark
    cmd/hyperkube
    federation/cmd/federated-apiserver
    plugin/cmd/kube-scheduler
  )
  if [ -n "${KUBERNETES_CONTRIB:-}" ]; then
    for contrib in "${KUBERNETES_CONTRIB}"; do
      targets+=($(eval "kube::contrib::${contrib}::server_targets"))
    done
  fi
  echo "${targets[@]}"
}
readonly KUBE_SERVER_TARGETS=($(kube::golang::server_targets))
readonly KUBE_SERVER_BINARIES=("${KUBE_SERVER_TARGETS[@]##*/}")

if [[ "${KUBE_FASTBUILD:-}" == "true" ]]; then
  readonly KUBE_SERVER_PLATFORMS=(linux/amd64)
  readonly KUBE_TEST_PLATFORMS=(linux/amd64)
  if [[ "${KUBE_BUILDER_OS:-}" == "darwin"* ]]; then
    readonly KUBE_CLIENT_PLATFORMS=(
      darwin/amd64
      linux/amd64
    )
  else
    readonly KUBE_CLIENT_PLATFORMS=(linux/amd64)
  fi
else

  # The server platform we are building on.
  readonly KUBE_SERVER_PLATFORMS=(
    linux/amd64
    linux/arm
    linux/arm64
    linux/ppc64le
  )

  # If we update this we should also update the set of golang compilers we build
  # in 'build/build-image/cross/Dockerfile'. However, it's only a bit faster since go 1.5, not mandatory
  readonly KUBE_CLIENT_PLATFORMS=(
    linux/amd64
    linux/386
    linux/arm
    linux/arm64
    linux/ppc64le
    darwin/amd64
    darwin/386
    windows/amd64
    windows/386
  )

  # Which platforms we should compile test targets for. Not all client platforms need these tests
  readonly KUBE_TEST_PLATFORMS=(
    linux/amd64
    darwin/amd64
    windows/amd64
    linux/arm
  )
fi

# The set of client targets that we are building for all platforms
readonly KUBE_CLIENT_TARGETS=(
  cmd/kubectl
)
readonly KUBE_CLIENT_BINARIES=("${KUBE_CLIENT_TARGETS[@]##*/}")
readonly KUBE_CLIENT_BINARIES_WIN=("${KUBE_CLIENT_BINARIES[@]/%/.exe}")

# The set of test targets that we are building for all platforms
kube::golang::test_targets() {
  local targets=(
    cmd/integration
    cmd/gendocs
    cmd/genkubedocs
    cmd/genman
    cmd/genyaml
    cmd/mungedocs
    cmd/genbashcomp
    cmd/genswaggertypedocs
    cmd/linkcheck
    examples/k8petstore/web-server/src
    vendor/github.com/onsi/ginkgo/ginkgo
    test/e2e/e2e.test
    test/e2e_node/e2e_node.test
  )
  if [ -n "${KUBERNETES_CONTRIB:-}" ]; then
    for contrib in "${KUBERNETES_CONTRIB}"; do
      targets+=($(eval "kube::contrib::${contrib}::test_targets"))
    done
  fi
  echo "${targets[@]}"
}
readonly KUBE_TEST_TARGETS=($(kube::golang::test_targets))
readonly KUBE_TEST_BINARIES=("${KUBE_TEST_TARGETS[@]##*/}")
readonly KUBE_TEST_BINARIES_WIN=("${KUBE_TEST_BINARIES[@]/%/.exe}")
readonly KUBE_TEST_PORTABLE=(
  test/e2e/testing-manifests
  test/kubemark
  hack/e2e.go
  hack/e2e-internal
  hack/get-build.sh
  hack/ginkgo-e2e.sh
  hack/lib
)

# Gigabytes desired for parallel platform builds. 11 is fairly
# arbitrary, but is a reasonable splitting point for 2015
# laptops-versus-not.
#
# If you are using boot2docker, the following seems to work (note
# that 12000 rounds to 11G):
#   boot2docker down
#   VBoxManage modifyvm boot2docker-vm --memory 12000
#   boot2docker up
readonly KUBE_PARALLEL_BUILD_MEMORY=11

readonly KUBE_ALL_TARGETS=(
  "${KUBE_SERVER_TARGETS[@]}"
  "${KUBE_CLIENT_TARGETS[@]}"
  "${KUBE_TEST_TARGETS[@]}"
)
readonly KUBE_ALL_BINARIES=("${KUBE_ALL_TARGETS[@]##*/}")

readonly KUBE_STATIC_LIBRARIES=(
  kube-apiserver
  kube-controller-manager
  kube-scheduler
  kube-proxy
  kubectl
)

kube::golang::is_statically_linked_library() {
  local e
  for e in "${KUBE_STATIC_LIBRARIES[@]}"; do [[ "$1" == *"/$e" ]] && return 0; done;
  # Allow individual overrides--e.g., so that you can get a static build of
  # kubectl for inclusion in a container.
  if [ -n "${KUBE_STATIC_OVERRIDES:+x}" ]; then
    for e in "${KUBE_STATIC_OVERRIDES[@]}"; do [[ "$1" == *"/$e" ]] && return 0; done;
  fi
  return 1;
}

# kube::binaries_from_targets take a list of build targets and return the
# full go package to be built
kube::golang::binaries_from_targets() {
  local target
  for target; do
    # If the target starts with what looks like a domain name, assume it has a
    # fully-qualified package name rather than one that needs the Kubernetes
    # package prepended.
    if [[ "${target}" =~ ^([[:alnum:]]+".")+[[:alnum:]]+"/" ]]; then
      echo "${target}"
    else
      echo "${KUBE_GO_PACKAGE}/${target}"
    fi
  done
}

# Asks golang what it thinks the host platform is. The go tool chain does some
# slightly different things when the target platform matches the host platform.
kube::golang::host_platform() {
  echo "$(go env GOHOSTOS)/$(go env GOHOSTARCH)"
}

kube::golang::current_platform() {
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
kube::golang::set_platform_envs() {
  [[ -n ${1-} ]] || {
    kube::log::error_exit "!!! Internal error.  No platform set in kube::golang::set_platform_envs"
  }

  export GOOS=${platform%/*}
  export GOARCH=${platform##*/}

  # Dynamic CGO linking for other server architectures than linux/amd64 goes here
  # If you want to include support for more server platforms than these, add arch-specific gcc names here
  if [[ ${platform} == "linux/arm" ]]; then
    export CGO_ENABLED=1
    export CC=arm-linux-gnueabi-gcc
  elif [[ ${platform} == "linux/arm64" ]]; then
    export CGO_ENABLED=1
    export CC=aarch64-linux-gnu-gcc
  elif [[ ${platform} == "linux/ppc64le" ]]; then
    export CGO_ENABLED=1
    export CC=powerpc64le-linux-gnu-gcc
  fi
}

kube::golang::unset_platform_envs() {
  unset GOOS
  unset GOARCH
  unset CGO_ENABLED
  unset CC
}

# Create the GOPATH tree under $KUBE_OUTPUT
kube::golang::create_gopath_tree() {
  local go_pkg_dir="${KUBE_GOPATH}/src/${KUBE_GO_PACKAGE}"
  local go_pkg_basedir=$(dirname "${go_pkg_dir}")

  mkdir -p "${go_pkg_basedir}"
  rm -f "${go_pkg_dir}"

  # TODO: This symlink should be relative.
  ln -s "${KUBE_ROOT}" "${go_pkg_dir}"
}

# kube::golang::setup_env will check that the `go` commands is available in
# ${PATH}. If not running on Travis, it will also check that the Go version is
# good enough for the Kubernetes build.
#
# Inputs:
#   KUBE_EXTRA_GOPATH - If set, this is included in created GOPATH
#
# Outputs:
#   env-var GOPATH points to our local output dir
#   env-var GOBIN is unset (we want binaries in a predictable place)
#   env-var GO15VENDOREXPERIMENT=1
#   current directory is within GOPATH
kube::golang::setup_env() {
  kube::golang::create_gopath_tree

  if [[ -z "$(which go)" ]]; then
    kube::log::usage_from_stdin <<EOF
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
    if [[ "${go_version[2]}" < "go1.4" ]]; then
      kube::log::usage_from_stdin <<EOF
Detected go version: ${go_version[*]}.
Kubernetes requires go version 1.4 or greater.
Please install Go version 1.4 or later.
EOF
      exit 2
    fi
  fi

  export GOPATH=${KUBE_GOPATH}

  # Append KUBE_EXTRA_GOPATH to the GOPATH if it is defined.
  if [[ -n ${KUBE_EXTRA_GOPATH:-} ]]; then
    GOPATH="${GOPATH}:${KUBE_EXTRA_GOPATH}"
  fi

  # Change directories so that we are within the GOPATH.  Some tools get really
  # upset if this is not true.  We use a whole fake GOPATH here to collect the
  # resultant binaries.  Go will not let us use GOBIN with `go install` and
  # cross-compiling, and `go install -o <file>` only works for a single pkg.
  local subdir
  subdir=$(pwd | sed "s|$KUBE_ROOT||")
  cd "${KUBE_GOPATH}/src/${KUBE_GO_PACKAGE}/${subdir}"

  # Unset GOBIN in case it already exists in the current session.
  unset GOBIN

  # This seems to matter to some tools (godep, ugorji, ginkgo...)
  export GO15VENDOREXPERIMENT=1
}

# This will take binaries from $GOPATH/bin and copy them to the appropriate
# place in ${KUBE_OUTPUT_BINDIR}
#
# Ideally this wouldn't be necessary and we could just set GOBIN to
# KUBE_OUTPUT_BINDIR but that won't work in the face of cross compilation.  'go
# install' will place binaries that match the host platform directly in $GOBIN
# while placing cross compiled binaries into `platform_arch` subdirs.  This
# complicates pretty much everything else we do around packaging and such.
kube::golang::place_bins() {
  local host_platform
  host_platform=$(kube::golang::host_platform)

  kube::log::status "Placing binaries"

  local platform
  for platform in "${KUBE_CLIENT_PLATFORMS[@]}"; do
    # The substitution on platform_src below will replace all slashes with
    # underscores.  It'll transform darwin/amd64 -> darwin_amd64.
    local platform_src="/${platform//\//_}"
    if [[ $platform == $host_platform ]]; then
      platform_src=""
    fi

    local full_binpath_src="${KUBE_GOPATH}/bin${platform_src}"
    if [[ -d "${full_binpath_src}" ]]; then
      mkdir -p "${KUBE_OUTPUT_BINPATH}/${platform}"
      find "${full_binpath_src}" -maxdepth 1 -type f -exec \
        rsync -pt {} "${KUBE_OUTPUT_BINPATH}/${platform}" \;
    fi
  done
}

kube::golang::fallback_if_stdlib_not_installable() {
  local go_root_dir=$(go env GOROOT);
  local go_host_os=$(go env GOHOSTOS);
  local go_host_arch=$(go env GOHOSTARCH);
  local cgo_pkg_dir=${go_root_dir}/pkg/${go_host_os}_${go_host_arch}_cgo;

  if [ -e ${cgo_pkg_dir} ]; then
    return 0;
  fi

  if [ -w ${go_root_dir}/pkg ]; then
    return 0;
  fi

  kube::log::status "+++ Warning: stdlib pkg with cgo flag not found.";
  kube::log::status "+++ Warning: stdlib pkg cannot be rebuilt since ${go_root_dir}/pkg is not writable by `whoami`";
  kube::log::status "+++ Warning: Make ${go_root_dir}/pkg writable for `whoami` for a one-time stdlib install, Or"
  kube::log::status "+++ Warning: Rebuild stdlib using the command 'CGO_ENABLED=0 go install -a -installsuffix cgo std'";
  kube::log::status "+++ Falling back to go build, which is slower";

  use_go_build=true
}

# Try and replicate the native binary placement of go install without
# calling go install.
kube::golang::output_filename_for_binary() {
  local binary=$1
  local platform=$2
  local output_path="${KUBE_GOPATH}/bin"
  if [[ $platform != $host_platform ]]; then
    output_path="${output_path}/${platform//\//_}"
  fi
  local bin=$(basename "${binary}")
  if [[ ${GOOS} == "windows" ]]; then
    bin="${bin}.exe"
  fi
  echo "${output_path}/${bin}"
}

kube::golang::build_binaries_for_platform() {
  local platform=$1
  local use_go_build=${2-}

  local -a statics=()
  local -a nonstatics=()
  local -a tests=()
  for binary in "${binaries[@]}"; do
    if [[ "${binary}" =~ ".test"$ ]]; then
      tests+=($binary)
    elif kube::golang::is_statically_linked_library "${binary}"; then
      statics+=($binary)
    else
      nonstatics+=($binary)
    fi
  done
  if [[ "${#statics[@]}" != 0 ]]; then
      kube::golang::fallback_if_stdlib_not_installable;
  fi

  if [[ -n ${use_go_build:-} ]]; then
    kube::log::progress "    "
    for binary in "${statics[@]:+${statics[@]}}"; do
      local outfile=$(kube::golang::output_filename_for_binary "${binary}" "${platform}")
      CGO_ENABLED=0 go build -o "${outfile}" \
        "${goflags[@]:+${goflags[@]}}" \
        -ldflags "${goldflags}" \
        "${binary}"
      kube::log::progress "*"
    done
    for binary in "${nonstatics[@]:+${nonstatics[@]}}"; do
      local outfile=$(kube::golang::output_filename_for_binary "${binary}" "${platform}")
      go build -o "${outfile}" \
        "${goflags[@]:+${goflags[@]}}" \
        -ldflags "${goldflags}" \
        "${binary}"
      kube::log::progress "*"
    done
    kube::log::progress "\n"
  else
    # Use go install.
    if [[ "${#nonstatics[@]}" != 0 ]]; then
      go install "${goflags[@]:+${goflags[@]}}" \
        -ldflags "${goldflags}" \
        "${nonstatics[@]:+${nonstatics[@]}}"
    fi
    if [[ "${#statics[@]}" != 0 ]]; then
      CGO_ENABLED=0 go install -installsuffix cgo "${goflags[@]:+${goflags[@]}}" \
        -ldflags "${goldflags}" \
        "${statics[@]:+${statics[@]}}"
    fi
  fi

  for test in "${tests[@]:+${tests[@]}}"; do
    local outfile=$(kube::golang::output_filename_for_binary "${test}" \
      "${platform}")
    # Go 1.4 added -o to control where the binary is saved, but Go 1.3 doesn't
    # have this flag. Whenever we deprecate go 1.3, update to use -o instead of
    # changing into the output directory.
    mkdir -p "$(dirname ${outfile})"
    pushd "$(dirname ${outfile})" >/dev/null
    go test -c \
      "${goflags[@]:+${goflags[@]}}" \
      -ldflags "${goldflags}" \
      "$(dirname ${test})"
    popd >/dev/null
  done
}

# Return approximate physical memory available in gigabytes.
kube::golang::get_physmem() {
  local mem

  # Linux kernel version >=3.14, in kb
  if mem=$(grep MemAvailable /proc/meminfo | awk '{ print $2 }'); then
    echo $(( ${mem} / 1048576 ))
    return
  fi

  # Linux, in kb
  if mem=$(grep MemTotal /proc/meminfo | awk '{ print $2 }'); then
    echo $(( ${mem} / 1048576 ))
    return
  fi

  # OS X, in bytes. Note that get_physmem, as used, should only ever
  # run in a Linux container (because it's only used in the multiple
  # platform case, which is a Dockerized build), but this is provided
  # for completeness.
  if mem=$(sysctl -n hw.memsize 2>/dev/null); then
    echo $(( ${mem} / 1073741824 ))
    return
  fi

  # If we can't infer it, just give up and assume a low memory system
  echo 1
}

# Build binaries targets specified
#
# Input:
#   $@ - targets and go flags.  If no targets are set then all binaries targets
#     are built.
#   KUBE_BUILD_PLATFORMS - Incoming variable of targets to build for.  If unset
#     then just the host architecture is built.
kube::golang::build_binaries() {
  # Create a sub-shell so that we don't pollute the outer environment
  (
    # Check for `go` binary and set ${GOPATH}.
    kube::golang::setup_env
    echo "Go version: $(go version)"

    local host_platform
    host_platform=$(kube::golang::host_platform)

    # Use eval to preserve embedded quoted strings.
    local goflags goldflags
    eval "goflags=(${KUBE_GOFLAGS:-})"
    goldflags="${KUBE_GOLDFLAGS:-} $(kube::version::ldflags)"

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
      targets=("${KUBE_ALL_TARGETS[@]}")
    fi

    local -a platforms=("${KUBE_BUILD_PLATFORMS[@]:+${KUBE_BUILD_PLATFORMS[@]}}")
    if [[ ${#platforms[@]} -eq 0 ]]; then
      platforms=("${host_platform}")
    fi

    local binaries
    binaries=($(kube::golang::binaries_from_targets "${targets[@]}"))

    local parallel=false
    if [[ ${#platforms[@]} -gt 1 ]]; then
      local gigs
      gigs=$(kube::golang::get_physmem)

      if [[ ${gigs} -ge ${KUBE_PARALLEL_BUILD_MEMORY} ]]; then
        kube::log::status "Multiple platforms requested and available ${gigs}G >= threshold ${KUBE_PARALLEL_BUILD_MEMORY}G, building platforms in parallel"
        parallel=true
      else
        kube::log::status "Multiple platforms requested, but available ${gigs}G < threshold ${KUBE_PARALLEL_BUILD_MEMORY}G, building platforms in serial"
        parallel=false
      fi
    fi

    if [[ "${parallel}" == "true" ]]; then
      kube::log::status "Building go targets for ${platforms[@]} in parallel (output will appear in a burst when complete):" "${targets[@]}"
      local platform
      for platform in "${platforms[@]}"; do (
          kube::golang::set_platform_envs "${platform}"
          kube::log::status "${platform}: go build started"
          kube::golang::build_binaries_for_platform ${platform} ${use_go_build:-}
          kube::log::status "${platform}: go build finished"
        ) &> "/tmp//${platform//\//_}.build" &
      done

      local fails=0
      for job in $(jobs -p); do
        wait ${job} || let "fails+=1"
      done

      for platform in "${platforms[@]}"; do
        cat "/tmp//${platform//\//_}.build"
      done

      exit ${fails}
    else
      for platform in "${platforms[@]}"; do
        kube::log::status "Building go targets for ${platform}:" "${targets[@]}"
        kube::golang::set_platform_envs "${platform}"
        kube::golang::build_binaries_for_platform ${platform} ${use_go_build:-}
      done
    fi
  )
}
