#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

# The set of server targets that we are only building for Linux
# If you update this list, please also update build/BUILD.
kube::golang::server_targets() {
  local targets=(
    cmd/kube-proxy
    cmd/kube-apiserver
    cmd/kube-controller-manager
    cmd/cloud-controller-manager
    cmd/kubelet
    cmd/kubeadm
    cmd/hyperkube
    vendor/k8s.io/kube-aggregator
    vendor/k8s.io/apiextensions-apiserver
    plugin/cmd/kube-scheduler
    cluster/gce/gci/mounter
  )
  echo "${targets[@]}"
}

readonly KUBE_SERVER_TARGETS=($(kube::golang::server_targets))
readonly KUBE_SERVER_BINARIES=("${KUBE_SERVER_TARGETS[@]##*/}")

# The set of server targets that we are only building for Kubernetes nodes
# If you update this list, please also update build/BUILD.
kube::golang::node_targets() {
  local targets=(
    cmd/kube-proxy
    cmd/kubeadm
    cmd/kubelet
  )
  echo "${targets[@]}"
}

readonly KUBE_NODE_TARGETS=($(kube::golang::node_targets))
readonly KUBE_NODE_BINARIES=("${KUBE_NODE_TARGETS[@]##*/}")
readonly KUBE_NODE_BINARIES_WIN=("${KUBE_NODE_BINARIES[@]/%/.exe}")

if [[ -n "${KUBE_BUILD_PLATFORMS:-}" ]]; then
  readonly KUBE_SERVER_PLATFORMS=(${KUBE_BUILD_PLATFORMS})
  readonly KUBE_NODE_PLATFORMS=(${KUBE_BUILD_PLATFORMS})
  readonly KUBE_TEST_PLATFORMS=(${KUBE_BUILD_PLATFORMS})
  readonly KUBE_CLIENT_PLATFORMS=(${KUBE_BUILD_PLATFORMS})
elif [[ "${KUBE_FASTBUILD:-}" == "true" ]]; then
  readonly KUBE_SERVER_PLATFORMS=(linux/amd64)
  readonly KUBE_NODE_PLATFORMS=(linux/amd64)
  if [[ "${KUBE_BUILDER_OS:-}" == "darwin"* ]]; then
    readonly KUBE_TEST_PLATFORMS=(
      darwin/amd64
      linux/amd64
    )
    readonly KUBE_CLIENT_PLATFORMS=(
      darwin/amd64
      linux/amd64
    )
  else
    readonly KUBE_TEST_PLATFORMS=(linux/amd64)
    readonly KUBE_CLIENT_PLATFORMS=(linux/amd64)
  fi
else

  # The server platform we are building on.
  readonly KUBE_SERVER_PLATFORMS=(
    linux/amd64
    linux/arm
    linux/arm64
    linux/s390x
    linux/ppc64le
  )

  # The node platforms we build for
  readonly KUBE_NODE_PLATFORMS=(
    linux/amd64
    linux/arm
    linux/arm64
    linux/s390x
    linux/ppc64le
    windows/amd64
  )

  # If we update this we should also update the set of platforms whose standard library is precompiled for in build/build-image/cross/Dockerfile
  readonly KUBE_CLIENT_PLATFORMS=(
    linux/amd64
    linux/386
    linux/arm
    linux/arm64
    linux/s390x
    linux/ppc64le
    darwin/amd64
    darwin/386
    windows/amd64
    windows/386
  )

  # Which platforms we should compile test targets for. Not all client platforms need these tests
  readonly KUBE_TEST_PLATFORMS=(
    linux/amd64
    linux/arm
    linux/arm64
    linux/s390x
    linux/ppc64le
    darwin/amd64
    windows/amd64
  )
fi

# The set of client targets that we are building for all platforms
# If you update this list, please also update build/BUILD.
readonly KUBE_CLIENT_TARGETS=(
  cmd/kubectl
)
readonly KUBE_CLIENT_BINARIES=("${KUBE_CLIENT_TARGETS[@]##*/}")
readonly KUBE_CLIENT_BINARIES_WIN=("${KUBE_CLIENT_BINARIES[@]/%/.exe}")

# The set of test targets that we are building for all platforms
# If you update this list, please also update build/BUILD.
kube::golang::test_targets() {
  local targets=(
    cmd/gendocs
    cmd/genkubedocs
    cmd/genman
    cmd/genyaml
    cmd/genswaggertypedocs
    cmd/linkcheck
    vendor/github.com/onsi/ginkgo/ginkgo
    test/e2e/e2e.test
  )
  echo "${targets[@]}"
}
readonly KUBE_TEST_TARGETS=($(kube::golang::test_targets))
readonly KUBE_TEST_BINARIES=("${KUBE_TEST_TARGETS[@]##*/}")
readonly KUBE_TEST_BINARIES_WIN=("${KUBE_TEST_BINARIES[@]/%/.exe}")
# If you update this list, please also update build/BUILD.
readonly KUBE_TEST_PORTABLE=(
  test/e2e/testing-manifests
  test/kubemark
  hack/e2e.go
  hack/e2e-internal
  hack/get-build.sh
  hack/ginkgo-e2e.sh
  hack/lib
)

# Test targets which run on the Kubernetes clusters directly, so we only
# need to target server platforms.
# These binaries will be distributed in the kubernetes-test tarball.
# If you update this list, please also update build/BUILD.
kube::golang::server_test_targets() {
  local targets=(
    cmd/kubemark
    vendor/github.com/onsi/ginkgo/ginkgo
  )

  if [[ "${OSTYPE:-}" == "linux"* ]]; then
    targets+=( test/e2e_node/e2e_node.test )
  fi

  echo "${targets[@]}"
}

readonly KUBE_TEST_SERVER_TARGETS=($(kube::golang::server_test_targets))
readonly KUBE_TEST_SERVER_BINARIES=("${KUBE_TEST_SERVER_TARGETS[@]##*/}")
readonly KUBE_TEST_SERVER_PLATFORMS=("${KUBE_SERVER_PLATFORMS[@]}")

# Gigabytes desired for parallel platform builds. 11 is fairly
# arbitrary, but is a reasonable splitting point for 2015
# laptops-versus-not.
readonly KUBE_PARALLEL_BUILD_MEMORY=11

# TODO(pipejakob) gke-certificates-controller is included here to exercise its
# compilation, but it doesn't need to be distributed in any of our tars. Its
# code is only living in this repo temporarily until it finds a new home.
readonly KUBE_ALL_TARGETS=(
  "${KUBE_SERVER_TARGETS[@]}"
  "${KUBE_CLIENT_TARGETS[@]}"
  "${KUBE_TEST_TARGETS[@]}"
  "${KUBE_TEST_SERVER_TARGETS[@]}"
  cmd/gke-certificates-controller
)
readonly KUBE_ALL_BINARIES=("${KUBE_ALL_TARGETS[@]##*/}")

readonly KUBE_STATIC_LIBRARIES=(
  cloud-controller-manager
  kube-apiserver
  kube-controller-manager
  kube-scheduler
  kube-proxy
  kube-aggregator
  kubeadm
  kubectl
)

# Add any files with those //generate annotations in the array below.
readonly KUBE_BINDATAS=(
  test/e2e/generated/gobindata_util.go
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

# Takes the platform name ($1) and sets the appropriate golang env variables
# for that platform.
kube::golang::set_platform_envs() {
  [[ -n ${1-} ]] || {
    kube::log::error_exit "!!! Internal error. No platform set in kube::golang::set_platform_envs"
  }

  export GOOS=${platform%/*}
  export GOARCH=${platform##*/}

  # Do not set CC when building natively on a platform, only if cross-compiling from linux/amd64
  if [[ $(kube::golang::host_platform) == "linux/amd64" ]]; then
    # Dynamic CGO linking for other server architectures than linux/amd64 goes here
    # If you want to include support for more server platforms than these, add arch-specific gcc names here
    case "${platform}" in
      "linux/arm")
        export CGO_ENABLED=1
        export CC=arm-linux-gnueabihf-gcc
        ;;
      "linux/arm64")
        export CGO_ENABLED=1
        export CC=aarch64-linux-gnu-gcc
        ;;
      "linux/ppc64le")
        export CGO_ENABLED=1
        export CC=powerpc64le-linux-gnu-gcc
        ;;
      "linux/s390x")
        export CGO_ENABLED=1
        export CC=s390x-linux-gnu-gcc
        ;;
    esac
  fi
}

kube::golang::unset_platform_envs() {
  unset GOOS
  unset GOARCH
  unset GOROOT
  unset CGO_ENABLED
  unset CC
}

# Create the GOPATH tree under $KUBE_OUTPUT
kube::golang::create_gopath_tree() {
  local go_pkg_dir="${KUBE_GOPATH}/src/${KUBE_GO_PACKAGE}"
  local go_pkg_basedir=$(dirname "${go_pkg_dir}")

  mkdir -p "${go_pkg_basedir}"

  # TODO: This symlink should be relative.
  if [[ ! -e "${go_pkg_dir}" || "$(readlink ${go_pkg_dir})" != "${KUBE_ROOT}" ]]; then
    ln -snf "${KUBE_ROOT}" "${go_pkg_dir}"
  fi

  cat >"${KUBE_GOPATH}/BUILD" <<EOF
# This dummy BUILD file prevents Bazel from trying to descend through the
# infinite loop created by the symlink at
# ${go_pkg_dir}
EOF
}

# Ensure the go tool exists and is a viable version.
kube::golang::verify_go_version() {
  if [[ -z "$(which go)" ]]; then
    kube::log::usage_from_stdin <<EOF
Can't find 'go' in PATH, please fix and retry.
See http://golang.org/doc/install for installation instructions.
EOF
    return 2
  fi

  local go_version
  go_version=($(go version))
  local minimum_go_version
  minimum_go_version=go1.9.1
  if [[ "${go_version[2]}" < "${minimum_go_version}" && "${go_version[2]}" != "devel" ]]; then
    kube::log::usage_from_stdin <<EOF
Detected go version: ${go_version[*]}.
Kubernetes requires ${minimum_go_version} or greater.
Please install ${minimum_go_version} or later.
EOF
    return 2
  fi
}

# kube::golang::setup_env will check that the `go` commands is available in
# ${PATH}. It will also check that the Go version is good enough for the
# Kubernetes build.
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
  kube::golang::verify_go_version

  kube::golang::create_gopath_tree

  export GOPATH="${KUBE_GOPATH}"

  # Append KUBE_EXTRA_GOPATH to the GOPATH if it is defined.
  if [[ -n ${KUBE_EXTRA_GOPATH:-} ]]; then
    GOPATH="${GOPATH}:${KUBE_EXTRA_GOPATH}"
  fi

  # Make sure our own Go binaries are in PATH.
  export PATH="${KUBE_GOPATH}/bin:${PATH}"

  # Change directories so that we are within the GOPATH.  Some tools get really
  # upset if this is not true.  We use a whole fake GOPATH here to collect the
  # resultant binaries.  Go will not let us use GOBIN with `go install` and
  # cross-compiling, and `go install -o <file>` only works for a single pkg.
  local subdir
  subdir=$(kube::realpath . | sed "s|$KUBE_ROOT||")
  cd "${KUBE_GOPATH}/src/${KUBE_GO_PACKAGE}/${subdir}"

  # Set GOROOT so binaries that parse code can work properly.
  export GOROOT=$(go env GOROOT)

  # Unset GOBIN in case it already exists in the current session.
  unset GOBIN

  # This seems to matter to some tools (godep, ginkgo...)
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

  V=2 kube::log::status "Placing binaries"

  local platform
  for platform in "${KUBE_CLIENT_PLATFORMS[@]}"; do
    # The substitution on platform_src below will replace all slashes with
    # underscores.  It'll transform darwin/amd64 -> darwin_amd64.
    local platform_src="/${platform//\//_}"
    if [[ $platform == $host_platform ]]; then
      platform_src=""
      rm -f "${THIS_PLATFORM_BIN}"
      ln -s "${KUBE_OUTPUT_BINPATH}/${platform}" "${THIS_PLATFORM_BIN}"
    fi

    local full_binpath_src="${KUBE_GOPATH}/bin${platform_src}"
    if [[ -d "${full_binpath_src}" ]]; then
      mkdir -p "${KUBE_OUTPUT_BINPATH}/${platform}"
      find "${full_binpath_src}" -maxdepth 1 -type f -exec \
        rsync -pc {} "${KUBE_OUTPUT_BINPATH}/${platform}" \;
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

# Builds the toolchain necessary for building kube. This needs to be
# built only on the host platform.
# TODO: Find this a proper home.
# Ideally, not a shell script because testing shell scripts is painful.
kube::golang::build_kube_toolchain() {
  local targets=(
    hack/cmd/teststale
    vendor/github.com/jteeuwen/go-bindata/go-bindata
  )

  local binaries
  binaries=($(kube::golang::binaries_from_targets "${targets[@]}"))

  kube::log::status "Building the toolchain targets:" "${binaries[@]}"
  go install "${goflags[@]:+${goflags[@]}}" \
        -gcflags "${gogcflags}" \
        -ldflags "${goldflags}" \
        "${binaries[@]:+${binaries[@]}}"
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

  V=2 kube::log::info "Env for ${platform}: GOOS=${GOOS-} GOARCH=${GOARCH-} GOROOT=${GOROOT-} CGO_ENABLED=${CGO_ENABLED-} CC=${CC-}"

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
        -gcflags "${gogcflags}" \
        -ldflags "${goldflags}" \
        "${binary}"
      kube::log::progress "*"
    done
    for binary in "${nonstatics[@]:+${nonstatics[@]}}"; do
      local outfile=$(kube::golang::output_filename_for_binary "${binary}" "${platform}")
      go build -o "${outfile}" \
        "${goflags[@]:+${goflags[@]}}" \
        -gcflags "${gogcflags}" \
        -ldflags "${goldflags}" \
        "${binary}"
      kube::log::progress "*"
    done
    kube::log::progress "\n"
  else
    # Use go install.
    if [[ "${#nonstatics[@]}" != 0 ]]; then
      go install "${goflags[@]:+${goflags[@]}}" \
        -gcflags "${gogcflags}" \
        -ldflags "${goldflags}" \
        "${nonstatics[@]:+${nonstatics[@]}}"
    fi
    if [[ "${#statics[@]}" != 0 ]]; then
      CGO_ENABLED=0 go install -installsuffix cgo "${goflags[@]:+${goflags[@]}}" \
        -gcflags "${gogcflags}" \
        -ldflags "${goldflags}" \
        "${statics[@]:+${statics[@]}}"
    fi
  fi

  for test in "${tests[@]:+${tests[@]}}"; do
    local outfile=$(kube::golang::output_filename_for_binary "${test}" \
      "${platform}")

    local testpkg="$(dirname ${test})"

    # Staleness check always happens on the host machine, so we don't
    # have to locate the `teststale` binaries for the other platforms.
    # Since we place the host binaries in `$KUBE_GOPATH/bin`, we can
    # assume that the binary exists there, if it exists at all.
    # Otherwise, something has gone wrong with building the `teststale`
    # binary and we should safely proceed building the test binaries
    # assuming that they are stale. There is no good reason to error
    # out.
    if test -x "${KUBE_GOPATH}/bin/teststale" && ! "${KUBE_GOPATH}/bin/teststale" -binary "${outfile}" -package "${testpkg}"
    then
      continue
    fi

    # `go test -c` below directly builds the binary. It builds the packages,
    # but it never installs them. `go test -i` only installs the dependencies
    # of the test, but not the test package itself. So neither `go test -c`
    # nor `go test -i` installs, for example, test/e2e.a. And without that,
    # doing a staleness check on k8s.io/kubernetes/test/e2e package always
    # returns true (always stale). And that's why we need to install the
    # test package.
    go install "${goflags[@]:+${goflags[@]}}" \
        -gcflags "${gogcflags}" \
        -ldflags "${goldflags}" \
        "${testpkg}"

    mkdir -p "$(dirname ${outfile})"
    go test -i -c \
      "${goflags[@]:+${goflags[@]}}" \
      -gcflags "${gogcflags}" \
      -ldflags "${goldflags}" \
      -o "${outfile}" \
      "${testpkg}"
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
    V=2 kube::log::info "Go version: $(go version)"

    local host_platform
    host_platform=$(kube::golang::host_platform)

    # Use eval to preserve embedded quoted strings.
    local goflags goldflags gogcflags
    eval "goflags=(${GOFLAGS:-})"
    goldflags="${GOLDFLAGS:-} $(kube::version::ldflags)"
    gogcflags="${GOGCFLAGS:-}"

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

    local -a platforms=(${KUBE_BUILD_PLATFORMS:-})
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

    # First build the toolchain before building any other targets
    kube::golang::build_kube_toolchain

    kube::log::status "Generating bindata:" "${KUBE_BINDATAS[@]}"
    for bindata in ${KUBE_BINDATAS[@]}; do
      # Only try to generate bindata if the file exists, since in some cases
      # one-off builds of individual directories may exclude some files.
      if [[ -f "${KUBE_ROOT}/${bindata}" ]]; then
        go generate "${goflags[@]:+${goflags[@]}}" "${KUBE_ROOT}/${bindata}"
      fi
    done

    if [[ "${parallel}" == "true" ]]; then
      kube::log::status "Building go targets for {${platforms[*]}} in parallel (output will appear in a burst when complete):" "${targets[@]}"
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
        (
          kube::golang::set_platform_envs "${platform}"
          kube::golang::build_binaries_for_platform ${platform} ${use_go_build:-}
        )
      done
    fi
  )
}
