#!/usr/bin/env bash

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

# The server platform we are building on.
readonly KUBE_SUPPORTED_SERVER_PLATFORMS=(
  linux/amd64
  linux/arm
  linux/arm64
  linux/s390x
  linux/ppc64le
)

# The node platforms we build for
readonly KUBE_SUPPORTED_NODE_PLATFORMS=(
  linux/amd64
  linux/arm
  linux/arm64
  linux/s390x
  linux/ppc64le
  windows/amd64
)

# If we update this we should also update the set of platforms whose standard
# library is precompiled for in build/build-image/cross/Dockerfile
readonly KUBE_SUPPORTED_CLIENT_PLATFORMS=(
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

# Which platforms we should compile test targets for.
# Not all client platforms need these tests
readonly KUBE_SUPPORTED_TEST_PLATFORMS=(
  linux/amd64
  linux/arm
  linux/arm64
  linux/s390x
  linux/ppc64le
  darwin/amd64
  windows/amd64
)

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
    cmd/kube-scheduler
    vendor/k8s.io/apiextensions-apiserver
    cluster/gce/gci/mounter
  )
  echo "${targets[@]}"
}

IFS=" " read -ra KUBE_SERVER_TARGETS <<< "$(kube::golang::server_targets)"
readonly KUBE_SERVER_TARGETS
readonly KUBE_SERVER_BINARIES=("${KUBE_SERVER_TARGETS[@]##*/}")

# The set of server targets we build docker images for
kube::golang::server_image_targets() {
  # NOTE: this contains cmd targets for kube::build::get_docker_wrapped_binaries
  local targets=(
    cmd/cloud-controller-manager
    cmd/kube-apiserver
    cmd/kube-controller-manager
    cmd/kube-scheduler
    cmd/kube-proxy
  )
  echo "${targets[@]}"
}

IFS=" " read -ra KUBE_SERVER_IMAGE_TARGETS <<< "$(kube::golang::server_image_targets)"
readonly KUBE_SERVER_IMAGE_TARGETS
readonly KUBE_SERVER_IMAGE_BINARIES=("${KUBE_SERVER_IMAGE_TARGETS[@]##*/}")

# The set of conformance targets we build docker image for
kube::golang::conformance_image_targets() {
  # NOTE: this contains cmd targets for kube::release::build_conformance_image
  local targets=(
    vendor/github.com/onsi/ginkgo/ginkgo
    test/e2e/e2e.test
    cmd/kubectl
  )
  echo "${targets[@]}"
}

IFS=" " read -ra KUBE_CONFORMANCE_IMAGE_TARGETS <<< "$(kube::golang::conformance_image_targets)"
readonly KUBE_CONFORMANCE_IMAGE_TARGETS

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

IFS=" " read -ra KUBE_NODE_TARGETS <<< "$(kube::golang::node_targets)"
readonly KUBE_NODE_TARGETS
readonly KUBE_NODE_BINARIES=("${KUBE_NODE_TARGETS[@]##*/}")
readonly KUBE_NODE_BINARIES_WIN=("${KUBE_NODE_BINARIES[@]/%/.exe}")

# ------------
# NOTE: All functions that return lists should use newlines.
# bash functions can't return arrays, and spaces are tricky, so newline
# separators are the preferred pattern.
# To transform a string of newline-separated items to an array, use mapfile -t:
# mapfile -t FOO <<< "$(kube::golang::dups a b c a)"
#
# ALWAYS remember to quote your subshells. Not doing so will break in
# bash 4.3, and potentially cause other issues.
# ------------

# Returns a sorted newline-separated list containing only duplicated items.
kube::golang::dups() {
  # We use printf to insert newlines, which are required by sort.
  printf "%s\n" "$@" | sort | uniq -d
}

# Returns a sorted newline-separated list with duplicated items removed.
kube::golang::dedup() {
  # We use printf to insert newlines, which are required by sort.
  printf "%s\n" "$@" | sort -u
}

# Depends on values of user-facing KUBE_BUILD_PLATFORMS, KUBE_FASTBUILD,
# and KUBE_BUILDER_OS.
# Configures KUBE_SERVER_PLATFORMS, KUBE_NODE_PLATFOMRS,
# KUBE_TEST_PLATFORMS, and KUBE_CLIENT_PLATFORMS, then sets them
# to readonly.
# The configured vars will only contain platforms allowed by the
# KUBE_SUPPORTED* vars at the top of this file.
kube::golang::setup_platforms() {
  if [[ -n "${KUBE_BUILD_PLATFORMS:-}" ]]; then
    # KUBE_BUILD_PLATFORMS needs to be read into an array before the next
    # step, or quoting treats it all as one element.
    local -a platforms
    IFS=" " read -ra platforms <<< "${KUBE_BUILD_PLATFORMS}"

    # Deduplicate to ensure the intersection trick with kube::golang::dups
    # is not defeated by duplicates in user input.
    mapfile -t platforms <<< "$(kube::golang::dedup "${platforms[@]}")"

    # Use kube::golang::dups to restrict the builds to the platforms in
    # KUBE_SUPPORTED_*_PLATFORMS. Items should only appear at most once in each
    # set, so if they appear twice after the merge they are in the intersection.
    mapfile -t KUBE_SERVER_PLATFORMS <<< "$(kube::golang::dups \
        "${platforms[@]}" \
        "${KUBE_SUPPORTED_SERVER_PLATFORMS[@]}" \
      )"
    readonly KUBE_SERVER_PLATFORMS

    mapfile -t KUBE_NODE_PLATFORMS <<< "$(kube::golang::dups \
        "${platforms[@]}" \
        "${KUBE_SUPPORTED_NODE_PLATFORMS[@]}" \
      )"
    readonly KUBE_NODE_PLATFORMS

    mapfile -t KUBE_TEST_PLATFORMS <<< "$(kube::golang::dups \
        "${platforms[@]}" \
        "${KUBE_SUPPORTED_TEST_PLATFORMS[@]}" \
      )"
    readonly KUBE_TEST_PLATFORMS

    mapfile -t KUBE_CLIENT_PLATFORMS <<< "$(kube::golang::dups \
        "${platforms[@]}" \
        "${KUBE_SUPPORTED_CLIENT_PLATFORMS[@]}" \
      )"
    readonly KUBE_CLIENT_PLATFORMS

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
    KUBE_SERVER_PLATFORMS=("${KUBE_SUPPORTED_SERVER_PLATFORMS[@]}")
    readonly KUBE_SERVER_PLATFORMS

    KUBE_NODE_PLATFORMS=("${KUBE_SUPPORTED_NODE_PLATFORMS[@]}")
    readonly KUBE_NODE_PLATFORMS

    KUBE_CLIENT_PLATFORMS=("${KUBE_SUPPORTED_CLIENT_PLATFORMS[@]}")
    readonly KUBE_CLIENT_PLATFORMS

    KUBE_TEST_PLATFORMS=("${KUBE_SUPPORTED_TEST_PLATFORMS[@]}")
    readonly KUBE_TEST_PLATFORMS
  fi
}

kube::golang::setup_platforms

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
IFS=" " read -ra KUBE_TEST_TARGETS <<< "$(kube::golang::test_targets)"
readonly KUBE_TEST_TARGETS
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

IFS=" " read -ra KUBE_TEST_SERVER_TARGETS <<< "$(kube::golang::server_test_targets)"
readonly KUBE_TEST_SERVER_TARGETS
readonly KUBE_TEST_SERVER_BINARIES=("${KUBE_TEST_SERVER_TARGETS[@]##*/}")
readonly KUBE_TEST_SERVER_PLATFORMS=("${KUBE_SERVER_PLATFORMS[@]}")

# Gigabytes necessary for parallel platform builds.
# As of January 2018, RAM usage is exceeding 30G
# Setting to 40 to provide some headroom
readonly KUBE_PARALLEL_BUILD_MEMORY=40

readonly KUBE_ALL_TARGETS=(
  "${KUBE_SERVER_TARGETS[@]}"
  "${KUBE_CLIENT_TARGETS[@]}"
  "${KUBE_TEST_TARGETS[@]}"
  "${KUBE_TEST_SERVER_TARGETS[@]}"
)
readonly KUBE_ALL_BINARIES=("${KUBE_ALL_TARGETS[@]##*/}")

readonly KUBE_STATIC_LIBRARIES=(
  cloud-controller-manager
  kube-apiserver
  kube-controller-manager
  kube-scheduler
  kube-proxy
  kubeadm
  kubectl
)

# Fully-qualified package names that we want to instrument for coverage information.
readonly KUBE_COVERAGE_INSTRUMENTED_PACKAGES=(
  k8s.io/kubernetes/cmd/kube-apiserver
  k8s.io/kubernetes/cmd/kube-controller-manager
  k8s.io/kubernetes/cmd/kube-scheduler
  k8s.io/kubernetes/cmd/kube-proxy
  k8s.io/kubernetes/cmd/kubelet
)

# KUBE_CGO_OVERRIDES is a space-separated list of binaries which should be built
# with CGO enabled, assuming CGO is supported on the target platform.
# This overrides any entry in KUBE_STATIC_LIBRARIES.
IFS=" " read -ra KUBE_CGO_OVERRIDES <<< "${KUBE_CGO_OVERRIDES:-}"
readonly KUBE_CGO_OVERRIDES
# KUBE_STATIC_OVERRIDES is a space-separated list of binaries which should be
# built with CGO disabled. This is in addition to the list in
# KUBE_STATIC_LIBRARIES.
IFS=" " read -ra KUBE_STATIC_OVERRIDES <<< "${KUBE_STATIC_OVERRIDES:-}"
readonly KUBE_STATIC_OVERRIDES

kube::golang::is_statically_linked_library() {
  local e
  # Explicitly enable cgo when building kubectl for darwin from darwin.
  [[ "$(go env GOHOSTOS)" == "darwin" && "$(go env GOOS)" == "darwin" &&
    "$1" == *"/kubectl" ]] && return 1
  if [[ -n "${KUBE_CGO_OVERRIDES:+x}" ]]; then
    for e in "${KUBE_CGO_OVERRIDES[@]}"; do [[ "${1}" == *"/${e}" ]] && return 1; done;
  fi
  for e in "${KUBE_STATIC_LIBRARIES[@]}"; do [[ "${1}" == *"/${e}" ]] && return 0; done;
  if [[ -n "${KUBE_STATIC_OVERRIDES:+x}" ]]; then
    for e in "${KUBE_STATIC_OVERRIDES[@]}"; do [[ "${1}" == *"/${e}" ]] && return 0; done;
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

  # Using bazel with a recursive target (e.g. bazel test ...) will abort due to
  # the symlink loop created in this function, so create this special file which
  # tells bazel not to follow the symlink.
  touch "${go_pkg_basedir}/DONT_FOLLOW_SYMLINKS_WHEN_TRAVERSING_THIS_DIRECTORY_VIA_A_RECURSIVE_TARGET_PATTERN"
  # Additionally, the //:package-srcs glob recursively includes all
  # subdirectories, and similarly fails due to the symlink loop. By creating a
  # BUILD.bazel file, we effectively create a dummy package, which stops the
  # glob from descending further into the tree and hitting the loop.
  cat >"${KUBE_GOPATH}/BUILD.bazel" <<EOF
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
  IFS=" " read -ra go_version <<< "$(go version)"
  local minimum_go_version
  minimum_go_version=go1.12.1
  if [[ "${minimum_go_version}" != $(echo -e "${minimum_go_version}\n${go_version[2]}" | sort -s -t. -k 1,1 -k 2,2n -k 3,3n | head -n1) && "${go_version[2]}" != "devel" ]]; then
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
  export GOCACHE="${KUBE_GOPATH}/cache"

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
  subdir=$(kube::realpath . | sed "s|${KUBE_ROOT}||")
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
    if [[ "${platform}" == "${host_platform}" ]]; then
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

# Try and replicate the native binary placement of go install without
# calling go install.
kube::golang::outfile_for_binary() {
  local binary=$1
  local platform=$2
  local output_path="${KUBE_GOPATH}/bin"
  if [[ "${platform}" != "${host_platform}" ]]; then
    output_path="${output_path}/${platform//\//_}"
  fi
  local bin=$(basename "${binary}")
  if [[ ${GOOS} == "windows" ]]; then
    bin="${bin}.exe"
  fi
  echo "${output_path}/${bin}"
}

# Argument: the name of a Kubernetes package.
# Returns 0 if the binary can be built with coverage, 1 otherwise.
# NB: this ignores whether coverage is globally enabled or not.
kube::golang::is_instrumented_package() {
  return $(kube::util::array_contains "$1" "${KUBE_COVERAGE_INSTRUMENTED_PACKAGES[@]}")
}

# Argument: the name of a Kubernetes package (e.g. k8s.io/kubernetes/cmd/kube-scheduler)
# Echos the path to a dummy test used for coverage information.
kube::golang::path_for_coverage_dummy_test() {
  local package="$1"
  local path="${KUBE_GOPATH}/src/${package}"
  local name=$(basename "${package}")
  echo "${path}/zz_generated_${name}_test.go"
}

# Argument: the name of a Kubernetes package (e.g. k8s.io/kubernetes/cmd/kube-scheduler).
# Creates a dummy unit test on disk in the source directory for the given package.
# This unit test will invoke the package's standard entry point when run.
kube::golang::create_coverage_dummy_test() {
  local package="$1"
  local name="$(basename "${package}")"
  cat <<EOF > $(kube::golang::path_for_coverage_dummy_test "${package}")
package main
import (
  "testing"
  "k8s.io/kubernetes/pkg/util/coverage"
)

func TestMain(m *testing.M) {
  // Get coverage running
  coverage.InitCoverage("${name}")

  // Go!
  main()

  // Make sure we actually write the profiling information to disk, if we make it here.
  // On long-running services, or anything that calls os.Exit(), this is insufficient,
  // so we also flush periodically with a default period of five seconds (configurable by
  // the KUBE_COVERAGE_FLUSH_INTERVAL environment variable).
  coverage.FlushCoverage()
}
EOF
}

# Argument: the name of a Kubernetes package (e.g. k8s.io/kubernetes/cmd/kube-scheduler).
# Deletes a test generated by kube::golang::create_coverage_dummy_test.
# It is not an error to call this for a nonexistent test.
kube::golang::delete_coverage_dummy_test() {
  local package="$1"
  rm -f $(kube::golang::path_for_coverage_dummy_test "${package}")
}

# Arguments: a list of kubernetes packages to build.
# Expected variables: ${build_args} should be set to an array of Go build arguments.
# In addition, ${package} and ${platform} should have been set earlier, and if
# ${KUBE_BUILD_WITH_COVERAGE} is set, coverage instrumentation will be enabled.
#
# Invokes Go to actually build some packages. If coverage is disabled, simply invokes
# go install. If coverage is enabled, builds covered binaries using go test, temporarily
# producing the required unit test files and then cleaning up after itself.
# Non-covered binaries are then built using go install as usual.
kube::golang::build_some_binaries() {
  if [[ -n "${KUBE_BUILD_WITH_COVERAGE:-}" ]]; then
    local -a uncovered=()
    for package in "$@"; do
      if kube::golang::is_instrumented_package "${package}"; then
        V=2 kube::log::info "Building ${package} with coverage..."

        kube::golang::create_coverage_dummy_test "${package}"
        kube::util::trap_add "kube::golang::delete_coverage_dummy_test \"${package}\"" EXIT

        go test -c -o "$(kube::golang::outfile_for_binary "${package}" "${platform}")" \
          -covermode count \
          -coverpkg k8s.io/...,k8s.io/kubernetes/vendor/k8s.io/... \
          "${build_args[@]}" \
          -tags coverage \
          "${package}"
      else
        uncovered+=("${package}")
      fi
    done
    if [[ "${#uncovered[@]}" != 0 ]]; then
      V=2 kube::log::info "Building ${uncovered[@]} without coverage..."
      go install "${build_args[@]}" "${uncovered[@]}"
    else
      V=2 kube::log::info "Nothing to build without coverage."
     fi
   else
    V=2 kube::log::info "Coverage is disabled."
    go install "${build_args[@]}" "$@"
   fi
}

kube::golang::build_binaries_for_platform() {
  local platform=$1

  local -a statics=()
  local -a nonstatics=()
  local -a tests=()

  V=2 kube::log::info "Env for ${platform}: GOOS=${GOOS-} GOARCH=${GOARCH-} GOROOT=${GOROOT-} CGO_ENABLED=${CGO_ENABLED-} CC=${CC-}"

  for binary in "${binaries[@]}"; do
    if [[ "${binary}" =~ ".test"$ ]]; then
      tests+=(${binary})
    elif kube::golang::is_statically_linked_library "${binary}"; then
      statics+=(${binary})
    else
      nonstatics+=(${binary})
    fi
  done

  local -a build_args
  if [[ "${#statics[@]}" != 0 ]]; then
    build_args=(
      -installsuffix static
      ${goflags:+"${goflags[@]}"}
      -gcflags "${gogcflags:-}"
      -asmflags "${goasmflags:-}"
      -ldflags "${goldflags:-}"
    )
    CGO_ENABLED=0 kube::golang::build_some_binaries "${statics[@]}"
  fi

  if [[ "${#nonstatics[@]}" != 0 ]]; then
    build_args=(
      ${goflags:+"${goflags[@]}"}
      -gcflags "${gogcflags:-}"
      -asmflags "${goasmflags:-}"
      -ldflags "${goldflags:-}"
    )
    kube::golang::build_some_binaries "${nonstatics[@]}"
  fi

  for test in "${tests[@]:+${tests[@]}}"; do
    local outfile=$(kube::golang::outfile_for_binary "${test}" "${platform}")
    local testpkg="$(dirname ${test})"

    mkdir -p "$(dirname ${outfile})"
    go test -c \
      ${goflags:+"${goflags[@]}"} \
      -gcflags "${gogcflags:-}" \
      -asmflags "${goasmflags:-}" \
      -ldflags "${goldflags:-}" \
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

    local goflags goldflags goasmflags gogcflags
    goldflags="${GOLDFLAGS:-} -s -w $(kube::version::ldflags)"
    goasmflags="-trimpath=${KUBE_ROOT}"
    gogcflags="${GOGCFLAGS:-} -trimpath=${KUBE_ROOT}"

    local -a targets=()
    local arg

    for arg; do
      if [[ "${arg}" == -* ]]; then
        # Assume arguments starting with a dash are flags to pass to go.
        goflags+=("${arg}")
      else
        targets+=("${arg}")
      fi
    done

    if [[ ${#targets[@]} -eq 0 ]]; then
      targets=("${KUBE_ALL_TARGETS[@]}")
    fi

    local -a platforms
    IFS=" " read -ra platforms <<< "${KUBE_BUILD_PLATFORMS:-}"
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
      kube::log::status "Building go targets for {${platforms[*]}} in parallel (output will appear in a burst when complete):" "${targets[@]}"
      local platform
      for platform in "${platforms[@]}"; do (
          kube::golang::set_platform_envs "${platform}"
          kube::log::status "${platform}: build started"
          kube::golang::build_binaries_for_platform ${platform}
          kube::log::status "${platform}: build finished"
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
          kube::golang::build_binaries_for_platform ${platform}
        )
      done
    fi
  )
}
