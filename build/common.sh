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

# Common utilities, variables and checks for all build scripts.
set -o errexit
set -o nounset
set -o pipefail

DOCKER_OPTS=${DOCKER_OPTS:-""}
DOCKER_NATIVE=${DOCKER_NATIVE:-""}
DOCKER=(docker ${DOCKER_OPTS})
DOCKER_HOST=${DOCKER_HOST:-""}
DOCKER_MACHINE_NAME=${DOCKER_MACHINE_NAME:-"kube-dev"}
readonly DOCKER_MACHINE_DRIVER=${DOCKER_MACHINE_DRIVER:-"virtualbox --virtualbox-memory 4096 --virtualbox-cpu-count -1"}

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
cd "${KUBE_ROOT}"

# This'll canonicalize the path
KUBE_ROOT=$PWD

source hack/lib/init.sh

# Incoming options
#
readonly KUBE_SKIP_CONFIRMATIONS="${KUBE_SKIP_CONFIRMATIONS:-n}"
readonly KUBE_GCS_UPLOAD_RELEASE="${KUBE_GCS_UPLOAD_RELEASE:-n}"
readonly KUBE_GCS_NO_CACHING="${KUBE_GCS_NO_CACHING:-y}"
readonly KUBE_GCS_MAKE_PUBLIC="${KUBE_GCS_MAKE_PUBLIC:-y}"
# KUBE_GCS_RELEASE_BUCKET default: kubernetes-releases-${project_hash}
readonly KUBE_GCS_RELEASE_PREFIX=${KUBE_GCS_RELEASE_PREFIX-devel}/
readonly KUBE_GCS_DOCKER_REG_PREFIX=${KUBE_GCS_DOCKER_REG_PREFIX-docker-reg}/
readonly KUBE_GCS_PUBLISH_VERSION=${KUBE_GCS_PUBLISH_VERSION:-}
readonly KUBE_GCS_DELETE_EXISTING="${KUBE_GCS_DELETE_EXISTING:-n}"

# Constants
readonly KUBE_BUILD_IMAGE_REPO=kube-build
readonly KUBE_BUILD_IMAGE_CROSS_TAG="v1.6.2-1"
# KUBE_BUILD_DATA_CONTAINER_NAME=kube-build-data-<hash>"

# Here we map the output directories across both the local and remote _output
# directories:
#
# *_OUTPUT_ROOT    - the base of all output in that environment.
# *_OUTPUT_SUBPATH - location where golang stuff is built/cached.  Also
#                    persisted across docker runs with a volume mount.
# *_OUTPUT_BINPATH - location where final binaries are placed.  If the remote
#                    is really remote, this is the stuff that has to be copied
#                    back.
# OUT_DIR can come in from the Makefile, so honor it.
readonly LOCAL_OUTPUT_ROOT="${KUBE_ROOT}/${OUT_DIR:-_output}"
readonly LOCAL_OUTPUT_SUBPATH="${LOCAL_OUTPUT_ROOT}/dockerized"
readonly LOCAL_OUTPUT_BINPATH="${LOCAL_OUTPUT_SUBPATH}/bin"
readonly LOCAL_OUTPUT_IMAGE_STAGING="${LOCAL_OUTPUT_ROOT}/images"

readonly OUTPUT_BINPATH="${CUSTOM_OUTPUT_BINPATH:-$LOCAL_OUTPUT_BINPATH}"

readonly REMOTE_OUTPUT_ROOT="/go/src/${KUBE_GO_PACKAGE}/_output"
readonly REMOTE_OUTPUT_SUBPATH="${REMOTE_OUTPUT_ROOT}/dockerized"
readonly REMOTE_OUTPUT_BINPATH="${REMOTE_OUTPUT_SUBPATH}/bin"

readonly DOCKER_MOUNT_ARGS_BASE=(
  --volume "${OUTPUT_BINPATH}:${REMOTE_OUTPUT_BINPATH}"
  --volume /etc/localtime:/etc/localtime:ro
)

# We create a Docker data container to cache incremental build artifacts.
readonly REMOTE_OUTPUT_GOPATH="${REMOTE_OUTPUT_SUBPATH}/go"
readonly DOCKER_DATA_MOUNT_ARGS=(
  --volume "${REMOTE_OUTPUT_GOPATH}"
)

# This is where the final release artifacts are created locally
readonly RELEASE_STAGE="${LOCAL_OUTPUT_ROOT}/release-stage"
readonly RELEASE_DIR="${LOCAL_OUTPUT_ROOT}/release-tars"
readonly GCS_STAGE="${LOCAL_OUTPUT_ROOT}/gcs-stage"

# Get the set of master binaries that run in Docker (on Linux)
# Entry format is "<name-of-binary>,<base-image>".
# Binaries are placed in /usr/local/bin inside the image.
#
# $1 - server architecture
kube::build::get_docker_wrapped_binaries() {
  case $1 in
    "amd64")
        local targets=(
          kube-apiserver,busybox
          kube-controller-manager,busybox
          kube-scheduler,busybox
          kube-proxy,gcr.io/google_containers/debian-iptables-amd64:v3
        );;
    "arm")
        local targets=(
          kube-apiserver,armel/busybox
          kube-controller-manager,armel/busybox
          kube-scheduler,armel/busybox
          kube-proxy,gcr.io/google_containers/debian-iptables-arm:v3
        );;
    "arm64")
        local targets=(
          kube-apiserver,aarch64/busybox
          kube-controller-manager,aarch64/busybox
          kube-scheduler,aarch64/busybox
          kube-proxy,gcr.io/google_containers/debian-iptables-arm64:v3
        );;
    "ppc64le")
        local targets=(
          kube-apiserver,ppc64le/busybox
          kube-controller-manager,ppc64le/busybox
          kube-scheduler,ppc64le/busybox
          kube-proxy,gcr.io/google_containers/debian-iptables-ppc64le:v3
        );;
  esac

  echo "${targets[@]}"
}

# ---------------------------------------------------------------------------
# Basic setup functions

# Verify that the right utilities and such are installed for building Kube. Set
# up some dynamic constants.
#
# Vars set:
#   KUBE_ROOT_HASH
#   KUBE_BUILD_IMAGE_TAG
#   KUBE_BUILD_IMAGE
#   KUBE_BUILD_CONTAINER_NAME
#   KUBE_BUILD_DATA_CONTAINER_NAME
#   DOCKER_MOUNT_ARGS
#   LOCAL_OUTPUT_BUILD_CONTEXT
function kube::build::verify_prereqs() {
  kube::log::status "Verifying Prerequisites...."
  kube::build::ensure_tar || return 1
  kube::build::ensure_docker_in_path || return 1
  if kube::build::is_osx; then
      kube::build::docker_available_on_osx || return 1
  fi
  kube::build::ensure_docker_daemon_connectivity || return 1

  KUBE_ROOT_HASH=$(kube::build::short_hash "${HOSTNAME:-}:${KUBE_ROOT}")
  KUBE_BUILD_IMAGE_TAG="build-${KUBE_ROOT_HASH}"
  KUBE_BUILD_IMAGE="${KUBE_BUILD_IMAGE_REPO}:${KUBE_BUILD_IMAGE_TAG}"
  KUBE_BUILD_CONTAINER_NAME="kube-build-${KUBE_ROOT_HASH}"
  KUBE_BUILD_DATA_CONTAINER_NAME="kube-build-data-${KUBE_ROOT_HASH}"
  DOCKER_MOUNT_ARGS=("${DOCKER_MOUNT_ARGS_BASE[@]}" --volumes-from "${KUBE_BUILD_DATA_CONTAINER_NAME}")
  LOCAL_OUTPUT_BUILD_CONTEXT="${LOCAL_OUTPUT_IMAGE_STAGING}/${KUBE_BUILD_IMAGE}"
}

# ---------------------------------------------------------------------------
# Utility functions

function kube::build::docker_available_on_osx() {
  if [[ -z "${DOCKER_HOST}" ]]; then
    kube::log::status "No docker host is set. Checking options for setting one..."

    if [[ -z "$(which docker-machine)" && -z "$(which boot2docker)" ]]; then
      kube::log::status "It looks like you're running Mac OS X, and neither docker-machine or boot2docker are nowhere to be found."
      kube::log::status "See: https://docs.docker.com/machine/ for installation instructions."
      return 1
    elif [[ -n "$(which docker-machine)" ]]; then
      kube::build::prepare_docker_machine
    elif [[ -n "$(which boot2docker)" ]]; then
      kube::build::prepare_boot2docker
    fi
  fi
}

function kube::build::prepare_docker_machine() {
  kube::log::status "docker-machine was found."
  docker-machine inspect "${DOCKER_MACHINE_NAME}" &> /dev/null || {
    kube::log::status "Creating a machine to build Kubernetes"
    docker-machine create --driver ${DOCKER_MACHINE_DRIVER} \
      --engine-env HTTP_PROXY="${KUBERNETES_HTTP_PROXY:-}" \
      --engine-env HTTPS_PROXY="${KUBERNETES_HTTPS_PROXY:-}" \
      --engine-env NO_PROXY="${KUBERNETES_NO_PROXY:-127.0.0.1}" \
      "${DOCKER_MACHINE_NAME}" > /dev/null || {
      kube::log::error "Something went wrong creating a machine."
      kube::log::error "Try the following: "
      kube::log::error "docker-machine create -d ${DOCKER_MACHINE_DRIVER} ${DOCKER_MACHINE_NAME}"
      return 1
    }
  }
  docker-machine start "${DOCKER_MACHINE_NAME}" &> /dev/null
  # it takes `docker-machine env` a few seconds to work if the machine was just started
  while ! docker-machine env ${DOCKER_MACHINE_NAME} &> /dev/null; do
    sleep 1
  done
  eval $(docker-machine env "${DOCKER_MACHINE_NAME}")
  kube::log::status "A Docker host using docker-machine named '${DOCKER_MACHINE_NAME}' is ready to go!"
  return 0
}

function kube::build::prepare_boot2docker() {
  kube::log::status "boot2docker cli has been deprecated in favor of docker-machine."
  kube::log::status "See: https://github.com/boot2docker/boot2docker-cli for more details."
  if [[ $(boot2docker status) != "running" ]]; then
    kube::log::status "boot2docker isn't running. We'll try to start it."
    boot2docker up || {
      kube::log::error "Can't start boot2docker."
      kube::log::error "You may need to 'boot2docker init' to create your VM."
      return 1
    }
  fi

  # Reach over and set the clock. After sleep/resume the clock will skew.
  kube::log::status "Setting boot2docker clock"
  boot2docker ssh sudo date -u -D "%Y%m%d%H%M.%S" --set "$(date -u +%Y%m%d%H%M.%S)" >/dev/null

  kube::log::status "Setting boot2docker env variables"
  $(boot2docker shellinit)
  kube::log::status "boot2docker-vm has been successfully started."

  return 0
}

function kube::build::is_osx() {
  [[ "$(uname)" == "Darwin" ]]
}

function kube::build::update_dockerfile() {
  if kube::build::is_osx; then
    sed_opts=(-i '')
  else
    sed_opts=(-i)
  fi
  sed "${sed_opts[@]}" "s/KUBE_BUILD_IMAGE_CROSS_TAG/${KUBE_BUILD_IMAGE_CROSS_TAG}/" "${LOCAL_OUTPUT_BUILD_CONTEXT}/Dockerfile"
}

function kube::build::ensure_docker_in_path() {
  if [[ -z "$(which docker)" ]]; then
    kube::log::error "Can't find 'docker' in PATH, please fix and retry."
    kube::log::error "See https://docs.docker.com/installation/#installation for installation instructions."
    return 1
  fi
}

function kube::build::ensure_docker_daemon_connectivity {
  if ! "${DOCKER[@]}" info > /dev/null 2>&1 ; then
    {
      echo "Can't connect to 'docker' daemon.  please fix and retry."
      echo
      echo "Possible causes:"
      echo "  - On Mac OS X, DOCKER_HOST hasn't been set. You may need to: "
      echo "    - Create and start your VM using docker-machine or boot2docker: "
      echo "      - docker-machine create -d ${DOCKER_MACHINE_DRIVER} ${DOCKER_MACHINE_NAME}"
      echo "      - boot2docker init && boot2docker start"
      echo "    - Set your environment variables using: "
      echo "      - eval \$(docker-machine env ${DOCKER_MACHINE_NAME})"
      echo "      - \$(boot2docker shellinit)"
      echo "  - On Linux, user isn't in 'docker' group.  Add and relogin."
      echo "    - Something like 'sudo usermod -a -G docker ${USER-user}'"
      echo "    - RHEL7 bug and workaround: https://bugzilla.redhat.com/show_bug.cgi?id=1119282#c8"
      echo "  - On Linux, Docker daemon hasn't been started or has crashed."
    } >&2
    return 1
  fi
}

function kube::build::ensure_tar() {
  if [[ -n "${TAR:-}" ]]; then
    return
  fi

  # Find gnu tar if it is available, bomb out if not.
  TAR=tar
  if which gtar &>/dev/null; then
      TAR=gtar
  else
      if which gnutar &>/dev/null; then
	  TAR=gnutar
      fi
  fi
  if ! "${TAR}" --version | grep -q GNU; then
    echo "  !!! Cannot find GNU tar. Build on Linux or install GNU tar"
    echo "      on Mac OS X (brew install gnu-tar)."
    return 1
  fi
}

function kube::build::clean_output() {
  # Clean out the output directory if it exists.
  if kube::build::has_docker ; then
    if kube::build::build_image_built ; then
      kube::log::status "Cleaning out _output/dockerized/bin/ via docker build image"
      kube::build::run_build_command bash -c "rm -rf '${REMOTE_OUTPUT_BINPATH}'/*"
    else
      kube::log::error "Build image not built.  Cannot clean via docker build image."
    fi

    kube::log::status "Removing data container"
    "${DOCKER[@]}" rm -v "${KUBE_BUILD_DATA_CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi

  kube::log::status "Cleaning out local _output directory"
  rm -rf "${LOCAL_OUTPUT_ROOT}"
}

# Make sure the _output directory is created and mountable by docker
function kube::build::prepare_output() {
  mkdir -p "${LOCAL_OUTPUT_SUBPATH}"

  # On RHEL/Fedora SELinux is enabled by default and currently breaks docker
  # volume mounts.  We can work around this by explicitly adding a security
  # context to the _output directory.
  # Details: http://www.projectatomic.io/blog/2015/06/using-volumes-with-docker-can-cause-problems-with-selinux/
  if which selinuxenabled &>/dev/null && \
      selinuxenabled && \
      which chcon >/dev/null ; then
    if [[ ! $(ls -Zd "${LOCAL_OUTPUT_ROOT}") =~ svirt_sandbox_file_t ]] ; then
      kube::log::status "Applying SELinux policy to '_output' directory."
      if ! chcon -Rt svirt_sandbox_file_t "${LOCAL_OUTPUT_ROOT}"; then
        echo "    ***Failed***.  This may be because you have root owned files under _output."
        echo "    Continuing, but this build may fail later if SELinux prevents access."
      fi
    fi
    number=${#DOCKER_MOUNT_ARGS[@]}
    for (( i=0; i<number; i++ )); do
      if [[ "${DOCKER_MOUNT_ARGS[i]}" =~ "${KUBE_ROOT}" ]]; then
        DOCKER_MOUNT_ARGS[i]="${DOCKER_MOUNT_ARGS[i]}:Z"
      fi
    done
  fi

}

function kube::build::has_docker() {
  which docker &> /dev/null
}

# Detect if a specific image exists
#
# $1 - image repo name
# #2 - image tag
function kube::build::docker_image_exists() {
  [[ -n $1 && -n $2 ]] || {
    kube::log::error "Internal error. Image not specified in docker_image_exists."
    exit 2
  }

  # We cannot just specify the IMAGE here as `docker images` doesn't behave as
  # expected.  See: https://github.com/docker/docker/issues/8048
  "${DOCKER[@]}" images | grep -Eq "^(\S+/)?${1}\s+${2}\s+"
}

# Takes $1 and computes a short has for it. Useful for unique tag generation
function kube::build::short_hash() {
  [[ $# -eq 1 ]] || {
    kube::log::error "Internal error.  No data based to short_hash."
    exit 2
  }

  local short_hash
  if which md5 >/dev/null 2>&1; then
    short_hash=$(md5 -q -s "$1")
  else
    short_hash=$(echo -n "$1" | md5sum)
  fi
  echo ${short_hash:0:10}
}

# Pedantically kill, wait-on and remove a container. The -f -v options
# to rm don't actually seem to get the job done, so force kill the
# container, wait to ensure it's stopped, then try the remove. This is
# a workaround for bug https://github.com/docker/docker/issues/3968.
function kube::build::destroy_container() {
  "${DOCKER[@]}" kill "$1" >/dev/null 2>&1 || true
  "${DOCKER[@]}" wait "$1" >/dev/null 2>&1 || true
  "${DOCKER[@]}" rm -f -v "$1" >/dev/null 2>&1 || true
}

# Validate a release version
#
# Globals:
#   None
# Arguments:
#   version
# Returns:
#   If version is a valid release version
# Sets:                    (e.g. for '1.2.3-alpha.4')
#   VERSION_MAJOR          (e.g. '1')
#   VERSION_MINOR          (e.g. '2')
#   VERSION_PATCH          (e.g. '3')
#   VERSION_EXTRA          (e.g. '-alpha.4')
#   VERSION_PRERELEASE     (e.g. 'alpha')
#   VERSION_PRERELEASE_REV (e.g. '4')
function kube::release::parse_and_validate_release_version() {
  local -r version_regex="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(-(beta|alpha)\\.(0|[1-9][0-9]*))?$"
  local -r version="${1-}"
  [[ "${version}" =~ ${version_regex} ]] || {
    kube::log::error "Invalid release version: '${version}', must match regex ${version_regex}"
    return 1
  }
  VERSION_MAJOR="${BASH_REMATCH[1]}"
  VERSION_MINOR="${BASH_REMATCH[2]}"
  VERSION_PATCH="${BASH_REMATCH[3]}"
  VERSION_EXTRA="${BASH_REMATCH[4]}"
  VERSION_PRERELEASE="${BASH_REMATCH[5]}"
  VERSION_PRERELEASE_REV="${BASH_REMATCH[6]}"
}

# Validate a ci version
#
# Globals:
#   None
# Arguments:
#   version
# Returns:
#   If version is a valid ci version
# Sets:                    (e.g. for '1.2.3-alpha.4.56+abcdef12345678')
#   VERSION_MAJOR          (e.g. '1')
#   VERSION_MINOR          (e.g. '2')
#   VERSION_PATCH          (e.g. '3')
#   VERSION_PRERELEASE     (e.g. 'alpha')
#   VERSION_PRERELEASE_REV (e.g. '4')
#   VERSION_BUILD_INFO     (e.g. '.56+abcdef12345678')
#   VERSION_COMMITS        (e.g. '56')
function kube::release::parse_and_validate_ci_version() {
  # Accept things like "v1.2.3-alpha.4.56+abcdef12345678" or "v1.2.3-beta.4"
  local -r version_regex="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)-(beta|alpha)\\.(0|[1-9][0-9]*)(\\.(0|[1-9][0-9]*)\\+[0-9a-f]{7,40})?$"
  local -r version="${1-}"
  [[ "${version}" =~ ${version_regex} ]] || {
    kube::log::error "Invalid ci version: '${version}', must match regex ${version_regex}"
    return 1
  }
  VERSION_MAJOR="${BASH_REMATCH[1]}"
  VERSION_MINOR="${BASH_REMATCH[2]}"
  VERSION_PATCH="${BASH_REMATCH[3]}"
  VERSION_PRERELEASE="${BASH_REMATCH[4]}"
  VERSION_PRERELEASE_REV="${BASH_REMATCH[5]}"
  VERSION_BUILD_INFO="${BASH_REMATCH[6]}"
  VERSION_COMMITS="${BASH_REMATCH[7]}"
}

# ---------------------------------------------------------------------------
# Building

function kube::build::build_image_built() {
  kube::build::docker_image_exists "${KUBE_BUILD_IMAGE_REPO}" "${KUBE_BUILD_IMAGE_TAG}"
}

# The set of source targets to include in the kube-build image
function kube::build::source_targets() {
  local targets=(
    api
    build
    cluster
    cmd
    docs
    examples
    federation
    Godeps/Godeps.json
    hack
    LICENSE
    pkg
    plugin
    DESIGN.md
    README.md
    test
    third_party
    vendor
    contrib/completions/bash/kubectl
    contrib/mesos
    .generated_docs
  )
  if [ -n "${KUBERNETES_CONTRIB:-}" ]; then
    for contrib in "${KUBERNETES_CONTRIB}"; do
      targets+=($(eval "kube::contrib::${contrib}::source_targets"))
    done
  fi
  echo "${targets[@]}"
}

# Set up the context directory for the kube-build image and build it.
function kube::build::build_image() {
  kube::build::ensure_tar

  mkdir -p "${LOCAL_OUTPUT_BUILD_CONTEXT}"
  "${TAR}" czf "${LOCAL_OUTPUT_BUILD_CONTEXT}/kube-source.tar.gz" $(kube::build::source_targets)

  kube::version::get_version_vars
  kube::version::save_version_vars "${LOCAL_OUTPUT_BUILD_CONTEXT}/kube-version-defs"

  cp build/build-image/Dockerfile "${LOCAL_OUTPUT_BUILD_CONTEXT}/Dockerfile"
  kube::build::update_dockerfile

  kube::build::docker_build "${KUBE_BUILD_IMAGE}" "${LOCAL_OUTPUT_BUILD_CONTEXT}" 'false'
}

# Build a docker image from a Dockerfile.
# $1 is the name of the image to build
# $2 is the location of the "context" directory, with the Dockerfile at the root.
# $3 is the value to set the --pull flag for docker build; true by default
function kube::build::docker_build() {
  local -r image=$1
  local -r context_dir=$2
  local -r pull="${3:-true}"
  local -ra build_cmd=("${DOCKER[@]}" build -t "${image}" "--pull=${pull}" "${context_dir}")

  kube::log::status "Building Docker image ${image}."
  local docker_output
  docker_output=$("${build_cmd[@]}" 2>&1) || {
    cat <<EOF >&2
+++ Docker build command failed for ${image}

${docker_output}

To retry manually, run:

${build_cmd[*]}

EOF
    return 1
  }
}

function kube::build::clean_image() {
  local -r image=$1

  kube::log::status "Deleting docker image ${image}"
  "${DOCKER[@]}" rmi ${image} 2> /dev/null || true
}

function kube::build::clean_images() {
  kube::build::has_docker || return 0

  kube::build::clean_image "${KUBE_BUILD_IMAGE}"

  kube::log::status "Cleaning all other untagged docker images"
  "${DOCKER[@]}" rmi $("${DOCKER[@]}" images -q --filter 'dangling=true') 2> /dev/null || true
}

function kube::build::ensure_data_container() {
  # This is temporary, while last remnants of _workspace are obliterated.  If
  # the data container exists it might be from before the change from
  # Godeps/_workspace/ to vendor/, and thereby still have a Godeps/_workspace/
  # directory in it, which trips up godep (yay godep!).  Once we are confident
  # that this has run ~everywhere we care about, we can remove this.
  # TODO(thockin): remove this after v1.3 is cut.
  if "${DOCKER[@]}" inspect "${KUBE_BUILD_DATA_CONTAINER_NAME}" \
      | grep -q "Godeps/_workspace/pkg"; then
    docker rm -f "${KUBE_BUILD_DATA_CONTAINER_NAME}"
  fi
  if ! "${DOCKER[@]}" inspect "${KUBE_BUILD_DATA_CONTAINER_NAME}" >/dev/null 2>&1; then
    kube::log::status "Creating data container"
    local -ra docker_cmd=(
      "${DOCKER[@]}" run
      "${DOCKER_DATA_MOUNT_ARGS[@]}"
      --name "${KUBE_BUILD_DATA_CONTAINER_NAME}"
      "${KUBE_BUILD_IMAGE}"
      true
    )
    "${docker_cmd[@]}"
  fi
}

# Run a command in the kube-build image.  This assumes that the image has
# already been built.  This will sync out all output data from the build.
function kube::build::run_build_command() {
  kube::log::status "Running build command...."
  [[ $# != 0 ]] || { echo "Invalid input - please specify a command to run." >&2; return 4; }

  kube::build::ensure_data_container
  kube::build::prepare_output

  local -a docker_run_opts=(
    "--name=${KUBE_BUILD_CONTAINER_NAME}"
    "${DOCKER_MOUNT_ARGS[@]}"
  )

  if [ -n "${KUBERNETES_CONTRIB:-}" ]; then
    docker_run_opts+=(-e "KUBERNETES_CONTRIB=${KUBERNETES_CONTRIB}")
  fi

  docker_run_opts+=(
    --env "KUBE_FASTBUILD=${KUBE_FASTBUILD:-false}"
    --env "KUBE_BUILDER_OS=${OSTYPE:-notdetected}"
  )

  # If we have stdin we can run interactive.  This allows things like 'shell.sh'
  # to work.  However, if we run this way and don't have stdin, then it ends up
  # running in a daemon-ish mode.  So if we don't have a stdin, we explicitly
  # attach stderr/stdout but don't bother asking for a tty.
  if [[ -t 0 ]]; then
    docker_run_opts+=(--interactive --tty)
  else
    docker_run_opts+=(--attach=stdout --attach=stderr)
  fi

  local -ra docker_cmd=(
    "${DOCKER[@]}" run "${docker_run_opts[@]}" "${KUBE_BUILD_IMAGE}")

  # Clean up container from any previous run
  kube::build::destroy_container "${KUBE_BUILD_CONTAINER_NAME}"
  "${docker_cmd[@]}" "$@"
  kube::build::destroy_container "${KUBE_BUILD_CONTAINER_NAME}"
}

# Test if the output directory is remote (and can only be accessed through
# docker) or if it is "local" and we can access the output without going through
# docker.
function kube::build::is_output_remote() {
  rm -f "${LOCAL_OUTPUT_SUBPATH}/test_for_remote"
  kube::build::run_build_command touch "${REMOTE_OUTPUT_BINPATH}/test_for_remote"

  [[ ! -e "${LOCAL_OUTPUT_BINPATH}/test_for_remote" ]]
}

# If the Docker server is remote, copy the results back out.
function kube::build::copy_output() {
  if kube::build::is_output_remote; then
    # At time of this code, docker cp does not work when copying from a volume.
    # As a workaround, the binaries are first copied to a local filesystem,
    # /tmp, then docker cp'd to the local binaries output directory.
    # The fix for the volume bug has been accepted and once it's widely
    # deployed the code below should be simplified to a simple docker cp
    # Bug: https://github.com/docker/docker/pull/8509
    local -a docker_run_opts=(
      "--name=${KUBE_BUILD_CONTAINER_NAME}"
       "${DOCKER_MOUNT_ARGS[@]}"
       -d
      )

    local -ra docker_cmd=(
      "${DOCKER[@]}" run "${docker_run_opts[@]}" "${KUBE_BUILD_IMAGE}"
    )

    kube::log::status "Syncing back _output/dockerized/bin directory from remote Docker"
    rm -rf "${LOCAL_OUTPUT_BINPATH}"
    mkdir -p "${LOCAL_OUTPUT_BINPATH}"

    kube::build::destroy_container "${KUBE_BUILD_CONTAINER_NAME}"
    "${docker_cmd[@]}" bash -c "cp -r ${REMOTE_OUTPUT_BINPATH} /tmp/bin;touch /tmp/finished;rm /tmp/bin/test_for_remote;/bin/sleep 600" > /dev/null 2>&1

    # Wait until binaries have finished coppying
    count=0
    while true;do
      if "${DOCKER[@]}" cp "${KUBE_BUILD_CONTAINER_NAME}:/tmp/finished" "${LOCAL_OUTPUT_BINPATH}" > /dev/null 2>&1;then
        "${DOCKER[@]}" cp "${KUBE_BUILD_CONTAINER_NAME}:/tmp/bin" "${LOCAL_OUTPUT_SUBPATH}"
        break;
      fi

      let count=count+1
      if [[ $count -eq 60 ]]; then
        # break after 5m
        kube::log::error "Timed out waiting for binaries..."
        break
      fi
      sleep 5
    done

    "${DOCKER[@]}" rm -f -v "${KUBE_BUILD_CONTAINER_NAME}" >/dev/null 2>&1 || true
  else
    kube::log::status "Output directory is local.  No need to copy results out."
  fi
}

# ---------------------------------------------------------------------------
# Build final release artifacts
function kube::release::clean_cruft() {
  # Clean out cruft
  find ${RELEASE_STAGE} -name '*~' -exec rm {} \;
  find ${RELEASE_STAGE} -name '#*#' -exec rm {} \;
  find ${RELEASE_STAGE} -name '.DS*' -exec rm {} \;
}

function kube::release::package_hyperkube() {
  # If we have these variables set then we want to build all docker images.
  if [[ -n "${KUBE_DOCKER_IMAGE_TAG-}" && -n "${KUBE_DOCKER_REGISTRY-}" ]]; then
    for arch in "${KUBE_SERVER_PLATFORMS[@]##*/}"; do

      kube::log::status "Building hyperkube image for arch: ${arch}"
      REGISTRY="${KUBE_DOCKER_REGISTRY}" VERSION="${KUBE_DOCKER_IMAGE_TAG}" ARCH="${arch}" make -C cluster/images/hyperkube/ build
    done
  fi
}

function kube::release::package_tarballs() {
  # Clean out any old releases
  rm -rf "${RELEASE_DIR}"
  mkdir -p "${RELEASE_DIR}"
  kube::release::package_build_image_tarball &
  kube::release::package_client_tarballs &
  kube::release::package_server_tarballs &
  kube::release::package_salt_tarball &
  kube::release::package_kube_manifests_tarball &
  kube::util::wait-for-jobs || { kube::log::error "previous tarball phase failed"; return 1; }

  kube::release::package_full_tarball & # _full depends on all the previous phases
  kube::release::package_test_tarball & # _test doesn't depend on anything
  kube::util::wait-for-jobs || { kube::log::error "previous tarball phase failed"; return 1; }
}

# Package the build image we used from the previous stage, for compliance/licensing/audit/yadda.
function kube::release::package_build_image_tarball() {
  kube::log::status "Building tarball: src"
  "${TAR}" czf "${RELEASE_DIR}/kubernetes-src.tar.gz" -C "${LOCAL_OUTPUT_BUILD_CONTEXT}" .
}

# Package up all of the cross compiled clients. Over time this should grow into
# a full SDK
function kube::release::package_client_tarballs() {
   # Find all of the built client binaries
  local platform platforms
  platforms=($(cd "${LOCAL_OUTPUT_BINPATH}" ; echo */*))
  for platform in "${platforms[@]}"; do
    local platform_tag=${platform/\//-} # Replace a "/" for a "-"
    kube::log::status "Starting tarball: client $platform_tag"

    (
      local release_stage="${RELEASE_STAGE}/client/${platform_tag}/kubernetes"
      rm -rf "${release_stage}"
      mkdir -p "${release_stage}/client/bin"

      local client_bins=("${KUBE_CLIENT_BINARIES[@]}")
      if [[ "${platform%/*}" == "windows" ]]; then
        client_bins=("${KUBE_CLIENT_BINARIES_WIN[@]}")
      fi

      # This fancy expression will expand to prepend a path
      # (${LOCAL_OUTPUT_BINPATH}/${platform}/) to every item in the
      # KUBE_CLIENT_BINARIES array.
      cp "${client_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
        "${release_stage}/client/bin/"

      kube::release::clean_cruft

      local package_name="${RELEASE_DIR}/kubernetes-client-${platform_tag}.tar.gz"
      kube::release::create_tarball "${package_name}" "${release_stage}/.."
    ) &
  done

  kube::log::status "Waiting on tarballs"
  kube::util::wait-for-jobs || { kube::log::error "client tarball creation failed"; exit 1; }
}

# Package up all of the server binaries
function kube::release::package_server_tarballs() {
  local platform
  for platform in "${KUBE_SERVER_PLATFORMS[@]}"; do
    local platform_tag=${platform/\//-} # Replace a "/" for a "-"
    local arch=$(basename ${platform})
    kube::log::status "Building tarball: server $platform_tag"

    local release_stage="${RELEASE_STAGE}/server/${platform_tag}/kubernetes"
    rm -rf "${release_stage}"
    mkdir -p "${release_stage}/server/bin"
    mkdir -p "${release_stage}/addons"

    # This fancy expression will expand to prepend a path
    # (${LOCAL_OUTPUT_BINPATH}/${platform}/) to every item in the
    # KUBE_SERVER_BINARIES array.
    cp "${KUBE_SERVER_BINARIES[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/server/bin/"

    kube::release::create_docker_images_for_server "${release_stage}/server/bin" "${arch}"

    # Include the client binaries here too as they are useful debugging tools.
    local client_bins=("${KUBE_CLIENT_BINARIES[@]}")
    if [[ "${platform%/*}" == "windows" ]]; then
      client_bins=("${KUBE_CLIENT_BINARIES_WIN[@]}")
    fi
    cp "${client_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/server/bin/"

    cp "${KUBE_ROOT}/Godeps/LICENSES" "${release_stage}/"

    kube::release::clean_cruft

    local package_name="${RELEASE_DIR}/kubernetes-server-${platform_tag}.tar.gz"
    kube::release::create_tarball "${package_name}" "${release_stage}/.."
  done
}

function kube::release::md5() {
  if which md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    md5sum "$1" | awk '{ print $1 }'
  fi
}

function kube::release::sha1() {
  if which shasum >/dev/null 2>&1; then
    shasum -a1 "$1" | awk '{ print $1 }'
  else
    sha1sum "$1" | awk '{ print $1 }'
  fi
}

# This will take binaries that run on master and creates Docker images
# that wrap the binary in them. (One docker image per binary)
# Args:
#  $1 - binary_dir, the directory to save the tared images to.
#  $2 - arch, architecture for which we are building docker images.
function kube::release::create_docker_images_for_server() {
  # Create a sub-shell so that we don't pollute the outer environment
  (
    local binary_dir="$1"
    local arch="$2"
    local binary_name
    local binaries=($(kube::build::get_docker_wrapped_binaries ${arch}))

    for wrappable in "${binaries[@]}"; do

      local oldifs=$IFS
      IFS=","
      set $wrappable
      IFS=$oldifs

      local binary_name="$1"
      local base_image="$2"

      kube::log::status "Starting Docker build for image: ${binary_name}"

      (
        local md5_sum
        md5_sum=$(kube::release::md5 "${binary_dir}/${binary_name}")

        local docker_build_path="${binary_dir}/${binary_name}.dockerbuild"
        local docker_file_path="${docker_build_path}/Dockerfile"
        local binary_file_path="${binary_dir}/${binary_name}"

        rm -rf ${docker_build_path}
        mkdir -p ${docker_build_path}
        ln ${binary_dir}/${binary_name} ${docker_build_path}/${binary_name}
        printf " FROM ${base_image} \n ADD ${binary_name} /usr/local/bin/${binary_name}\n" > ${docker_file_path}

        if [[ ${arch} == "amd64" ]]; then
          # If we are building a amd64 docker image, preserve the original image name
          local docker_image_tag=gcr.io/google_containers/${binary_name}:${md5_sum}
        else
          # If we are building a docker image for another architecture, append the arch in the image tag
          local docker_image_tag=gcr.io/google_containers/${binary_name}-${arch}:${md5_sum}
        fi

        "${DOCKER[@]}" build -q -t "${docker_image_tag}" ${docker_build_path} >/dev/null
        "${DOCKER[@]}" save ${docker_image_tag} > ${binary_dir}/${binary_name}.tar
        echo $md5_sum > ${binary_dir}/${binary_name}.docker_tag

        rm -rf ${docker_build_path}

        # If we are building an official/alpha/beta release we want to keep docker images
        # and tag them appropriately.
        if [[ -n "${KUBE_DOCKER_IMAGE_TAG-}" && -n "${KUBE_DOCKER_REGISTRY-}" ]]; then
          local release_docker_image_tag="${KUBE_DOCKER_REGISTRY}/${binary_name}-${arch}:${KUBE_DOCKER_IMAGE_TAG}"
          kube::log::status "Tagging docker image ${docker_image_tag} as ${release_docker_image_tag}"
          "${DOCKER[@]}" tag -f "${docker_image_tag}" "${release_docker_image_tag}" 2>/dev/null
        fi

        kube::log::status "Deleting docker image ${docker_image_tag}"
        "${DOCKER[@]}" rmi ${docker_image_tag} 2>/dev/null || true
      ) &
    done

    kube::util::wait-for-jobs || { kube::log::error "previous Docker build failed"; return 1; }
    kube::log::status "Docker builds done"
  )

}

# Package up the salt configuration tree.  This is an optional helper to getting
# a cluster up and running.
function kube::release::package_salt_tarball() {
  kube::log::status "Building tarball: salt"

  local release_stage="${RELEASE_STAGE}/salt/kubernetes"
  rm -rf "${release_stage}"
  mkdir -p "${release_stage}"

  cp -R "${KUBE_ROOT}/cluster/saltbase" "${release_stage}/"

  # TODO(#3579): This is a temporary hack. It gathers up the yaml,
  # yaml.in, json files in cluster/addons (minus any demos) and overlays
  # them into kube-addons, where we expect them. (This pipeline is a
  # fancy copy, stripping anything but the files we don't want.)
  local objects
  objects=$(cd "${KUBE_ROOT}/cluster/addons" && find . \( -name \*.yaml -or -name \*.yaml.in -or -name \*.json \) | grep -v demo)
  tar c -C "${KUBE_ROOT}/cluster/addons" ${objects} | tar x -C "${release_stage}/saltbase/salt/kube-addons"

  kube::release::clean_cruft

  local package_name="${RELEASE_DIR}/kubernetes-salt.tar.gz"
  kube::release::create_tarball "${package_name}" "${release_stage}/.."
}

# This will pack kube-system manifests files for distros without using salt
# such as GCI and Ubuntu Trusty. We directly copy manifests from
# cluster/addons and cluster/saltbase/salt. The script of cluster initialization
# will remove the salt configuration and evaluate the variables in the manifests.
function kube::release::package_kube_manifests_tarball() {
  kube::log::status "Building tarball: manifests"

  local release_stage="${RELEASE_STAGE}/manifests/kubernetes"
  rm -rf "${release_stage}"
  local dst_dir="${release_stage}/gci-trusty"
  mkdir -p "${dst_dir}"

  local salt_dir="${KUBE_ROOT}/cluster/saltbase/salt"
  cp "${salt_dir}/fluentd-es/fluentd-es.yaml" "${release_stage}/"
  cp "${salt_dir}/fluentd-gcp/fluentd-gcp.yaml" "${release_stage}/"
  cp "${salt_dir}/kube-registry-proxy/kube-registry-proxy.yaml" "${release_stage}/"
  cp "${salt_dir}/kube-proxy/kube-proxy.manifest" "${release_stage}/"
  cp "${salt_dir}/etcd/etcd.manifest" "${dst_dir}"
  cp "${salt_dir}/kube-scheduler/kube-scheduler.manifest" "${dst_dir}"
  cp "${salt_dir}/kube-apiserver/kube-apiserver.manifest" "${dst_dir}"
  cp "${salt_dir}/kube-apiserver/abac-authz-policy.jsonl" "${dst_dir}"
  cp "${salt_dir}/kube-controller-manager/kube-controller-manager.manifest" "${dst_dir}"
  cp "${salt_dir}/kube-addons/kube-addon-manager.yaml" "${dst_dir}"
  cp "${KUBE_ROOT}/cluster/gce/trusty/configure-helper.sh" "${dst_dir}"
  cp -r "${salt_dir}/kube-admission-controls/limit-range" "${dst_dir}"
  local objects
  objects=$(cd "${KUBE_ROOT}/cluster/addons" && find . \( -name \*.yaml -or -name \*.yaml.in -or -name \*.json \) | grep -v demo)
  tar c -C "${KUBE_ROOT}/cluster/addons" ${objects} | tar x -C "${dst_dir}"

  # This is for coreos only. ContainerVM, GCI, or Trusty does not use it.
  cp -r "${KUBE_ROOT}/cluster/gce/coreos/kube-manifests"/* "${release_stage}/"

  kube::release::clean_cruft

  local package_name="${RELEASE_DIR}/kubernetes-manifests.tar.gz"
  kube::release::create_tarball "${package_name}" "${release_stage}/.."
}

# This is the stuff you need to run tests from the binary distribution.
function kube::release::package_test_tarball() {
  kube::log::status "Building tarball: test"

  local release_stage="${RELEASE_STAGE}/test/kubernetes"
  rm -rf "${release_stage}"
  mkdir -p "${release_stage}"

  local platform
  for platform in "${KUBE_TEST_PLATFORMS[@]}"; do
    local test_bins=("${KUBE_TEST_BINARIES[@]}")
    if [[ "${platform%/*}" == "windows" ]]; then
      test_bins=("${KUBE_TEST_BINARIES_WIN[@]}")
    fi
    mkdir -p "${release_stage}/platforms/${platform}"
    cp "${test_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/platforms/${platform}"
  done

  # Add the test image files
  mkdir -p "${release_stage}/test/images"
  cp -fR "${KUBE_ROOT}/test/images" "${release_stage}/test/"
  tar c ${KUBE_TEST_PORTABLE[@]} | tar x -C ${release_stage}

  kube::release::clean_cruft

  local package_name="${RELEASE_DIR}/kubernetes-test.tar.gz"
  kube::release::create_tarball "${package_name}" "${release_stage}/.."
}

# This is all the stuff you need to run/install kubernetes.  This includes:
#   - precompiled binaries for client
#   - Cluster spin up/down scripts and configs for various cloud providers
#   - tarballs for server binary and salt configs that are ready to be uploaded
#     to master by whatever means appropriate.
function kube::release::package_full_tarball() {
  kube::log::status "Building tarball: full"

  local release_stage="${RELEASE_STAGE}/full/kubernetes"
  rm -rf "${release_stage}"
  mkdir -p "${release_stage}"

  # Copy all of the client binaries in here, but not test or server binaries.
  # The server binaries are included with the server binary tarball.
  local platform
  for platform in "${KUBE_CLIENT_PLATFORMS[@]}"; do
    local client_bins=("${KUBE_CLIENT_BINARIES[@]}")
    if [[ "${platform%/*}" == "windows" ]]; then
      client_bins=("${KUBE_CLIENT_BINARIES_WIN[@]}")
    fi
    mkdir -p "${release_stage}/platforms/${platform}"
    cp "${client_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/platforms/${platform}"
  done

  # We want everything in /cluster except saltbase.  That is only needed on the
  # server.
  cp -R "${KUBE_ROOT}/cluster" "${release_stage}/"
  rm -rf "${release_stage}/cluster/saltbase"

  mkdir -p "${release_stage}/server"
  cp "${RELEASE_DIR}/kubernetes-salt.tar.gz" "${release_stage}/server/"
  cp "${RELEASE_DIR}"/kubernetes-server-*.tar.gz "${release_stage}/server/"
  cp "${RELEASE_DIR}/kubernetes-manifests.tar.gz" "${release_stage}/server/"

  mkdir -p "${release_stage}/third_party"
  cp -R "${KUBE_ROOT}/third_party/htpasswd" "${release_stage}/third_party/htpasswd"

  cp -R "${KUBE_ROOT}/examples" "${release_stage}/"
  cp -R "${KUBE_ROOT}/docs" "${release_stage}/"
  cp "${KUBE_ROOT}/README.md" "${release_stage}/"
  cp "${KUBE_ROOT}/Godeps/LICENSES" "${release_stage}/"
  cp "${KUBE_ROOT}/Vagrantfile" "${release_stage}/"
  mkdir -p "${release_stage}/contrib/completions/bash"
  cp "${KUBE_ROOT}/contrib/completions/bash/kubectl" "${release_stage}/contrib/completions/bash"

  echo "${KUBE_GIT_VERSION}" > "${release_stage}/version"

  kube::release::clean_cruft

  local package_name="${RELEASE_DIR}/kubernetes.tar.gz"
  kube::release::create_tarball "${package_name}" "${release_stage}/.."
}

# Build a release tarball.  $1 is the output tar name.  $2 is the base directory
# of the files to be packaged.  This assumes that ${2}/kubernetes is what is
# being packaged.
function kube::release::create_tarball() {
  kube::build::ensure_tar

  local tarfile=$1
  local stagingdir=$2

  "${TAR}" czf "${tarfile}" -C "${stagingdir}" kubernetes --owner=0 --group=0
}

# ---------------------------------------------------------------------------
# GCS Release

function kube::release::gcs::release() {
  [[ ${KUBE_GCS_UPLOAD_RELEASE} =~ ^[yY]$ ]] || return 0

  kube::release::gcs::verify_prereqs || return 1
  kube::release::gcs::ensure_release_bucket || return 1
  kube::release::gcs::copy_release_artifacts || return 1
}

# Verify things are set up for uploading to GCS
function kube::release::gcs::verify_prereqs() {
  if [[ -z "$(which gsutil)" || -z "$(which gcloud)" ]]; then
    echo "Releasing Kubernetes requires gsutil and gcloud.  Please download,"
    echo "install and authorize through the Google Cloud SDK: "
    echo
    echo "  https://developers.google.com/cloud/sdk/"
    return 1
  fi

  if [[ -z "${GCLOUD_ACCOUNT-}" ]]; then
    GCLOUD_ACCOUNT=$(gcloud auth list 2>/dev/null | awk '/(active)/ { print $2 }')
  fi
  if [[ -z "${GCLOUD_ACCOUNT-}" ]]; then
    echo "No account authorized through gcloud.  Please fix with:"
    echo
    echo "  gcloud auth login"
    return 1
  fi

  if [[ -z "${GCLOUD_PROJECT-}" ]]; then
    GCLOUD_PROJECT=$(gcloud config list project | awk '{project = $3} END {print project}')
  fi
  if [[ -z "${GCLOUD_PROJECT-}" ]]; then
    echo "No account authorized through gcloud.  Please fix with:"
    echo
    echo "  gcloud config set project <project id>"
    return 1
  fi
}

# Create a unique bucket name for releasing Kube and make sure it exists.
function kube::release::gcs::ensure_release_bucket() {
  local project_hash
  project_hash=$(kube::build::short_hash "$GCLOUD_PROJECT")
  KUBE_GCS_RELEASE_BUCKET=${KUBE_GCS_RELEASE_BUCKET-kubernetes-releases-${project_hash}}

  if ! gsutil ls "gs://${KUBE_GCS_RELEASE_BUCKET}" >/dev/null 2>&1 ; then
    echo "Creating Google Cloud Storage bucket: $KUBE_GCS_RELEASE_BUCKET"
    gsutil mb -p "${GCLOUD_PROJECT}" "gs://${KUBE_GCS_RELEASE_BUCKET}" || return 1
  fi
}

function kube::release::gcs::stage_and_hash() {
  kube::build::ensure_tar || return 1

  # Split the args into srcs... and dst
  local -r args=( "$@" )
  local -r split=$((${#args[@]}-1)) # Split point for src/dst args
  local -r srcs=( "${args[@]::${split}}" )
  local -r dst="${args[${split}]}"

  for src in ${srcs[@]}; do
    srcdir=$(dirname ${src})
    srcthing=$(basename ${src})
    mkdir -p ${GCS_STAGE}/${dst} || return 1
    "${TAR}" c -C ${srcdir} ${srcthing} | "${TAR}" x -C ${GCS_STAGE}/${dst} || return 1
  done
}

function kube::release::gcs::copy_release_artifacts() {
  # TODO: This isn't atomic.  There will be points in time where there will be
  # no active release.  Also, if something fails, the release could be half-
  # copied.  The real way to do this would perhaps to have some sort of release
  # version so that we are never overwriting a destination.
  local -r gcs_destination="gs://${KUBE_GCS_RELEASE_BUCKET}/${KUBE_GCS_RELEASE_PREFIX}"

  kube::log::status "Staging release artifacts to ${GCS_STAGE}"

  rm -rf ${GCS_STAGE} || return 1
  mkdir -p ${GCS_STAGE} || return 1

  # Stage everything in release directory
  kube::release::gcs::stage_and_hash "${RELEASE_DIR}"/* . || return 1

  # Having the configure-vm.sh script and trusty code from the GCE cluster
  # deploy hosted with the release is useful for GKE.
  # TODO(andyzheng0831): Replace the trusty path with GCI after finshing the GCI code.
  kube::release::gcs::stage_and_hash "${RELEASE_STAGE}/full/kubernetes/cluster/gce/configure-vm.sh" extra/gce || return 1
  kube::release::gcs::stage_and_hash "${RELEASE_STAGE}/full/kubernetes/cluster/gce/trusty/node.yaml" extra/gce || return 1
  kube::release::gcs::stage_and_hash "${RELEASE_STAGE}/full/kubernetes/cluster/gce/trusty/master.yaml" extra/gce || return 1
  kube::release::gcs::stage_and_hash "${RELEASE_STAGE}/full/kubernetes/cluster/gce/trusty/configure.sh" extra/gce || return 1

  # Upload the "naked" binaries to GCS.  This is useful for install scripts that
  # download the binaries directly and don't need tars.
  local platform platforms
  platforms=($(cd "${RELEASE_STAGE}/client" ; echo *))
  for platform in "${platforms[@]}"; do
    local src="${RELEASE_STAGE}/client/${platform}/kubernetes/client/bin/*"
    local dst="bin/${platform/-//}/"
    # We assume here the "server package" is a superset of the "client package"
    if [[ -d "${RELEASE_STAGE}/server/${platform}" ]]; then
      src="${RELEASE_STAGE}/server/${platform}/kubernetes/server/bin/*"
    fi
    kube::release::gcs::stage_and_hash "$src" "$dst" || return 1
  done

  kube::log::status "Hashing files in ${GCS_STAGE}"
  find ${GCS_STAGE} -type f | while read path; do
    kube::release::md5 ${path} > "${path}.md5" || return 1
    kube::release::sha1 ${path} > "${path}.sha1" || return 1
  done

  kube::log::status "Copying release artifacts to ${gcs_destination}"

  # First delete all objects at the destination
  if gsutil ls "${gcs_destination}" >/dev/null 2>&1; then
    kube::log::error "${gcs_destination} not empty."
    [[ ${KUBE_GCS_DELETE_EXISTING} =~ ^[yY]$ ]] || {
      read -p "Delete everything under ${gcs_destination}? [y/n] " -r || {
        kube::log::status "EOF on prompt.  Skipping upload"
        return
      }
      [[ $REPLY =~ ^[yY]$ ]] || {
        kube::log::status "Skipping upload"
        return
      }
    }
    kube::log::status "Deleting everything under ${gcs_destination}"
    gsutil -q -m rm -f -R "${gcs_destination}" || return 1
  fi

  local gcs_options=()
  if [[ ${KUBE_GCS_NO_CACHING} =~ ^[yY]$ ]]; then
    gcs_options=("-h" "Cache-Control:private, max-age=0")
  fi

  gsutil -q -m "${gcs_options[@]+${gcs_options[@]}}" cp -r "${GCS_STAGE}"/* ${gcs_destination} || return 1

  # TODO(jbeda): Generate an HTML page with links for this release so it is easy
  # to see it.  For extra credit, generate a dynamic page that builds up the
  # release list using the GCS JSON API.  Use Angular and Bootstrap for extra
  # extra credit.

  if [[ ${KUBE_GCS_MAKE_PUBLIC} =~ ^[yY]$ ]]; then
    kube::log::status "Marking all uploaded objects public"
    gsutil -q -m acl ch -R -g all:R "${gcs_destination}" >/dev/null 2>&1 || return 1
  fi

  gsutil ls -lhr "${gcs_destination}" || return 1
}

# Publish a new ci version, (latest,) but only if the release files actually
# exist on GCS.
#
# Globals:
#   See callees
# Arguments:
#   None
# Returns:
#   Success
function kube::release::gcs::publish_ci() {
  kube::release::gcs::verify_release_files || return 1

  kube::release::parse_and_validate_ci_version "${KUBE_GCS_PUBLISH_VERSION}" || return 1
  local -r version_major="${VERSION_MAJOR}"
  local -r version_minor="${VERSION_MINOR}"

  local -r publish_files=(ci/latest.txt ci/latest-${version_major}.txt ci/latest-${version_major}.${version_minor}.txt)

  for publish_file in ${publish_files[*]}; do
    # If there's a version that's above the one we're trying to release, don't
    # do anything, and just try the next one.
    kube::release::gcs::verify_ci_ge "${publish_file}" || continue
    kube::release::gcs::publish "${publish_file}" || return 1
  done
}

# Publish a new official version, (latest or stable,) but only if the release
# files actually exist on GCS and the release we're dealing with is newer than
# the contents in GCS.
#
# Globals:
#   KUBE_GCS_PUBLISH_VERSION
#   See callees
# Arguments:
#   release_kind: either 'latest' or 'stable'
# Returns:
#   Success
function kube::release::gcs::publish_official() {
  local -r release_kind="${1-}"

  kube::release::gcs::verify_release_files || return 1

  kube::release::parse_and_validate_release_version "${KUBE_GCS_PUBLISH_VERSION}" || return 1
  local -r version_major="${VERSION_MAJOR}"
  local -r version_minor="${VERSION_MINOR}"

  local publish_files
  if [[ "${release_kind}" == 'latest' ]]; then
    publish_files=(release/latest.txt release/latest-${version_major}.txt release/latest-${version_major}.${version_minor}.txt)
  elif [[ "${release_kind}" == 'stable' ]]; then
    publish_files=(release/stable.txt release/stable-${version_major}.txt release/stable-${version_major}.${version_minor}.txt)
  else
    kube::log::error "Wrong release_kind: must be 'latest' or 'stable'."
    return 1
  fi

  for publish_file in ${publish_files[*]}; do
    # If there's a version that's above the one we're trying to release, don't
    # do anything, and just try the next one.
    kube::release::gcs::verify_release_gt "${publish_file}" || continue
    kube::release::gcs::publish "${publish_file}" || return 1
  done
}

# Verify that the release files we expect actually exist.
#
# Globals:
#   KUBE_GCS_RELEASE_BUCKET
#   KUBE_GCS_RELEASE_PREFIX
# Arguments:
#   None
# Returns:
#   If release files exist
function kube::release::gcs::verify_release_files() {
  local -r release_dir="gs://${KUBE_GCS_RELEASE_BUCKET}/${KUBE_GCS_RELEASE_PREFIX}"
  if ! gsutil ls "${release_dir}" >/dev/null 2>&1 ; then
    kube::log::error "Release files don't exist at '${release_dir}'"
    return 1
  fi
}

# Check if the new version is greater than the version currently published on
# GCS.
#
# Globals:
#   KUBE_GCS_PUBLISH_VERSION
#   KUBE_GCS_RELEASE_BUCKET
# Arguments:
#   publish_file: the GCS location to look in
# Returns:
#   If new version is greater than the GCS version
#
# TODO(16529): This should all be outside of build an in release, and should be
# refactored to reduce code duplication.  Also consider using strictly nested
# if and explicit handling of equals case.
function kube::release::gcs::verify_release_gt() {
  local -r publish_file="${1-}"
  local -r new_version=${KUBE_GCS_PUBLISH_VERSION}
  local -r publish_file_dst="gs://${KUBE_GCS_RELEASE_BUCKET}/${publish_file}"

  kube::release::parse_and_validate_release_version "${new_version}" || return 1

  local -r version_major="${VERSION_MAJOR}"
  local -r version_minor="${VERSION_MINOR}"
  local -r version_patch="${VERSION_PATCH}"
  local -r version_prerelease="${VERSION_PRERELEASE}"
  local -r version_prerelease_rev="${VERSION_PRERELEASE_REV}"

  local gcs_version
  if gcs_version="$(gsutil cat "${publish_file_dst}")"; then
    kube::release::parse_and_validate_release_version "${gcs_version}" || {
      kube::log::error "${publish_file_dst} contains invalid release version, can't compare: '${gcs_version}'"
      return 1
    }

    local -r gcs_version_major="${VERSION_MAJOR}"
    local -r gcs_version_minor="${VERSION_MINOR}"
    local -r gcs_version_patch="${VERSION_PATCH}"
    local -r gcs_version_prerelease="${VERSION_PRERELEASE}"
    local -r gcs_version_prerelease_rev="${VERSION_PRERELEASE_REV}"

    local greater=true
    if [[ "${version_major}" -lt "${gcs_version_major}" ]]; then
      greater=false
    elif [[ "${version_major}" -gt "${gcs_version_major}" ]]; then
      : # fall out
    elif [[ "${version_minor}" -lt "${gcs_version_minor}" ]]; then
      greater=false
    elif [[ "${version_minor}" -gt "${gcs_version_minor}" ]]; then
      : # fall out
    elif [[ "${version_patch}" -lt "${gcs_version_patch}" ]]; then
      greater=false
    elif [[ "${version_patch}" -gt "${gcs_version_patch}" ]]; then
      : # fall out
    # Use lexicographic (instead of integer) comparison because
    # version_prerelease is a string, ("alpha" or "beta",) but first check if
    # either is an official release (i.e. empty prerelease string).
    #
    # We have to do this because lexicographically "beta" > "alpha" > "", but
    # we want official > beta > alpha.
    elif [[ -n "${version_prerelease}" && -z "${gcs_version_prerelease}" ]]; then
      greater=false
    elif [[ -z "${version_prerelease}" && -n "${gcs_version_prerelease}" ]]; then
      : # fall out
    elif [[ "${version_prerelease}" < "${gcs_version_prerelease}" ]]; then
      greater=false
    elif [[ "${version_prerelease}" > "${gcs_version_prerelease}" ]]; then
      : # fall out
    # Finally resort to -le here, since we want strictly-greater-than.
    elif [[ "${version_prerelease_rev}" -le "${gcs_version_prerelease_rev}" ]]; then
      greater=false
    fi

    if [[ "${greater}" != "true" ]]; then
      kube::log::status "${new_version} (just uploaded) <= ${gcs_version} (latest on GCS), not updating ${publish_file_dst}"
      return 1
    else
      kube::log::status "${new_version} (just uploaded) > ${gcs_version} (latest on GCS), updating ${publish_file_dst}"
    fi
  else  # gsutil cat failed; file does not exist
    kube::log::error "Release file '${publish_file_dst}' does not exist.  Continuing."
    return 0
  fi
}

# Check if the new version is greater than or equal to the version currently
# published on GCS.  (Ignore the build; if it's different, overwrite anyway.)
#
# Globals:
#   KUBE_GCS_PUBLISH_VERSION
#   KUBE_GCS_RELEASE_BUCKET
# Arguments:
#   publish_file: the GCS location to look in
# Returns:
#   If new version is greater than the GCS version
#
# TODO(16529): This should all be outside of build an in release, and should be
# refactored to reduce code duplication.  Also consider using strictly nested
# if and explicit handling of equals case.
function kube::release::gcs::verify_ci_ge() {
  local -r publish_file="${1-}"
  local -r new_version=${KUBE_GCS_PUBLISH_VERSION}
  local -r publish_file_dst="gs://${KUBE_GCS_RELEASE_BUCKET}/${publish_file}"

  kube::release::parse_and_validate_ci_version "${new_version}" || return 1

  local -r version_major="${VERSION_MAJOR}"
  local -r version_minor="${VERSION_MINOR}"
  local -r version_patch="${VERSION_PATCH}"
  local -r version_prerelease="${VERSION_PRERELEASE}"
  local -r version_prerelease_rev="${VERSION_PRERELEASE_REV}"
  local -r version_commits="${VERSION_COMMITS}"

  local gcs_version
  if gcs_version="$(gsutil cat "${publish_file_dst}")"; then
    kube::release::parse_and_validate_ci_version "${gcs_version}" || {
      kube::log::error "${publish_file_dst} contains invalid ci version, can't compare: '${gcs_version}'"
      return 1
    }

    local -r gcs_version_major="${VERSION_MAJOR}"
    local -r gcs_version_minor="${VERSION_MINOR}"
    local -r gcs_version_patch="${VERSION_PATCH}"
    local -r gcs_version_prerelease="${VERSION_PRERELEASE}"
    local -r gcs_version_prerelease_rev="${VERSION_PRERELEASE_REV}"
    local -r gcs_version_commits="${VERSION_COMMITS}"

    local greater=true
    if [[ "${version_major}" -lt "${gcs_version_major}" ]]; then
      greater=false
    elif [[ "${version_major}" -gt "${gcs_version_major}" ]]; then
      : # fall out
    elif [[ "${version_minor}" -lt "${gcs_version_minor}" ]]; then
      greater=false
    elif [[ "${version_minor}" -gt "${gcs_version_minor}" ]]; then
      : # fall out
    elif [[ "${version_patch}" -lt "${gcs_version_patch}" ]]; then
      greater=false
    elif [[ "${version_patch}" -gt "${gcs_version_patch}" ]]; then
      : # fall out
    # Use lexicographic (instead of integer) comparison because
    # version_prerelease is a string, ("alpha" or "beta")
    elif [[ "${version_prerelease}" < "${gcs_version_prerelease}" ]]; then
      greater=false
    elif [[ "${version_prerelease}" > "${gcs_version_prerelease}" ]]; then
      : # fall out
    elif [[ "${version_prerelease_rev}" -lt "${gcs_version_prerelease_rev}" ]]; then
      greater=false
    elif [[ "${version_prerelease_rev}" -gt "${gcs_version_prerelease_rev}" ]]; then
      : # fall out
    # If either version_commits is empty, it will be considered less-than, as
    # expected, (e.g. 1.2.3-beta < 1.2.3-beta.1).
    elif [[ "${version_commits}" -lt "${gcs_version_commits}" ]]; then
      greater=false
    fi

    if [[ "${greater}" != "true" ]]; then
      kube::log::status "${new_version} (just uploaded) < ${gcs_version} (latest on GCS), not updating ${publish_file_dst}"
      return 1
    else
      kube::log::status "${new_version} (just uploaded) >= ${gcs_version} (latest on GCS), updating ${publish_file_dst}"
    fi
  else  # gsutil cat failed; file does not exist
    kube::log::error "File '${publish_file_dst}' does not exist.  Continuing."
    return 0
  fi
}

# Publish a release to GCS: upload a version file, if KUBE_GCS_MAKE_PUBLIC,
# make it public, and verify the result.
#
# Globals:
#   KUBE_GCS_RELEASE_BUCKET
#   RELEASE_STAGE
#   KUBE_GCS_PUBLISH_VERSION
#   KUBE_GCS_MAKE_PUBLIC
# Arguments:
#   publish_file: the GCS location to look in
# Returns:
#   If new version is greater than the GCS version
function kube::release::gcs::publish() {
  local -r publish_file="${1-}"
  local -r publish_file_dst="gs://${KUBE_GCS_RELEASE_BUCKET}/${publish_file}"

  mkdir -p "${RELEASE_STAGE}/upload" || return 1
  echo "${KUBE_GCS_PUBLISH_VERSION}" > "${RELEASE_STAGE}/upload/latest" || return 1

  gsutil -m cp "${RELEASE_STAGE}/upload/latest" "${publish_file_dst}" || return 1

  local contents
  if [[ ${KUBE_GCS_MAKE_PUBLIC} =~ ^[yY]$ ]]; then
    kube::log::status "Making uploaded version file public and non-cacheable."
    gsutil acl ch -R -g all:R "${publish_file_dst}" >/dev/null 2>&1 || return 1
    gsutil setmeta -h "Cache-Control:private, max-age=0" "${publish_file_dst}" >/dev/null 2>&1 || return 1
    # If public, validate public link
    local -r public_link="https://storage.googleapis.com/${KUBE_GCS_RELEASE_BUCKET}/${publish_file}"
    kube::log::status "Validating uploaded version file at ${public_link}"
    contents="$(curl -s "${public_link}")"
  else
    # If not public, validate using gsutil
    kube::log::status "Validating uploaded version file at ${publish_file_dst}"
    contents="$(gsutil cat "${publish_file_dst}")"
  fi
  if [[ "${contents}" == "${KUBE_GCS_PUBLISH_VERSION}" ]]; then
    kube::log::status "Contents as expected: ${contents}"
  else
    kube::log::error "Expected contents of file to be ${KUBE_GCS_PUBLISH_VERSION}, but got ${contents}"
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Docker Release

# Releases all docker images to a docker registry specified by KUBE_DOCKER_REGISTRY
# using tag KUBE_DOCKER_IMAGE_TAG.
#
# Globals:
#   KUBE_DOCKER_REGISTRY
#   KUBE_DOCKER_IMAGE_TAG
#   KUBE_SERVER_PLATFORMS
# Returns:
#   If new pushing docker images was successful.
function kube::release::docker::release() {
  local binaries=(
    "kube-apiserver"
    "kube-controller-manager"
    "kube-scheduler"
    "kube-proxy"
    "hyperkube"
  )

  local docker_push_cmd=("${DOCKER[@]}")
  if [[ "${KUBE_DOCKER_REGISTRY}" == "gcr.io/"* ]]; then
    docker_push_cmd=("gcloud" "docker")
  fi

  if [[ "${KUBE_DOCKER_REGISTRY}" == "gcr.io/google_containers" ]]; then
    # Activate credentials for the k8s.production.user@gmail.com
    gcloud config set account k8s.production.user@gmail.com
  fi

  for arch in "${KUBE_SERVER_PLATFORMS[@]##*/}"; do
    for binary in "${binaries[@]}"; do

      local docker_target="${KUBE_DOCKER_REGISTRY}/${binary}-${arch}:${KUBE_DOCKER_IMAGE_TAG}"
      kube::log::status "Pushing ${binary} to ${docker_target}"
      "${docker_push_cmd[@]}" push "${docker_target}"

      # If we have a amd64 docker image. Tag it without -amd64 also and push it for compatibility with earlier versions
      if [[ ${arch} == "amd64" ]]; then
        local legacy_docker_target="${KUBE_DOCKER_REGISTRY}/${binary}:${KUBE_DOCKER_IMAGE_TAG}"

        "${DOCKER[@]}" tag -f "${docker_target}" "${legacy_docker_target}" 2>/dev/null

        kube::log::status "Pushing ${binary} to ${legacy_docker_target}"
        "${docker_push_cmd[@]}" push "${legacy_docker_target}"
      fi
    done
  done
  if [[ "${KUBE_DOCKER_REGISTRY}" == "gcr.io/google_containers" ]]; then
    # Activate default account
    gcloud config set account ${USER}@google.com
  fi
}

function kube::release::gcloud_account_is_active() {
  local -r account="${1-}"
  if [[ -n $(gcloud auth list --filter-account $account 2>/dev/null | grep "active") ]]; then
    return 0
  else
    return 1
  fi
}
