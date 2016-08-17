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

# This will canonicalize the path
KUBE_ROOT=$(cd $(dirname "${BASH_SOURCE}")/.. && pwd -P)

source "${KUBE_ROOT}/hack/lib/init.sh"

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

# Set KUBE_BUILD_PPC64LE to y to build for ppc64le in addition to other
# platforms.
# TODO(IBM): remove KUBE_BUILD_PPC64LE and reenable ppc64le compilation by
# default when
# https://github.com/kubernetes/kubernetes/issues/30384 and
# https://github.com/kubernetes/kubernetes/issues/25886 are fixed.
# The majority of the logic is in hack/lib/golang.sh.
readonly KUBE_BUILD_PPC64LE="${KUBE_BUILD_PPC64LE:-n}"

# Constants
readonly KUBE_BUILD_IMAGE_REPO=kube-build
readonly KUBE_BUILD_IMAGE_CROSS_TAG="$(cat ${KUBE_ROOT}/build/build-image/cross/VERSION)"
# KUBE_DATA_CONTAINER_NAME=kube-build-data-<hash>"

# This version number is used to cause everyone to rebuild their data containers
# and build image.  This is especially useful for automated build systems like
# Jenkins.
#
# Increment/change this number if you change the build image (anything
# under build/build-image, golang version) or change the set of volumes in the
# data container.
readonly KUBE_IMAGE_VERSION=5

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
readonly LOCAL_OUTPUT_GOPATH="${LOCAL_OUTPUT_SUBPATH}/go"
readonly LOCAL_OUTPUT_IMAGE_STAGING="${LOCAL_OUTPUT_ROOT}/images"

# This is a symlink to binaries for "this platform" (e.g. build tools).
readonly THIS_PLATFORM_BIN="${LOCAL_OUTPUT_ROOT}/bin"

readonly REMOTE_ROOT="/go/src/${KUBE_GO_PACKAGE}"
readonly REMOTE_OUTPUT_ROOT="${REMOTE_ROOT}/_output"
readonly REMOTE_OUTPUT_SUBPATH="${REMOTE_OUTPUT_ROOT}/dockerized"
readonly REMOTE_OUTPUT_BINPATH="${REMOTE_OUTPUT_SUBPATH}/bin"
readonly REMOTE_OUTPUT_GOPATH="${REMOTE_OUTPUT_SUBPATH}/go"

readonly KUBE_RSYNC_PORT="${KUBE_RSYNC_PORT:-}"

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
          kube-proxy,gcr.io/google_containers/debian-iptables-amd64:v4
        );;
    "arm")
        local targets=(
          kube-apiserver,armel/busybox
          kube-controller-manager,armel/busybox
          kube-scheduler,armel/busybox
          kube-proxy,gcr.io/google_containers/debian-iptables-arm:v4
        );;
    "arm64")
        local targets=(
          kube-apiserver,aarch64/busybox
          kube-controller-manager,aarch64/busybox
          kube-scheduler,aarch64/busybox
          kube-proxy,gcr.io/google_containers/debian-iptables-arm64:v4
        );;
    "ppc64le")
        local targets=(
          kube-apiserver,ppc64le/busybox
          kube-controller-manager,ppc64le/busybox
          kube-scheduler,ppc64le/busybox
          kube-proxy,gcr.io/google_containers/debian-iptables-ppc64le:v4
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
#   KUBE_BUILD_IMAGE_TAG_BASE
#   KUBE_BUILD_IMAGE_TAG
#   KUBE_BUILD_IMAGE
#   KUBE_BUILD_CONTAINER_NAME_BASE
#   KUBE_BUILD_CONTAINER_NAME
#   KUBE_DATA_CONTAINER_NAME_BASE
#   KUBE_DATA_CONTAINER_NAME
#   KUBE_RSYNC_CONTAINER_NAME_BASE
#   KUBE_RSYNC_CONTAINER_NAME
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

  if (( $KUBE_VERBOSE > 6 )); then
    kube::log::status "Docker Version:"
    "${DOCKER[@]}" version | kube::log::info_from_stdin
  fi

  KUBE_ROOT_HASH=$(kube::build::short_hash "${HOSTNAME:-}:${KUBE_ROOT}")
  KUBE_BUILD_IMAGE_TAG_BASE="build-${KUBE_ROOT_HASH}"
  KUBE_BUILD_IMAGE_TAG="${KUBE_BUILD_IMAGE_TAG_BASE}-${KUBE_IMAGE_VERSION}"
  KUBE_BUILD_IMAGE="${KUBE_BUILD_IMAGE_REPO}:${KUBE_BUILD_IMAGE_TAG}"
  KUBE_BUILD_CONTAINER_NAME_BASE="kube-build-${KUBE_ROOT_HASH}"
  KUBE_BUILD_CONTAINER_NAME="${KUBE_BUILD_CONTAINER_NAME_BASE}-${KUBE_IMAGE_VERSION}"
  KUBE_RSYNC_CONTAINER_NAME_BASE="kube-rsync-${KUBE_ROOT_HASH}"
  KUBE_RSYNC_CONTAINER_NAME="${KUBE_RSYNC_CONTAINER_NAME_BASE}-${KUBE_IMAGE_VERSION}"
  KUBE_DATA_CONTAINER_NAME_BASE="kube-build-data-${KUBE_ROOT_HASH}"
  KUBE_DATA_CONTAINER_NAME="${KUBE_DATA_CONTAINER_NAME_BASE}-${KUBE_IMAGE_VERSION}"
  DOCKER_MOUNT_ARGS=(--volumes-from "${KUBE_DATA_CONTAINER_NAME}")
  LOCAL_OUTPUT_BUILD_CONTEXT="${LOCAL_OUTPUT_IMAGE_STAGING}/${KUBE_BUILD_IMAGE}"
}

# ---------------------------------------------------------------------------
# Utility functions

function kube::build::docker_available_on_osx() {
  if [[ -z "${DOCKER_HOST}" ]]; then
    if [[ -S "/var/run/docker.sock" ]]; then
      kube::log::status "Using Docker for MacOS"
      return 0
    fi

    kube::log::status "No docker host is set. Checking options for setting one..."
    if [[ -z "$(which docker-machine)" ]]; then
      kube::log::status "It looks like you're running Mac OS X, yet neither Docker for Mac or docker-machine can be found."
      kube::log::status "See: https://docs.docker.com/engine/installation/mac/ for installation instructions."
      return 1
    elif [[ -n "$(which docker-machine)" ]]; then
      kube::build::prepare_docker_machine
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
  local docker_machine_out
  while ! docker_machine_out=$(docker-machine env "${DOCKER_MACHINE_NAME}" 2>&1); do
    if [[ ${docker_machine_out} =~ "Error checking TLS connection" ]]; then
      echo ${docker_machine_out}
      docker-machine regenerate-certs ${DOCKER_MACHINE_NAME}
    else
      sleep 1
    fi
  done
  eval $(docker-machine env "${DOCKER_MACHINE_NAME}")
  kube::log::status "A Docker host using docker-machine named '${DOCKER_MACHINE_NAME}' is ready to go!"
  return 0
}

function kube::build::is_osx() {
  [[ "$(uname)" == "Darwin" ]]
}

function kube::build::is_gnu_sed() {
  [[ $(sed --version 2>&1) == *GNU* ]]
}

function kube::build::update_dockerfile() {
  if kube::build::is_gnu_sed; then
    sed_opts=(-i)
  else
    sed_opts=(-i '')
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
    cat <<'EOF' >&2
Can't connect to 'docker' daemon.  please fix and retry.

Possible causes:
  - Docker Daemon not started
    - Linux: confirm via your init system
    - macOS w/ docker-machine: run `docker-machine ls` and `docker-machine start <name>`
    - macOS w/ Docker for Mac: Check the menu bar and start the Docker application
  - DOCKER_HOST hasn't been set of is set incorrectly
    - Linux: domain socket is used, DOCKER_* should be unset. In Bash run `unset ${!DOCKER_*}`
    - macOS w/ docker-machine: run `eval "$(docker-machine env <name>)"`
    - macOS w/ Docker for Mac: domain socket is used, DOCKER_* should be unset. In Bash run `unset ${!DOCKER_*}`
  - Other things to check:
    - Linux: User isn't in 'docker' group.  Add and relogin.
      - Something like 'sudo usermod -a -G docker ${USER}'
      - RHEL7 bug and workaround: https://bugzilla.redhat.com/show_bug.cgi?id=1119282#c8
EOF
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

  [[ $(docker images -q "${1}:${2}") ]]
}

# Delete all images that match a tag prefix except for the "current" version
#
# $1: The image repo/name
# $2: The tag base. We consider any image that matches $2*
# $3: The current image not to delete if provided
function kube::build::docker_delete_old_images() {
  # In Docker 1.12, we can replace this with
  #    docker images "$1" --format "{{.Tag}}"
  for tag in $(docker images kube-build | tail -n +2 | awk '{print $2}') ; do
    if [[ "$tag" != "$2"* ]] ; then
      V=6 kube::log::status "Keeping image $1:$tag"
      continue
    fi

    if [[ -z "${3:-}" || "$tag" != $3 ]] ; then
      V=2 kube::log::status "Deleting image $1:$tag"
      "${DOCKER[@]}" rmi "$1:$tag" >/dev/null
    else
      V=6 kube::log::status "Keeping image $1:$tag"
    fi
  done
}

# Stop and delete all containers that match a pattern
#
# $1: The base container prefix
# $2: The current container to keep, if provided
function kube::build::docker_delete_old_containers() {
  # In Docker 1.12 we can replace this line with
  #   docker ps -a --format="{{.Names}}"
  for container in $(docker ps -a | tail -n +2 | awk '{print $NF}') ; do
    if [[ "$container" != "$1"* ]] ; then
      V=6 kube::log::status "Keeping container $container"
      continue
    fi
    if [[ -z "${2:-}" || "$container" != "$2" ]] ; then
      V=2 kube::log::status "Deleting container $container"
      kube::build::destroy_container "$container"
    else
      V=6 kube::log::status "Keeping container $container"
    fi
  done
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


function kube::build::clean() {
  if kube::build::has_docker ; then
    kube::build::docker_delete_old_containers "${KUBE_BUILD_CONTAINER_NAME_BASE}"
    kube::build::docker_delete_old_containers "${KUBE_RSYNC_CONTAINER_NAME_BASE}"
    kube::build::docker_delete_old_containers "${KUBE_DATA_CONTAINER_NAME_BASE}"
    kube::build::docker_delete_old_images "${KUBE_BUILD_IMAGE_REPO}" "${KUBE_BUILD_IMAGE_TAG_BASE}"

    V=2 kube::log::status "Cleaning all untagged docker images"
    "${DOCKER[@]}" rmi $("${DOCKER[@]}" images -q --filter 'dangling=true') 2> /dev/null || true
  fi

  kube::log::status "Removing _output directory"
  rm -rf "${LOCAL_OUTPUT_ROOT}"
}

function kube::build::build_image_built() {
  kube::build::docker_image_exists "${KUBE_BUILD_IMAGE_REPO}" "${KUBE_BUILD_IMAGE_TAG}"
}

# Set up the context directory for the kube-build image and build it.
function kube::build::build_image() {
  if ! kube::build::build_image_built; then
    mkdir -p "${LOCAL_OUTPUT_BUILD_CONTEXT}"

    kube::version::get_version_vars
    kube::version::save_version_vars "${LOCAL_OUTPUT_BUILD_CONTEXT}/kube-version-defs"

    cp /etc/localtime "${LOCAL_OUTPUT_BUILD_CONTEXT}/"

    cp build/build-image/Dockerfile "${LOCAL_OUTPUT_BUILD_CONTEXT}/Dockerfile"
    cp build/build-image/rsyncd.sh "${LOCAL_OUTPUT_BUILD_CONTEXT}/"
    dd if=/dev/urandom bs=512 count=1 2>/dev/null | LC_ALL=C tr -dc 'A-Za-z0-9' | dd bs=32 count=1 2>/dev/null > "${LOCAL_OUTPUT_BUILD_CONTEXT}/rsyncd.password"
    chmod go= "${LOCAL_OUTPUT_BUILD_CONTEXT}/rsyncd.password"

    kube::build::update_dockerfile

    kube::build::docker_build "${KUBE_BUILD_IMAGE}" "${LOCAL_OUTPUT_BUILD_CONTEXT}" 'false'
  fi

  # Clean up old versions of everything
  kube::build::docker_delete_old_containers "${KUBE_BUILD_CONTAINER_NAME_BASE}" "${KUBE_BUILD_CONTAINER_NAME}"
  kube::build::docker_delete_old_containers "${KUBE_RSYNC_CONTAINER_NAME_BASE}" "${KUBE_RSYNC_CONTAINER_NAME}"
  kube::build::docker_delete_old_containers "${KUBE_DATA_CONTAINER_NAME_BASE}" "${KUBE_DATA_CONTAINER_NAME}"
  kube::build::docker_delete_old_images "${KUBE_BUILD_IMAGE_REPO}" "${KUBE_BUILD_IMAGE_TAG_BASE}" "${KUBE_BUILD_IMAGE_TAG}"

  kube::build::ensure_data_container
  kube::build::sync_to_container
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

  kube::log::status "Building Docker image ${image}"
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

function kube::build::ensure_data_container() {
  # If the data container exists AND exited successfully, we can use it.
  # Otherwise nuke it and start over.
  local ret=0
  local code=$(docker inspect \
      -f '{{.State.ExitCode}}' \
      "${KUBE_DATA_CONTAINER_NAME}" 2>/dev/null || ret=$?)
  if [[ "${ret}" == 0 && "${code}" != 0 ]]; then
    kube::build::destroy_container "${KUBE_DATA_CONTAINER_NAME}"
    ret=1
  fi
  if [[ "${ret}" != 0 ]]; then
    kube::log::status "Creating data container ${KUBE_DATA_CONTAINER_NAME}"
    # We have to ensure the directory exists, or else the docker run will
    # create it as root.
    mkdir -p "${LOCAL_OUTPUT_GOPATH}"
    # We want this to run as root to be able to chown, so non-root users can
    # later use the result as a data container.  This run both creates the data
    # container and chowns the GOPATH.
    #
    # The data container creates volumes for all of the directories that store
    # intermediates for the Go build. This enables incremental builds across
    # Docker sessions. The *_cgo paths are re-compiled versions of the go std
    # libraries for true static building.
    local -ra docker_cmd=(
      "${DOCKER[@]}" run
      --volume "${REMOTE_ROOT}"   # white-out the whole output dir
      --volume /usr/local/go/pkg/linux_386_cgo
      --volume /usr/local/go/pkg/linux_amd64_cgo
      --volume /usr/local/go/pkg/linux_arm_cgo
      --volume /usr/local/go/pkg/linux_arm64_cgo
      --volume /usr/local/go/pkg/linux_ppc64le_cgo
      --volume /usr/local/go/pkg/darwin_amd64_cgo
      --volume /usr/local/go/pkg/darwin_386_cgo
      --volume /usr/local/go/pkg/windows_amd64_cgo
      --volume /usr/local/go/pkg/windows_386_cgo
      --name "${KUBE_DATA_CONTAINER_NAME}"
      --hostname "${HOSTNAME}"
      "${KUBE_BUILD_IMAGE}"
      chown -R $(id -u).$(id -g)
        "${REMOTE_ROOT}"
        /usr/local/go/pkg/
    )
    "${docker_cmd[@]}"
  fi
}

# Run a command in the kube-build image.  This assumes that the image has
# already been built.  This will sync out all output data from the build.
function kube::build::run_build_command() {
  kube::log::status "Running build command..."
  kube::build::run_build_command_ex "${KUBE_BUILD_CONTAINER_NAME}" -- "$@"
}

# Run a command in the kube-build image.  This assumes that the image has
# already been built.  This will sync out all output data from the build.
#
# Arguments are in the form of
#  <container name> <extra docker args> -- <command>
function kube::build::run_build_command_ex() {
  [[ $# != 0 ]] || { echo "Invalid input - please specify a the container name." >&2; return 4; }
  local container_name="${1}"
  shift

  local -a docker_run_opts=(
    "--name=${container_name}"
    "--user=$(id -u):$(id -g)"
    "--hostname=${HOSTNAME}"
    "${DOCKER_MOUNT_ARGS[@]}"
  )

  local detach=false

  [[ $# != 0 ]] || { echo "Invalid input - please specify docker arguments followed by --." >&2; return 4; }
  # Everything before "--" is an arg to docker
  until [ -z "${1-}" ] ; do
    if [[ "$1" == "--" ]]; then
      shift
      break
    fi
    docker_run_opts+=("$1")
    if [[ "$1" == "-d" || "$1" == "--detach" ]] ; then
      detach=true
    fi
    shift
  done

  # Everything after "--" is the command to run
  [[ $# != 0 ]] || { echo "Invalid input - please specify a command to run." >&2; return 4; }
  local -a cmd=()
  until [ -z "${1-}" ] ; do
    cmd+=("$1")
    shift
  done

  if [ -n "${KUBERNETES_CONTRIB:-}" ]; then
    docker_run_opts+=(-e "KUBERNETES_CONTRIB=${KUBERNETES_CONTRIB}")
  fi

  docker_run_opts+=(
    --env "KUBE_FASTBUILD=${KUBE_FASTBUILD:-false}"
    --env "KUBE_BUILDER_OS=${OSTYPE:-notdetected}"
    --env "KUBE_BUILD_PPC64LE=${KUBE_BUILD_PPC64LE}"  # TODO(IBM): remove
  )

  # If we have stdin we can run interactive.  This allows things like 'shell.sh'
  # to work.  However, if we run this way and don't have stdin, then it ends up
  # running in a daemon-ish mode.  So if we don't have a stdin, we explicitly
  # attach stderr/stdout but don't bother asking for a tty.
  if [[ -t 0 ]]; then
    docker_run_opts+=(--interactive --tty)
  elif [[ "$detach" == false ]]; then
    docker_run_opts+=(--attach=stdout --attach=stderr)
  fi

  local -ra docker_cmd=(
    "${DOCKER[@]}" run "${docker_run_opts[@]}" "${KUBE_BUILD_IMAGE}")

  # Clean up container from any previous run
  kube::build::destroy_container "${KUBE_BUILD_CONTAINER_NAME}"
  "${docker_cmd[@]}" "${cmd[@]}"
  kube::build::destroy_container "${KUBE_BUILD_CONTAINER_NAME}"
}

function kube::build::probe_address {
  # Apple has an ancient version of netcat with custom timeout flags.  This is
  # the best way I (jbeda) could find to test for that.
  local nc
  if nc 2>&1 | grep -e 'apple' >/dev/null ; then
    netcat="nc -G 1"
  else
    netcat="nc -w 1"
  fi

  # Wait unil rsync is up and running.
  if ! which nc >/dev/null ; then
    V=6 kube::log::info "netcat not installed, waiting for 1s"
    sleep 1
    return 0
  fi

  local tries=10
  while (( $tries > 0 )) ; do
    if $netcat -z "$1" "$2" 2> /dev/null ; then
      return 0
    fi
    tries=$(($tries-1))
    sleep 0.1
  done

  return 1
}

function kube::build::start_rsyncd_container() {
  kube::build::stop_rsyncd_container
  V=6 kube::log::status "Starting rsyncd container"
  kube::build::run_build_command_ex \
    "${KUBE_RSYNC_CONTAINER_NAME}" -p 127.0.0.1:${KUBE_RSYNC_PORT}:8730 -d \
    -- /rsyncd.sh >/dev/null

  local mapped_port
  if ! mapped_port=$("${DOCKER[@]}" port "${KUBE_RSYNC_CONTAINER_NAME}" 8730 2> /dev/null | cut -d: -f 2) ; then
    kube:log:error "Could not get effective rsync port"
    return 1
  fi

  local container_ip
  container_ip=$("${DOCKER[@]}" inspect --format '{{ .NetworkSettings.IPAddress }}' "${KUBE_RSYNC_CONTAINER_NAME}")

  if kube::build::probe_address 127.0.0.1 ${mapped_port}; then
    KUBE_RSYNC_ADDR="127.0.0.1:${mapped_port}"
    sleep 0.5
    return 0
  elif kube::build::probe_address "${container_ip}" 8730; then
    KUBE_RSYNC_ADDR="${container_ip}:8730"
    sleep 0.5
    return 0
  fi

  kube::log::error "Could not connect to rsync container. See build/README.md for setting up remote Docker engine."
  return 1
}

function kube::build::stop_rsyncd_container() {
  V=6 kube::log::status "Stopping any currently running rsyncd container"
  kube::build::destroy_container "${KUBE_RSYNC_CONTAINER_NAME}"
}

# This will launch rsyncd in a container and then sync the source tree to the
# container over the local network.
function kube::build::sync_to_container() {
  kube::log::status "Syncing sources to container"

  kube::build::start_rsyncd_container

  local rsync_extra=""
  if (( $KUBE_VERBOSE >= 6 )); then
    rsync_extra="-iv"
  fi

  V=6 kube::log::status "Running rsync"
  rsync $rsync_extra \
    --archive \
    --prune-empty-dirs \
    --password-file="${LOCAL_OUTPUT_BUILD_CONTEXT}/rsyncd.password" \
    --filter='- /.git/' \
    --filter='- /.make/' \
    --filter='- /_tmp/' \
    --filter='- /_output/' \
    --filter='- /' \
    "${KUBE_ROOT}/" "rsync://k8s@${KUBE_RSYNC_ADDR}/k8s/"

  kube::build::stop_rsyncd_container
}

# If the Docker server is remote, copy the results back out.
function kube::build::copy_output() {
  kube::log::status "Syncing out of container"

  kube::build::start_rsyncd_container

  local rsync_extra=""
  if (( $KUBE_VERBOSE >= 6 )); then
    rsync_extra="-iv"
  fi

  # The filter syntax for rsync is a little obscure. It filters on files and
  # directories.  If you don't go in to a directory you won't find any files
  # there.  Rules are evaluated in order.  The last two rules are a little
  # magic. '+ */' says to go in to every directory and '- /**' says to ignore
  # any file or directory that isn't already specifically allowed.
  #
  # We are looking to copy out all of the built binaries along with various
  # generated files.
  V=6 kube::log::status "Running rsync"
  rsync $rsync_extra \
    --archive \
    --prune-empty-dirs \
    --password-file="${LOCAL_OUTPUT_BUILD_CONTEXT}/rsyncd.password" \
    --filter='- /vendor/' \
    --filter='- /_temp/' \
    --filter='+ /_output/dockerized/bin/**' \
    --filter='+ zz_generated.*' \
    --filter='+ */' \
    --filter='- /**' \
    "rsync://k8s@${KUBE_RSYNC_ADDR}/k8s/" "${KUBE_ROOT}"

  kube::build::stop_rsyncd_container
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

    cp "${RELEASE_DIR}/kubernetes-src.tar.gz" "${release_stage}/"

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
  cp "${salt_dir}/cluster-autoscaler/cluster-autoscaler.manifest" "${dst_dir}/"
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
  cp "${salt_dir}/l7-gcp/glbc.manifest" "${dst_dir}"
  cp "${salt_dir}/rescheduler/rescheduler.manifest" "${dst_dir}/"
  cp "${KUBE_ROOT}/cluster/gce/trusty/configure-helper.sh" "${dst_dir}/trusty-configure-helper.sh"
  cp "${KUBE_ROOT}/cluster/gce/gci/configure-helper.sh" "${dst_dir}/gci-configure-helper.sh"
  cp "${KUBE_ROOT}/cluster/gce/gci/health-monitor.sh" "${dst_dir}/health-monitor.sh"
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

  # Include only federation/cluster, federation/manifests and federation/deploy
  mkdir "${release_stage}/federation"
  cp -R "${KUBE_ROOT}/federation/cluster" "${release_stage}/federation/"
  cp -R "${KUBE_ROOT}/federation/manifests" "${release_stage}/federation/"
  cp -R "${KUBE_ROOT}/federation/deploy" "${release_stage}/federation/"

  cp -R "${KUBE_ROOT}/examples" "${release_stage}/"
  cp -R "${KUBE_ROOT}/docs" "${release_stage}/"
  cp "${KUBE_ROOT}/README.md" "${release_stage}/"
  cp "${KUBE_ROOT}/Godeps/LICENSES" "${release_stage}/"
  cp "${KUBE_ROOT}/Vagrantfile" "${release_stage}/"

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
    GCLOUD_ACCOUNT=$(gcloud config list --format='value(core.account)' 2>/dev/null)
  fi
  if [[ -z "${GCLOUD_ACCOUNT-}" ]]; then
    echo "No account authorized through gcloud.  Please fix with:"
    echo
    echo "  gcloud auth login"
    return 1
  fi

  if [[ -z "${GCLOUD_PROJECT-}" ]]; then
    GCLOUD_PROJECT=$(gcloud config list --format='value(core.project)' 2>/dev/null)
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

  # Having the configure-vm.sh script and GCI code from the GCE cluster
  # deploy hosted with the release is useful for GKE.
  kube::release::gcs::stage_and_hash "${RELEASE_STAGE}/full/kubernetes/cluster/gce/configure-vm.sh" extra/gce || return 1
  kube::release::gcs::stage_and_hash "${RELEASE_STAGE}/full/kubernetes/cluster/gce/gci/node.yaml" extra/gce || return 1
  kube::release::gcs::stage_and_hash "${RELEASE_STAGE}/full/kubernetes/cluster/gce/gci/master.yaml" extra/gce || return 1
  kube::release::gcs::stage_and_hash "${RELEASE_STAGE}/full/kubernetes/cluster/gce/gci/configure.sh" extra/gce || return 1

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

  if [[ -n "${KUBE_GCS_RELEASE_BUCKET_MIRROR:-}" ]] &&
     [[ "${KUBE_GCS_RELEASE_BUCKET_MIRROR}" != "${KUBE_GCS_RELEASE_BUCKET}" ]]; then
    local -r gcs_mirror="gs://${KUBE_GCS_RELEASE_BUCKET_MIRROR}/${KUBE_GCS_RELEASE_PREFIX}"
    kube::log::status "Mirroring build to ${gcs_mirror}"
    gsutil -q -m "${gcs_options[@]+${gcs_options[@]}}" rsync -d -r "${gcs_destination}" "${gcs_mirror}" || return 1
    if [[ ${KUBE_GCS_MAKE_PUBLIC} =~ ^[yY]$ ]]; then
      kube::log::status "Marking all uploaded mirror objects public"
      gsutil -q -m acl ch -R -g all:R "${gcs_mirror}" >/dev/null 2>&1 || return 1
    fi
  fi
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

  kube::release::gcs::publish_to_bucket "${KUBE_GCS_RELEASE_BUCKET}" "${publish_file}" || return 1

  if [[ -n "${KUBE_GCS_RELEASE_BUCKET_MIRROR:-}" ]] &&
     [[ "${KUBE_GCS_RELEASE_BUCKET_MIRROR}" != "${KUBE_GCS_RELEASE_BUCKET}" ]]; then
    kube::release::gcs::publish_to_bucket "${KUBE_GCS_RELEASE_BUCKET_MIRROR}" "${publish_file}" || return 1
  fi
}


function kube::release::gcs::publish_to_bucket() {
  local -r publish_bucket="${1}"
  local -r publish_file="${2}"
  local -r publish_file_dst="gs://${publish_bucket}/${publish_file}"

  mkdir -p "${RELEASE_STAGE}/upload" || return 1
  echo "${KUBE_GCS_PUBLISH_VERSION}" > "${RELEASE_STAGE}/upload/latest" || return 1

  gsutil -m cp "${RELEASE_STAGE}/upload/latest" "${publish_file_dst}" || return 1

  local contents
  if [[ ${KUBE_GCS_MAKE_PUBLIC} =~ ^[yY]$ ]]; then
    kube::log::status "Making uploaded version file public and non-cacheable."
    gsutil acl ch -R -g all:R "${publish_file_dst}" >/dev/null 2>&1 || return 1
    gsutil setmeta -h "Cache-Control:private, max-age=0" "${publish_file_dst}" >/dev/null 2>&1 || return 1
    # If public, validate public link
    local -r public_link="https://storage.googleapis.com/${publish_bucket}/${publish_file}"
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

      # TODO(IBM): Enable hyperkube builds for ppc64le again
      if [[ ${binary} != "hyperkube" || ${arch} != "ppc64le" ]]; then

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
  if [[ "$(gcloud config list --format='value(core.account)')" == "${account}" ]]; then
    return 0
  else
    return 1
  fi
}
